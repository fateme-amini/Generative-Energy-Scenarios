# ======================================================================================
#
#  Publication-Ready Framework for Wind Scenario Generation and Evaluation
#  (Version 2.0 with Enhanced Metrics)
#
#  This script provides a rigorous framework for comparing generative models for
#  wind power time series. It incorporates a strict temporal train-test split and
#  a suite of quantitative metrics to ensure robust and defensible results.
#
#  Author: Fatemeh Amini
#  Date: 2025/09/12
#
# ======================================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from copulas.multivariate import GaussianMultivariate
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# --- Quantitative Metrics Imports ---
from scipy.stats import wasserstein_distance
from scipy.fft import rfft # For Power Spectral Density
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- Suppress Warnings for cleaner output ---
warnings.filterwarnings("ignore")

# ======================================================================================
# --- [1] Configuration and Setup ---
# ======================================================================================
print("--- [1] Initializing Configuration ---")
CONFIG = {
    # --- Data and Preprocessing ---
    "data_file": "T1.csv",
    "power_column": "LV ActivePower (kW)",
    "date_column": "Date/Time",
    "date_format": "%d %m %Y %H:%M",
    "sequence_length": 48,      # 48 hours (2 days)
    "train_split_ratio": 0.8,   # 80% for training, 20% for testing (chronological)
    "n_scenarios_to_generate": 1000,
    "max_sequences_for_training": 10000, # Limit for faster training on large datasets

    # --- GAN Hyperparameters ---
    "batch_size": 256,
    "n_epochs": 300,
    "latent_dim": 100,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_critic": 5,             # Train critic 5 times per generator update
    "lambda_gp": 10,           # Gradient penalty coefficient
    "hidden_dim": 128,         # LSTM hidden dimension
    "n_layers": 2,             # Number of LSTM layers

    # --- ARIMA Configuration ---
    "arima_order": (5, 1, 2),  # (p, d, q) order for ARIMA model
    "arima_training_points": 365 * 24, # Use one year of recent data for fitting

    # --- File Paths ---
    "generator_save_path": "generator_wind_model.pth"
}

# --- Plotting Aesthetics ---
plt.style.use('seaborn-v0_8-whitegrid')
PLOT_COLORS = {"real": "#003f5c", "RNN-GAN": "#7a5195", "Copula": "#ef5675", "ARIMA": "#ffa600", "Gaussian MC": "#00aaff"}

# --- Device and Parallelization Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
n_cpu_workers = os.cpu_count()
print(f"Using device: {device} with {n_gpus} GPU(s) and {n_cpu_workers} CPU core(s).")


# ======================================================================================
# --- [2] Data Preparation with Temporal Splitting ---
# ======================================================================================
def prepare_datasets(config):
    """
    Loads, preprocesses, and splits the data into training and testing sets
    based on a chronological split to prevent data leakage.
    """
    print(f"--- [2] Loading and Preparing Datasets from {config['data_file']} ---")
    df = pd.read_csv(
        config["data_file"],
        usecols=[config["date_column"], config["power_column"]],
        parse_dates=[config["date_column"]],
        date_format=config["date_format"],
        index_col=config["date_column"],
        low_memory=False
    )
    # Handle non-numeric values and resample to a consistent hourly frequency
    power_series = pd.to_numeric(df[config["power_column"]], errors='coerce').fillna(0)
    power_hourly = power_series.resample('h').mean().fillna(method='ffill')
    print(f"Data resampled to hourly. Total hours: {len(power_hourly)}")

    # Create overlapping sequences of the specified length
    raw_values = power_hourly.values
    sequences = np.array([raw_values[i:i + config["sequence_length"]] for i in range(len(raw_values) - config["sequence_length"] + 1)])

    # Chronological split: first 80% for training, last 20% for testing
    split_index = int(len(sequences) * config["train_split_ratio"])
    train_sequences_raw = sequences[:split_index]
    test_sequences_raw = sequences[split_index:]
    print(f"Chronological Split: {len(train_sequences_raw)} training sequences, {len(test_sequences_raw)} testing sequences.")

    # Normalize training data for GAN. Scaler is fit ONLY on training data.
    gan_scaler = MinMaxScaler(feature_range=(-1, 1))
    gan_scaler.fit(train_sequences_raw.reshape(-1, 1))
    train_sequences_normalized = gan_scaler.transform(train_sequences_raw.reshape(-1, 1)).reshape(train_sequences_raw.shape)
    # Add a channel dimension for PyTorch compatibility (N, Seq_Len, 1)
    train_sequences_normalized = np.expand_dims(train_sequences_normalized, axis=-1)

    return train_sequences_raw, test_sequences_raw, train_sequences_normalized, gan_scaler, power_hourly


# ======================================================================================
# --- [3] Generative Model Definitions ---
# ======================================================================================
class GeneratorRNN(nn.Module):
    """RNN-based Generator: Expands latent vector into a time series."""
    def __init__(self, latent_dim, hidden_dim, seq_len, n_layers):
        super(GeneratorRNN, self).__init__()
        # LSTM layer learns temporal patterns
        self.lstm = nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True)
        # Linear layer maps hidden state to output value at each time step
        self.linear = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh() # Tanh activation matches the (-1, 1) scaling
    def forward(self, z):
        # Repeat the latent vector for each time step to feed into the LSTM
        z_seq = z.unsqueeze(1).repeat(1, CONFIG["sequence_length"], 1)
        lstm_out, _ = self.lstm(z_seq)
        out = self.linear(lstm_out)
        return self.tanh(out)

class CriticRNN(nn.Module):
    """RNN-based Critic: Evaluates the realism of a time series."""
    def __init__(self, hidden_dim, n_layers):
        super(CriticRNN, self).__init__()
        # LSTM layer processes the sequence
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        # Linear layer outputs a single score based on the final hidden state
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, seq):
        _, (h_n, _) = self.lstm(seq)
        # Use the last hidden state of the last layer as the sequence representation
        return self.linear(h_n[-1])

def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty for WGAN-GP to enforce Lipschitz constraint."""
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_and_generate_rnn_wgan(train_data_normalized, scaler, config):
    """Main function to train the RNN-WGAN-GP and generate final scenarios."""
    print("\n--- [3a] Training RNN-WGAN-GP ---")
    # Subsample training data if it's too large, for faster epochs
    if len(train_data_normalized) > config["max_sequences_for_training"]:
        indices = np.random.choice(len(train_data_normalized), config["max_sequences_for_training"], replace=False)
        train_data_normalized = train_data_normalized[indices]

    dataloader = DataLoader(
        TensorDataset(torch.FloatTensor(train_data_normalized)),
        batch_size=config["batch_size"] * max(1, n_gpus),
        shuffle=True,
        drop_last=True,
        num_workers=n_cpu_workers // 2 if n_cpu_workers else 0,
        pin_memory=True
    )

    generator = GeneratorRNN(config["latent_dim"], config["hidden_dim"], config["sequence_length"], config["n_layers"]).to(device)
    critic = CriticRNN(config["hidden_dim"], config["n_layers"]).to(device)

    # Use DataParallel for multi-GPU training
    if n_gpus > 1:
        generator = nn.DataParallel(generator)
        critic = nn.DataParallel(critic)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))

    # --- Training Loop ---
    for epoch in tqdm(range(config["n_epochs"]), desc="GAN Training"):
        for i, (real_seqs,) in enumerate(dataloader):
            real_seqs = real_seqs.to(device)
            
            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()
            z = torch.randn(real_seqs.size(0), config["latent_dim"], device=device)
            fake_seqs = generator(z).detach()
            
            # WGAN-GP loss: maximize (critic(real) - critic(fake)) + gradient penalty
            c_loss = -torch.mean(critic(real_seqs)) + torch.mean(critic(fake_seqs))
            gp = compute_gradient_penalty(critic, real_seqs.data, fake_seqs.data)
            c_loss_final = c_loss + config["lambda_gp"] * gp
            
            c_loss_final.backward()
            optimizer_C.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            # Update generator less frequently than the critic
            if i % config["n_critic"] == 0:
                optimizer_G.zero_grad()
                gen_z = torch.randn(real_seqs.size(0), config["latent_dim"], device=device)
                # Generator loss: maximize critic(fake) to fool the critic
                g_loss = -torch.mean(critic(generator(gen_z)))
                
                g_loss.backward()
                optimizer_G.step()
    
    # Save the trained generator model
    print(f"\n--- Training complete. Saving generator model to {config['generator_save_path']} ---")
    model_to_save = generator.module if isinstance(generator, nn.DataParallel) else generator
    torch.save(model_to_save.state_dict(), config['generator_save_path'])

    # --- Generate Scenarios ---
    print("--- Generating scenarios from trained model ---")
    model_to_eval = generator.module if isinstance(generator, nn.DataParallel) else generator
    model_to_eval.eval()
    with torch.no_grad():
        z_final = torch.randn(config["n_scenarios_to_generate"], config["latent_dim"], device=device)
        generated_normalized = model_to_eval(z_final).cpu().numpy()
    
    # Inverse transform to get real power values
    generated_reshaped = generated_normalized.reshape(-1, 1)
    generated_inversed = scaler.inverse_transform(generated_reshaped)
    return generated_inversed.reshape(config["n_scenarios_to_generate"], config["sequence_length"])

def _fit_and_simulate_arima_batch(train_data, n_sims, seq_len, order):
    """Helper function to fit and simulate ARIMA in a single process."""
    model = ARIMA(train_data, order=order).fit()
    return [model.simulate(nsimulations=seq_len) for _ in range(n_sims)]

def generate_arima_scenarios(train_series_full, config):
    """Generates ARIMA scenarios in parallel to save time."""
    print("\n--- [3b] Generating ARIMA Scenarios (Parallelized) ---")
    train_data = train_series_full.values[-config["arima_training_points"]:]
    # Divide the total number of simulations among available CPU workers
    sims_per_worker = int(np.ceil(config["n_scenarios_to_generate"] / n_cpu_workers))
    results_nested = Parallel(n_jobs=n_cpu_workers)(
        delayed(_fit_and_simulate_arima_batch)(
            train_data, sims_per_worker, config["sequence_length"], config["arima_order"]
        ) for _ in tqdm(range(n_cpu_workers), desc=f"ARIMA on {n_cpu_workers} cores")
    )
    simulations = [item for sublist in results_nested for item in sublist]
    # Ensure no negative power values and trim to the exact number required
    return np.maximum(np.array(simulations)[:config["n_scenarios_to_generate"]], 0)

def generate_copula_scenarios(train_sequences, config):
    """Generates scenarios using a Gaussian Copula to model dependencies."""
    print("\n--- [3c] Generating Copula Scenarios ---")
    df = pd.DataFrame(train_sequences, columns=[f'h_{i}' for i in range(config["sequence_length"])])
    copula = GaussianMultivariate()
    copula.fit(df)
    return copula.sample(config["n_scenarios_to_generate"]).values

def generate_gaussian_mc_scenarios(train_sequences, config):
    """Generates scenarios from a multivariate normal distribution (classic Monte Carlo)."""
    print("\n--- [3d] Generating Gaussian Monte Carlo Scenarios ---")
    mean_vector = np.mean(train_sequences, axis=0)
    cov_matrix = np.cov(train_sequences, rowvar=False)
    generated = np.random.multivariate_normal(mean_vector, cov_matrix, config["n_scenarios_to_generate"])
    # Ensure no negative power values
    return np.maximum(generated, 0)


# ======================================================================================
# --- [4] Quantitative Evaluation Framework ---
# ======================================================================================
def _dtw_distance(gen_sample, real_samples):
    """Calculates the DTW distance from one generated sample to the closest real sample."""
    # Use Manhattan distance (dist=1) for 1-D time series.
    return min(fastdtw(gen_sample, real_sample, dist=1)[0] for real_sample in real_samples)

def calculate_metrics(real_data, generated_data, model_name):
    """Calculates an expanded dictionary of metrics for a given model."""
    print(f"--- [4] Calculating metrics for {model_name} ---")
    metrics = {}
    
    # 1. Marginal Distribution: Wasserstein Distance
    metrics['Wasserstein Dist.'] = wasserstein_distance(real_data.flatten(), generated_data.flatten())
    
    # 2. Temporal Correlation: ACF Error
    nlags = real_data.shape[1] - 1
    def safe_acf(series):
        # Handle cases with zero variance which would cause acf to fail
        if np.var(series) < 1e-6:
            return np.full(nlags + 1, np.nan)
        return acf(series, nlags=nlags, fft=True)

    acf_real_mean = np.nanmean([safe_acf(s) for s in real_data], axis=0)
    acf_gen_mean = np.nanmean([safe_acf(s) for s in generated_data], axis=0)
    if np.isnan(acf_real_mean).any() or np.isnan(acf_gen_mean).any():
        metrics['ACF MAE'] = np.nan
    else:
        metrics['ACF MAE'] = mean_absolute_error(acf_real_mean, acf_gen_mean)
        
    # 3. Covariance Structure: Frobenius Norm of the difference in covariance matrices
    cov_real = np.cov(real_data, rowvar=False)
    cov_gen = np.cov(generated_data, rowvar=False)
    metrics['Covariance F-Norm'] = np.linalg.norm(cov_real - cov_gen, 'fro')

    # 4. Ramp Rate Distribution (using Wasserstein Distance)
    real_ramps = np.diff(real_data, axis=1).flatten()
    gen_ramps = np.diff(generated_data, axis=1).flatten()
    metrics['Ramp Wass. Dist.'] = wasserstein_distance(real_ramps, gen_ramps)

    # 5. Power Spectral Density (PSD) Error (Frequency domain analysis)
    def get_mean_psd(data):
        # Calculate the absolute value of the real Fast Fourier Transform
        psd_amplitudes = [np.abs(rfft(s)) for s in data]
        return np.mean(psd_amplitudes, axis=0)
    
    psd_real = get_mean_psd(real_data)
    psd_gen = get_mean_psd(generated_data)
    metrics['PSD MSE'] = mean_squared_error(psd_real, psd_gen)

    # 6. Dynamic Time Warping (DTW) Distance (Parallelized for speed)
    dtw_distances = Parallel(n_jobs=n_cpu_workers)(
        delayed(_dtw_distance)(gen_sample, real_data) 
        for gen_sample in tqdm(generated_data, desc=f"DTW for {model_name}", leave=False)
    )
    metrics['Avg. DTW'] = np.mean(dtw_distances)
    
    return metrics


# ======================================================================================
# --- [5] Visualization (MODIFIED FOR SEPARATE PLOTS) ---
# ======================================================================================
def plot_mean_profile(real_data, results_dict):
    """Generates and saves the Hourly Mean Profile plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.mean(real_data, axis=0), color=PLOT_COLORS["real"], label="Real Test Data", linewidth=3, zorder=10)
    for name, data in results_dict.items():
        ax.plot(np.mean(data, axis=0), label=name, color=PLOT_COLORS.get(name.replace(" ", "-")), linestyle='--')
    ax.set_title('Figure 1: Hourly Mean Profile', fontsize=16, weight='bold')
    ax.set_xlabel('Hour', fontsize=12); ax.set_ylabel('Power (kW)', fontsize=12)
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("figure_1_mean_profile.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

def plot_std_dev(real_data, results_dict):
    """Generates and saves the Hourly Standard Deviation plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.std(real_data, axis=0), color=PLOT_COLORS["real"], label="Real Test Data", linewidth=3, zorder=10)
    for name, data in results_dict.items():
        ax.plot(np.std(data, axis=0), label=name, color=PLOT_COLORS.get(name.replace(" ", "-")), linestyle='--')
    ax.set_title('Figure 2: Hourly Standard Deviation', fontsize=16, weight='bold')
    ax.set_xlabel('Hour', fontsize=12); ax.set_ylabel('Power Std. Dev. (kW)', fontsize=12)
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("figure_2_std_dev.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

def plot_acf(real_data, results_dict):
    """Generates and saves the Average Autocorrelation plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nlags_for_plot = CONFIG["sequence_length"] - 1
    
    def get_mean_acf(dataset):
        return np.nanmean([
            acf(s, nlags=nlags_for_plot, fft=True) if np.var(s) > 1e-6 
            else np.full(nlags_for_plot + 1, np.nan) 
            for s in dataset
        ], axis=0)

    ax.plot(get_mean_acf(real_data), color=PLOT_COLORS["real"], label="Real ACF", linewidth=3, marker='o', markersize=4, zorder=10)
    for name, data in results_dict.items():
        ax.plot(get_mean_acf(data), label=f'{name} ACF', color=PLOT_COLORS.get(name.replace(" ", "-")), linestyle='--', alpha=0.9)
        
    ax.set_title('Figure 3: Average Autocorrelation (ACF)', fontsize=16, weight='bold')
    ax.set_xlabel('Lag (Hours)', fontsize=12); ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("figure_3_acf.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

def plot_value_distribution(real_data, results_dict):
    """Generates and saves the Overall Value Distribution violin plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    df_dist = pd.DataFrame([{'Power (kW)': v, 'Type': 'Real'} for v in real_data.flatten()])
    for name, data in results_dict.items():
        df_dist = pd.concat([df_dist, pd.DataFrame([{'Power (kW)': v, 'Type': name} for v in data.flatten()])], ignore_index=True)
    
    palette = {name: PLOT_COLORS.get(name.replace(" ", "-")) for name in df_dist['Type'].unique()}
    palette['Real'] = PLOT_COLORS['real']

    sns.violinplot(data=df_dist, x='Type', y='Power (kW)', ax=ax, palette=palette)
    ax.set_title('Figure 4: Overall Value Distribution', fontsize=16, weight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlabel('Model Type', fontsize=12); ax.set_ylabel('Power (kW)', fontsize=12)
    plt.savefig("figure_4_value_distribution.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

def plot_ramp_rate(real_data, results_dict):
    """Generates and saves the Ramp Rate Distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(np.diff(real_data, axis=1).flatten(), ax=ax, color=PLOT_COLORS["real"], label="Real", linewidth=3, bw_adjust=1.5, zorder=10)
    for name, data in results_dict.items():
        sns.kdeplot(np.diff(data, axis=1).flatten(), ax=ax, color=PLOT_COLORS.get(name.replace(" ", "-")), label=name, linestyle='--', bw_adjust=1.5)
    ax.set_title('Figure 5: Ramp Rate Distribution', fontsize=16, weight='bold')
    ax.set_xlabel('Power Change (kW/hr)', fontsize=12); ax.set_ylabel('Density', fontsize=12)
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("figure_5_ramp_rate_distribution.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

# ======================================================================================
# --- Main Execution Workflow ---
# ======================================================================================
if __name__ == "__main__":
    # 1. Load and split data
    train_raw, test_raw, train_norm, gan_scaler, full_series = prepare_datasets(CONFIG)

    # 2. Train and generate scenarios from all models
    results = {}
    results['RNN-GAN'] = train_and_generate_rnn_wgan(train_norm, gan_scaler, CONFIG)
    results['ARIMA'] = generate_arima_scenarios(full_series[:len(train_raw) * CONFIG['sequence_length']], CONFIG)
    results['Copula'] = generate_copula_scenarios(train_raw, CONFIG)
    results['Gaussian MC'] = generate_gaussian_mc_scenarios(train_raw, CONFIG)

    # 3. Evaluate models against the unseen test set
    metrics_data = {}
    for name, generated_scenarios in results.items():
        metrics_data[name] = calculate_metrics(test_raw, generated_scenarios, name)
    
    metrics_df = pd.DataFrame(metrics_data).T.round(3)
    print("\n--- [FINAL] Quantitative Performance Metrics (Lower is Better) ---")
    print(metrics_df)
    metrics_df.to_csv("quantitative_metrics.csv")
    
    # 4. Visualize results with separate plots
    print("\n--- [5] Generating Final Comparison Plots (Separately) ---")
    plot_mean_profile(test_raw, results)
    plot_std_dev(test_raw, results)
    plot_acf(test_raw, results)
    plot_value_distribution(test_raw, results)
    plot_ramp_rate(test_raw, results)
    
    print("\n--- ✅ Workflow Complete ---")