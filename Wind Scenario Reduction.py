# ======================================================================================
#
#  Ultimate Framework for Wind Scenario Reduction via Deep Embedded Clustering (DEC)
#
#  This script provides a complete, robust, and validated pipeline for reducing a large set
#  of generated wind scenarios. It features a streamlined workflow, hyperparameter
#  optimization for the number of clusters (k), and a comprehensive evaluation framework
#  that assesses the statistical fidelity of the reduced set in real physical units.
#
#  Author: Fatemeh Amini
#  Date: 2025/09/14
#
# ======================================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_absolute_error
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import warnings
from tqdm.auto import tqdm
import pandas as pd
import joblib

# --- Suppress Warnings & Setup Plotting ---
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
CLUSTER_COLORS = plt.cm.get_cmap('tab20', 20)

# ======================================================================================
# --- [1] Configuration and Setup ---
# ======================================================================================
print("--- [1] Initializing Configuration ---")
CONFIG = {
    # --- File Paths ---
    "generator_path": "generator_wind_model.pth",      # Path to the generator from Script 1
    "autoencoder_path": "autoencoder_wind.pth",        # Path to save/load the pre-trained AE
    "scaler_path": "power_scaler.joblib",              # Path to save/load the power scaler

    # --- Scenario Generation & Reduction ---
    "num_scenarios_to_generate": 5000,
    "max_k_to_test": 15,           # Maximum number of clusters to test for optimization
    "default_k": 10,               # Fallback K if optimization is inconclusive
    "user_override_k": 5,          # Manually set K for the final run, e.g., to 5

    # --- DEC Model Hyperparameters ---
    "batch_size": 512,
    "pretrain_epochs": 100,        # Epochs for autoencoder pre-training
    "finetune_epochs": 100,        # Epochs for DEC fine-tuning
    "lr": 0.001,
    "gamma": 0.1,                  # Coefficient for KL divergence loss
    "tolerance": 0.001,            # Convergence tolerance for DEC fine-tuning
    "random_seed": 42,

    # --- GAN config must match the generator's training ---
    "gan_latent_dim": 100,
    "gan_seq_len": 48,
    "gan_hidden_dim": 128,
    "gan_n_layers": 2
}

# --- Device Setup & Reproducibility ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
n_cpu_workers = os.cpu_count()
print(f"Using device: {device} with {n_gpus} GPU(s) and {n_cpu_workers} CPU core(s).")

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
set_seeds(CONFIG["random_seed"])


# ======================================================================================
# --- [2] Scenario Generation ---
# ======================================================================================
class GeneratorRNN(nn.Module):
    """Must be the same architecture as the one used for training in Script 1."""
    def __init__(self, latent_dim, hidden_dim, seq_len, n_layers):
        super(GeneratorRNN, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
    def forward(self, z):
        z_seq = z.unsqueeze(1).repeat(1, CONFIG["gan_seq_len"], 1)
        lstm_out, _ = self.lstm(z_seq)
        out = self.linear(lstm_out)
        return self.tanh(out)

def generate_scenarios(config):
    """Generates a large set of scenarios using the pre-trained GAN generator."""
    print(f"--- [2] Generating {config['num_scenarios_to_generate']} scenarios from pre-trained RNN-GAN ---")
    if not os.path.exists(config["generator_path"]):
        raise FileNotFoundError(f"Generator model not found at '{config['generator_path']}'. Run Script 1 first.")

    generator = GeneratorRNN(
        config["gan_latent_dim"], config["gan_hidden_dim"],
        config["gan_seq_len"], config["gan_n_layers"]
    ).to(device)

    state_dict = torch.load(config["generator_path"], map_location=device)
    # Handle the 'module.' prefix if the model was saved using DataParallel
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    generator.load_state_dict(state_dict)

    generator.eval()
    with torch.no_grad():
        z = torch.randn(config["num_scenarios_to_generate"], config["gan_latent_dim"], device=device)
        # Permute to shape: (N, Channels=1, Seq_Len) for convolutional layers
        scenarios_normalized = generator(z).permute(0, 2, 1)
    print(f"Generated {scenarios_normalized.shape[0]} normalized scenarios.")
    return scenarios_normalized.cpu()


# ======================================================================================
# --- [3] DEC Model and Trainer Class ---
# ======================================================================================
class ConvAutoencoder(nn.Module):
    """A 1D Convolutional Autoencoder to learn features from time series data."""
    def __init__(self, seq_len: int, embedding_dim: int = 10):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            nn.Flatten(), nn.Linear(8 * (seq_len // 4), embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 8 * (seq_len // 4)), nn.ReLU(True),
            nn.Unflatten(1, (8, (seq_len // 4))),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class DEC_Trainer:
    """Manages the pre-training, fine-tuning, and prediction for the DEC model."""
    def __init__(self, seq_len, config, n_clusters=None):
        self.config = config
        self.n_clusters = n_clusters or config["default_k"]
        self.autoencoder = ConvAutoencoder(seq_len)
        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, 10))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.model = nn.ModuleDict({'autoencoder': self.autoencoder, 'cluster_layer': nn.ParameterList([self.cluster_layer])}).to(device)
        if n_gpus > 1:
            self.model['autoencoder'] = nn.DataParallel(self.model['autoencoder'])

    def _get_ae(self):
        return self.model['autoencoder'].module if n_gpus > 1 else self.model['autoencoder']

    def pretrain_autoencoder(self, scenarios_tensor, force_retrain=False):
        """Pre-trains the autoencoder using only reconstruction loss."""
        print("\n--- [3.1] Pre-training Convolutional Autoencoder ---")
        if os.path.exists(self.config["autoencoder_path"]) and not force_retrain:
            print(f"Loading pre-trained autoencoder from {self.config['autoencoder_path']}")
            self._get_ae().load_state_dict(torch.load(self.config["autoencoder_path"], map_location=device))
            return

        dataloader = DataLoader(
            TensorDataset(scenarios_tensor), batch_size=self.config["batch_size"] * max(1, n_gpus),
            shuffle=True, num_workers=n_cpu_workers // 2 if n_cpu_workers else 0
        )
        optimizer = torch.optim.Adam(self._get_ae().parameters(), lr=self.config["lr"])

        for epoch in tqdm(range(self.config["pretrain_epochs"]), desc="Pre-training AE"):
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(device)
                x_recon, _ = self._get_ae()(x_batch)
                loss = nn.MSELoss()(x_recon, x_batch)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        print(f"Saving pre-trained autoencoder to {self.config['autoencoder_path']}")
        torch.save(self._get_ae().state_dict(), self.config["autoencoder_path"])

    def get_latent_vectors(self, scenarios_tensor):
        """Extracts embedded feature vectors from the pre-trained encoder."""
        print("\n--- [3.2] Extracting latent space vectors ---")
        dataloader = DataLoader(TensorDataset(scenarios_tensor), batch_size=self.config["batch_size"] * max(1, n_gpus))
        latent_vectors = []
        self._get_ae().eval()
        with torch.no_grad():
            for (x_batch,) in dataloader:
                _, z = self._get_ae()(x_batch.to(device))
                latent_vectors.append(z.cpu())
        return torch.cat(latent_vectors).numpy()

    def finetune(self, scenarios_tensor, latent_vectors_np):
        """Fine-tunes the autoencoder and cluster centroids jointly."""
        print(f"\n--- [3.3] Initializing centroids and fine-tuning DEC for K={self.n_clusters} ---")
        dataloader = DataLoader(TensorDataset(scenarios_tensor), batch_size=self.config["batch_size"] * max(1, n_gpus))

        # Initialize cluster centroids using K-means on the latent space
        kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=self.config["random_seed"])
        y_pred_last = kmeans.fit_predict(latent_vectors_np)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        for epoch in tqdm(range(self.config["finetune_epochs"]), desc=f"Fine-tuning DEC (K={self.n_clusters})"):
            q_full = self.predict_q(scenarios_tensor)
            p_full = self._target_distribution(q_full.data) # Get the auxiliary target distribution
            y_pred = q_full.cpu().argmax(1).numpy()
            
            # Check for convergence
            delta_label = np.sum(y_pred != y_pred_last) / y_pred.shape[0]
            if epoch > 0 and delta_label < self.config["tolerance"]:
                print(f"Convergence at epoch {epoch}. Delta: {delta_label:.4f}"); break
            y_pred_last = y_pred

            for i, (x_batch,) in enumerate(dataloader):
                p_batch = p_full[i*dataloader.batch_size : (i+1)*dataloader.batch_size].to(device)
                x_batch = x_batch.to(device)
                x_recon, z, q_batch = self._forward_full(x_batch)
                
                # Combined loss: KL divergence for clustering + MSE for reconstruction
                kl_loss = nn.KLDivLoss(reduction='batchmean')(q_batch.log(), p_batch)
                mse_loss = nn.MSELoss()(x_recon, x_batch)
                loss = self.config["gamma"] * kl_loss + mse_loss
                
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        return y_pred_last

    def _target_distribution(self, q):
        """Computes the auxiliary target distribution P from Q."""
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def _forward_full(self, x):
        """Full forward pass returning reconstruction, latent vector, and soft assignments."""
        x_recon, z = self._get_ae()(x)
        # Calculate soft cluster assignments (q) using Student's t-distribution
        q = 1.0 / (1.0 + torch.sum(torch.square(z.unsqueeze(1) - self.cluster_layer), dim=2))
        q = (q.T / torch.sum(q, dim=1)).T
        return x_recon, z, q

    def predict_q(self, scenarios_tensor):
        """Predicts soft cluster assignments for the entire dataset."""
        dataloader = DataLoader(TensorDataset(scenarios_tensor), batch_size=self.config["batch_size"] * max(1, n_gpus))
        all_q = []
        self.model.eval()
        with torch.no_grad():
            for (x_batch,) in dataloader:
                _, _, q_batch = self._forward_full(x_batch.to(device))
                all_q.append(q_batch)
        return torch.cat(all_q, 0)

    def get_representative_profiles(self):
        """Decodes the final cluster centroids to get the representative time series."""
        self.model.eval()
        with torch.no_grad():
            return self._get_ae().decoder(self.cluster_layer.data).cpu().squeeze(1)


# ======================================================================================
# --- [4] Hyperparameter (K) Optimization and Evaluation ---
# ======================================================================================
def find_optimal_k(latent_vectors, max_k, config):
    """Finds the optimal number of clusters (K) using various clustering metrics."""
    print(f"\n--- [4.1] Finding Optimal K (up to K={max_k}) ---")
    k_range = range(2, max_k + 1)
    metrics = {"inertia": [], "silhouette": [], "calinski": [], "davies": []}
    for k in tqdm(k_range, desc="Testing K values"):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=config["random_seed"])
        labels = kmeans.fit_predict(latent_vectors)
        metrics["inertia"].append(kmeans.inertia_)
        metrics["silhouette"].append(silhouette_score(latent_vectors, labels))
        metrics["calinski"].append(calinski_harabasz_score(latent_vectors, labels))
        metrics["davies"].append(davies_bouldin_score(latent_vectors, labels))

    # Determine optimal k from silhouette score (good balance of cohesion and separation)
    optimal_k = k_range[np.argmax(metrics["silhouette"])]
    print(f"Optimal K based on Silhouette Score: {optimal_k}")
    
    # --- Visualization for Optimal K Analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analysis of Optimal Number of Clusters (K)', fontsize=20, weight='bold')
    
    axes[0, 0].plot(k_range, metrics["inertia"], 'o-'); axes[0, 0].set_title('Elbow Method (Inertia)')
    axes[0, 1].plot(k_range, metrics["silhouette"], 'o-'); axes[0, 1].set_title('Silhouette Score (Higher is Better)')
    axes[1, 0].plot(k_range, metrics["calinski"], 'o-'); axes[1, 0].set_title('Calinski-Harabasz (Higher is Better)')
    axes[1, 1].plot(k_range, metrics["davies"], 'o-'); axes[1, 1].set_title('Davies-Bouldin (Lower is Better)')
    
    for ax in axes.flat:
        ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}'); ax.legend()
        ax.set_xlabel('Number of Clusters (K)'); ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Optimal_K_Analysis.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    
    return optimal_k

def evaluate_reduction_quality(original_scenarios, reduced_scenarios, labels):
    """Evaluates how well the reduced set represents the original large set."""
    print("\n--- [4.2] Evaluating Statistical Fidelity of Reduced Scenario Set (in kW) ---")
    metrics = {}
    n_total = len(original_scenarios)
    # Calculate probability of each cluster
    probs = np.array([np.sum(labels == i) / n_total for i in range(len(reduced_scenarios))])

    # 1. Weighted Wasserstein Distance to compare overall distributions
    metrics['Wasserstein Dist.'] = wasserstein_distance(original_scenarios.flatten(), reduced_scenarios.flatten(), v_weights=np.repeat(probs, reduced_scenarios.shape[1]))

    # 2. Mean Profile Error (weighted average)
    mean_original = original_scenarios.mean(axis=0)
    mean_reduced_weighted = np.average(reduced_scenarios, axis=0, weights=probs)
    metrics['Mean Profile MAE'] = mean_absolute_error(mean_original, mean_reduced_weighted)

    # 3. Std Dev Profile Error (weighted average)
    std_original = original_scenarios.std(axis=0)
    variance_reduced_weighted = np.average((reduced_scenarios - mean_reduced_weighted)**2, axis=0, weights=probs)
    std_reduced_weighted = np.sqrt(variance_reduced_weighted)
    metrics['Std Profile MAE'] = mean_absolute_error(std_original, std_reduced_weighted)
    
    # 4. Ramp Rate Profile Error (weighted average)
    ramps_original = np.abs(np.diff(original_scenarios, axis=1)).mean(axis=0)
    ramps_reduced = np.abs(np.diff(reduced_scenarios, axis=1))
    ramps_reduced_weighted = np.average(ramps_reduced, axis=0, weights=probs)
    metrics['Ramp Profile MAE'] = mean_absolute_error(ramps_original, ramps_reduced_weighted)

    return pd.DataFrame([metrics]).T.rename(columns={0: 'Score'}).round(4)


# ======================================================================================
# --- [5] Visualization (MODIFIED FOR SEPARATE PLOTS) ---
# ======================================================================================
def plot_tsne_clusters(final_labels, latent_vectors, config):
    """Generates and saves the t-SNE visualization of latent space clusters."""
    print("--- [5.1] Generating t-SNE Cluster Visualization ---")
    n_clusters = len(np.unique(final_labels))
    fig, ax = plt.subplots(figsize=(12, 8))
    
    tsne = TSNE(n_components=2, perplexity=50, random_state=config["random_seed"], n_jobs=-1, init='pca')
    z_tsne = tsne.fit_transform(latent_vectors)
    
    for i in range(n_clusters):
        points = z_tsne[final_labels == i]
        ax.scatter(points[:, 0], points[:, 1], color=CLUSTER_COLORS(i), label=f'Cluster {i}', alpha=0.7)
    
    ax.set_title(f't-SNE Visualization of {n_clusters} Scenario Clusters in Latent Space', fontsize=16, weight='bold')
    ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(title="Clusters", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"figure_1_tsne_clusters_{n_clusters}.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

def plot_representative_profiles(rep_profiles, final_labels):
    """Generates and saves the plot of representative scenario profiles."""
    print("--- [5.2] Generating Representative Profiles Plot ---")
    n_clusters = len(rep_profiles)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    probs = [np.mean(final_labels == i) for i in range(n_clusters)]
    for i in range(n_clusters):
        ax.plot(rep_profiles[i], color=CLUSTER_COLORS(i), linewidth=2.5, label=f'Profile {i} (Prob: {probs[i]:.3f})')
        
    ax.set_title(f'Representative 48-Hour Wind Profiles ({n_clusters} Cluster Centroids)', fontsize=16, weight='bold')
    ax.set_xlabel('Hour', fontsize=12); ax.set_ylabel('Active Power (kW)', fontsize=12)
    ax.legend(title="Scenarios", loc='best'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"figure_2_representative_profiles_{n_clusters}.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

def plot_metrics_table(metrics_df):
    """Generates and saves the reduction quality metrics as a table figure."""
    print("--- [5.3] Generating Metrics Table Figure ---")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.set_title('Reduction Quality Metrics (Lower is Better)', fontsize=16, weight='bold', pad=20)
    
    table = ax.table(cellText=metrics_df.values, rowLabels=metrics_df.index, loc='center', colLabels=metrics_df.columns)
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.8)
    
    plt.savefig("figure_3_metrics_table.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)

# ======================================================================================
# --- Main Execution Workflow ---
# ======================================================================================
if __name__ == "__main__":
    try:
        # 1. Generate the initial large set of scenarios (normalized)
        scenarios_normalized_tensor = generate_scenarios(CONFIG)
        scenarios_normalized_np = scenarios_normalized_tensor.squeeze(1).numpy()

        # 2. Fit a scaler on the generated data to map back to physical units (kW)
        print("\n--- Fitting scaler to transform data to physical units (kW) ---")
        power_scaler = MinMaxScaler(feature_range=(0, 3600)) # Assuming a max power of 3600kW
        power_scaler.fit(scenarios_normalized_np.flatten().reshape(-1, 1))
        joblib.dump(power_scaler, CONFIG["scaler_path"])
        
        # 3. Initialize a temporary DEC trainer and pre-train the autoencoder
        temp_trainer = DEC_Trainer(CONFIG["gan_seq_len"], CONFIG)
        temp_trainer.pretrain_autoencoder(scenarios_normalized_tensor)
        latent_space = temp_trainer.get_latent_vectors(scenarios_normalized_tensor)

        # 4. Find the optimal number of clusters, k, and plot the analysis
        optimal_k_found = find_optimal_k(latent_space, CONFIG["max_k_to_test"], CONFIG)
        
        # Manually override k to a specific value for this experiment
        optimal_k = CONFIG.get("user_override_k", optimal_k_found)
        print(f"\n--- USER OVERRIDE: Proceeding with K = {optimal_k} for final clustering ---")
        
        # 5. Initialize the FINAL trainer with the desired k and fine-tune
        final_trainer = DEC_Trainer(CONFIG["gan_seq_len"], CONFIG, n_clusters=optimal_k)
        # Load the already pre-trained autoencoder weights to save time
        final_trainer._get_ae().load_state_dict(torch.load(CONFIG["autoencoder_path"], map_location=device))
        final_labels = final_trainer.finetune(scenarios_normalized_tensor, latent_space)

        # 6. Get representative profiles and un-normalize all data for evaluation
        rep_profiles_normalized = final_trainer.get_representative_profiles().numpy()
        scenarios_unscaled = power_scaler.inverse_transform(scenarios_normalized_np.reshape(-1, 1)).reshape(scenarios_normalized_np.shape)
        rep_profiles_unscaled = power_scaler.inverse_transform(rep_profiles_normalized.reshape(-1, 1)).reshape(rep_profiles_normalized.shape)

        # 7. Evaluate the reduction quality on the physical data
        metrics_df = evaluate_reduction_quality(scenarios_unscaled, rep_profiles_unscaled, final_labels)
        print("\n--- Final Metrics ---"); print(metrics_df)
        
        # 8. Visualize all results in separate figures
        plot_tsne_clusters(final_labels, latent_space, CONFIG)
        plot_representative_profiles(rep_profiles_unscaled, final_labels)
        plot_metrics_table(metrics_df)

        print("\n--- ✅ Workflow Complete ---")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure the 'generator_wind_model.pth' file from Script 1 is in the correct directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()