"""Clustering analysis for autoencoder embeddings in forest fire detection."""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import model_autoencoder
import model_utils
import core_logging
import viz_style
from config import Config


class EmbeddingClusterAnalyzer:
    """Analyzes and clusters embeddings from autoencoder models."""

    def __init__(self, config):
        self.config = config
        self.logger = core_logging.ProcessLogger(config, "Clustering")
        self.device = torch.device(config.DEVICE)
        
        # Visualization Setup
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

    def extract_embeddings(self, model, dataloader):
        """
        Extract embeddings from autoencoder model.
        
        Args:
            model: Trained autoencoder model
            dataloader: Data loader for extraction
            
        Returns:
            tuple: (embeddings, window_ids, fire_ids, distances, true_labels) arrays
        """
        embeddings = []
        all_window_ids = []
        all_fire_ids = []
        all_distances = []
        
        self.logger.log_step("Extracting embeddings from model")
        
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Extracting embeddings"):
                x, fire_id, distance, window_id = batch_data
                
                x = x.to(self.device)
                
                # Get embeddings using the model's get_embeddings method
                z = model.get_embeddings(x)
                embeddings.extend(z.cpu().numpy())
                all_window_ids.extend(window_id.cpu().numpy())
                all_fire_ids.extend(fire_id.cpu().numpy())
                all_distances.extend(distance.cpu().numpy())
        
        embeddings = np.array(embeddings)
        all_window_ids = np.array(all_window_ids)
        all_fire_ids = np.array(all_fire_ids)
        all_distances = np.array(all_distances)
        
        # Create true labels based on fire proximity
        true_labels = self._create_true_labels(all_fire_ids, all_distances)
        
        self.logger.log_step("Embeddings extracted", {
            'total_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'fire_samples': int(np.sum(true_labels)),
            'normal_samples': int(np.sum(true_labels == 0))
        })
        
        return embeddings, all_window_ids, all_fire_ids, all_distances, true_labels

    def _create_true_labels(self, all_fire_ids, all_distances):
        """Create binary labels based on fire proximity."""
        potential_fire_mask = all_fire_ids > 0
        distance_filtered_fire_mask = (
            potential_fire_mask &
            (all_distances <= self.config.DISTANCE_FILTER_THRESHOLD_M)
        )
        return distance_filtered_fire_mask.astype(int)

    def perform_kmeans_clustering(self, embeddings, n_clusters=2):
        """Perform K-Means clustering on embeddings."""
        self.logger.log_step(f"Performing K-Means clustering with {n_clusters} clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        self.logger.log_step("K-Means clustering completed", {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette_avg),
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_clusters)]
        })
        
        return cluster_labels, kmeans, silhouette_avg

    def perform_dbscan_clustering(self, embeddings, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering on embeddings."""
        self.logger.log_step(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        # Count clusters (excluding noise labeled as -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        # Calculate silhouette score (excluding noise points)
        if n_clusters > 1:
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:  # Need at least 2 samples for silhouette
                silhouette_avg = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
        
        self.logger.log_step("DBSCAN clustering completed", {
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'silhouette_score': float(silhouette_avg) if silhouette_avg != -1 else 'N/A',
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_clusters)] if n_clusters > 0 else []
        })
        
        return cluster_labels, dbscan, silhouette_avg

    def reduce_dimensionality(self, embeddings, method='tsne', n_components=2):
        """Reduce dimensionality for visualization."""
        self.logger.log_step(f"Reducing dimensionality using {method.upper()} to {n_components} components")
        
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=self.config.RANDOM_SEED, perplexity=30)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.config.RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings

    def visualize_clustering_results(self, reduced_embeddings, cluster_labels, true_labels,
                                   distances, clustering_method, silhouette_score, output_path):
        """Create comprehensive visualization of clustering results."""
        self.logger.log_step("Generating clustering visualization")
        
        # Set publication-ready style
        viz_style.set_publication_style()
        
        # Create figure with subplots - now 3x2 grid
        fig, axes = plt.subplots(3, 2, figsize=(16, 24))
        fig.suptitle(f'Embedding Clustering Analysis - {clustering_method}',
                    fontsize=18, fontweight='bold')
        
        # # 1. Clustering results (colored by cluster)
        # ax1 = axes[0, 0]
        # scatter1 = ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
        #                      c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
        # ax1.set_title(f'Clustering Results (Silhouette: {silhouette_score:.3f})',
        #              fontweight='bold', fontsize=15)
        # ax1.set_xlabel('Component 1', fontweight='bold')
        # ax1.set_ylabel('Component 2', fontweight='bold')
        # plt.colorbar(scatter1, ax=ax1, label='Cluster Label')
        # ax1.grid(True, alpha=0.3, linewidth=0.6)
        
        # # 2. Ground truth comparison (colored by true labels)
        # ax2 = axes[0, 1]
        # scatter2 = ax2.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
        #                      c=true_labels, cmap='coolwarm', alpha=0.7, s=30)
        # ax2.set_title('Ground Truth Labels', fontweight='bold', fontsize=15)
        # ax2.set_xlabel('Component 1', fontweight='bold')
        # ax2.set_ylabel('Component 2', fontweight='bold')
        # plt.colorbar(scatter2, ax=ax2, label='True Label (0=Normal, 1=Fire)')
        # ax2.grid(True, alpha=0.3, linewidth=0.6)
        
        # 3. Cluster size distribution
        ax3 = axes[1, 0]
        unique_clusters = np.unique(cluster_labels)
        
        fire_counts = []
        non_fire_counts = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            fire_in_cluster = np.sum(true_labels[cluster_mask] == 1)
            non_fire_in_cluster = np.sum(true_labels[cluster_mask] == 0)
            fire_counts.append(fire_in_cluster)
            non_fire_counts.append(non_fire_in_cluster)

        # Create stacked bar chart
        ax3.bar(unique_clusters, non_fire_counts, label='Non-Fire', color=viz_style.VALID_COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.bar(unique_clusters, fire_counts, bottom=non_fire_counts, label='Fire', color=viz_style.VALID_COLORS['red'], alpha=0.8, edgecolor='black', linewidth=0.8)

        ax3.set_title('Cluster Size Distribution', fontweight='bold', fontsize=15)
        ax3.set_xlabel('Cluster Label', fontweight='bold')
        ax3.set_ylabel('Number of Samples', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for i, cluster_id in enumerate(unique_clusters):
            total_height = non_fire_counts[i] + fire_counts[i]
            ax3.text(cluster_id, total_height + 0.01, f'{total_height}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Fire percentage per cluster
        ax4 = axes[1, 1]
        fire_percentages = []
        cluster_ids = []
        
        total_fire_samples = np.sum(true_labels == 1)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                fire_count_in_cluster = np.sum(true_labels[cluster_mask] == 1)
                
                if total_fire_samples > 0:
                    fire_percentage = (fire_count_in_cluster / total_fire_samples) * 100
                else:
                    fire_percentage = 0.0
                
                fire_percentages.append(fire_percentage)
                cluster_ids.append(cluster_id)

        bars = ax4.bar(cluster_ids, fire_percentages,
                      color=[viz_style.VALID_COLORS['red'] if p > 10 else viz_style.VALID_COLORS['blue'] for p in fire_percentages],
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        ax4.set_title('Distribution of Fire Samples per Cluster', fontweight='bold', fontsize=15)
        ax4.set_xlabel('Cluster Label', fontweight='bold')
        ax4.set_ylabel('Percentage of Total Fire Samples (%)', fontweight='bold')
        ax4.grid(True, alpha=0.3, linewidth=0.6)
        
        # Add value labels on bars
        for bar, percentage in zip(bars, fire_percentages):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 5. Cluster quality metrics
        ax5 = axes[2, 0]
        metrics = {
            'Silhouette Score': silhouette_score,
            'Number of Clusters': len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            'Noise Points': np.sum(cluster_labels == -1) if -1 in unique_clusters else 0
        }
        
        if len(np.unique(true_labels)) > 1 and len(np.unique(cluster_labels)) > 1:
            # Calculate Adjusted Rand Index if we have both true labels and multiple clusters
            ari = adjusted_rand_score(true_labels, cluster_labels)
            metrics['Adjusted Rand Index'] = ari
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax5.bar(metric_names, metric_values,
                      color=[viz_style.VALID_COLORS['green'], viz_style.VALID_COLORS['blue'],
                             viz_style.VALID_COLORS['orange'], viz_style.VALID_COLORS['purple']][:len(metric_names)],
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        ax5.set_title('Cluster Quality Metrics', fontweight='bold', fontsize=15)
        ax5.set_ylabel('Value', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, linewidth=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}' if isinstance(value, float) else f'{value}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        

        # 6. Average distance to fire per cluster
        ax6 = axes[2, 1]
        avg_distances = []
        cluster_ids_for_dist = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                # Only consider "close fire" samples for this plot
                close_fire_mask = true_labels == 1
                combined_mask = cluster_mask & close_fire_mask
                
                if np.sum(combined_mask) > 0:
                    cluster_distances = distances[combined_mask]
                    # Replace inf with nan for calculation
                    cluster_distances[cluster_distances == np.inf] = np.nan
                    avg_dist = np.nanmean(cluster_distances)
                else:
                    avg_dist = np.nan # Or 0, depending on desired representation
                    
                avg_distances.append(avg_dist)
                cluster_ids_for_dist.append(cluster_id)
        
        bars = ax6.bar(cluster_ids_for_dist, avg_distances,
                      color=[viz_style.VALID_COLORS['orange'] if d < 10000 else viz_style.VALID_COLORS['blue'] for d in avg_distances],
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        ax6.set_title('Average Distance to Fire per Cluster', fontweight='bold', fontsize=15)
        ax6.set_xlabel('Cluster Label', fontweight='bold')
        ax6.set_ylabel('Average Distance (m)', fontweight='bold')
        ax6.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, avg_distances):
            if not np.isnan(value):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.0f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        
        # Save outputs
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        self.logger.log_step(f"Clustering visualization saved to: {output_path}")

    def analyze_embeddings(self, dataloader, model_path=None, model=None):
        """
        Main method to perform complete embedding analysis.
        
        Args:
            dataloader: Data loader for analysis
            model_path: Path to trained model (optional if model is provided)
            model: Pre-loaded model (optional if model_path is provided)
            
        Returns:
            dict: Clustering results and metrics
        """
        self.logger.log_step("Starting embedding analysis")
        
        # Load model if not provided
        if model is None:
            if model_path is None:
                model_path = self.config.BEST_MODEL_PATH
            
            autoencoder_class = getattr(model_autoencoder, self.config.AUTOENCODER_CLASS)
            model = autoencoder_class(
                time_steps=self.config.WINDOW_SIZE,
                num_features=len(self.config.INPUT_COLUMNS),
                latent_dim=self.config.LATENT_DIM,
                hidden_dim=self.config.HIDDEN_DIM
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.to(self.device)
        
        # Extract embeddings
        embeddings, window_ids, fire_ids, distances, true_labels = \
            self.extract_embeddings(model, dataloader)
        
        # Reduce dimensionality for visualization
        # reduced_embeddings = self.reduce_dimensionality(embeddings, method=self.config.DIM_REDUCTION_METHOD)
        reduced_embeddings = None
        # Perform K-Means clustering
        kmeans_labels, kmeans_model, kmeans_silhouette = \
            self.perform_kmeans_clustering(embeddings, n_clusters=self.config.KMEANS_N_CLUSTERS)
        
        # Perform DBSCAN clustering
        dbscan_labels, dbscan_model, dbscan_silhouette = \
            self.perform_dbscan_clustering(embeddings, eps=self.config.DBSCAN_EPS, min_samples=self.config.DBSCAN_MIN_SAMPLES)
        
        # Visualize results
        kmeans_output_path = os.path.join(self.config.STATS_IMAGES_DIR, "kmeans_clustering_results.png")
        dbscan_output_path = os.path.join(self.config.STATS_IMAGES_DIR, "dbscan_clustering_results.png")
        
        self.visualize_clustering_results(reduced_embeddings, kmeans_labels, true_labels,
                                         distances, "K-Means", kmeans_silhouette, kmeans_output_path)
        self.visualize_clustering_results(reduced_embeddings, dbscan_labels, true_labels,
                                         distances, "DBSCAN", dbscan_silhouette, dbscan_output_path)
        
        # Save clustering results
        results = {
            'embeddings': embeddings,
            'reduced_embeddings': reduced_embeddings,
            'window_ids': window_ids,
            'fire_ids': fire_ids,
            'distances': distances,
            'true_labels': true_labels,
            'kmeans_labels': kmeans_labels,
            'kmeans_silhouette': kmeans_silhouette,
            'dbscan_labels': dbscan_labels,
            'dbscan_silhouette': dbscan_silhouette,
            'kmeans_model': kmeans_model,
            'dbscan_model': dbscan_model
        }
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'window_id': window_ids,
            'fire_id': fire_ids,
            'distance': distances,
            'true_label': true_labels,
            'kmeans_cluster': kmeans_labels,
            'dbscan_cluster': dbscan_labels
        })
        
        os.makedirs(self.config.STATS_CSV_DIR, exist_ok=True)
        results_df.to_csv(os.path.join(self.config.STATS_CSV_DIR, "clustering_results.csv"), index=False)
        
        self.logger.log_step("Embedding analysis completed successfully")
        self.logger.save_process_timeline()
        
        return results


# =========================================================================
# Main Function
# =========================================================================
def analyze_embeddings(config, dataloader=None, model_path=None, model=None):
    """
    Main entry point for embedding analysis.
    
    Args:
        config: Configuration object
        dataloader: Data loader for analysis (optional)
        model_path: Path to trained model (optional)
        model: Pre-loaded model (optional)
        
    Returns:
        dict: Clustering results from EmbeddingClusterAnalyzer
    """
    if dataloader is None:
        # If no dataloader is provided, create a default test loader
        _, _, dataloader = model_utils.create_dataloaders(config, remove_fire_labels=False)
    
    analyzer = EmbeddingClusterAnalyzer(config)
    return analyzer.analyze_embeddings(dataloader, model_path, model)