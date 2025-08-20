"""Clustering analysis for autoencoder embeddings in forest fire detection."""

import os
from typing import Dict, List, Tuple, Optional, Union, Any, cast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # Keep for type hinting
from matplotlib.axes import Axes  # Keep for type hinting
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from . import model_base
from ..data.utils import create_dataloaders
from ..common import core_logging
from ..common import viz_style
from ..common.config import Config


class EmbeddingClusterAnalyzer:
    """Analyzes and clusters embeddings from autoencoder models."""

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger: core_logging.ProcessLogger = core_logging.ProcessLogger(config, "Clustering")
        self.device: torch.device = torch.device(config.training.DEVICE)

        # Visualization Setup
        plt.style.use("seaborn-v0_8-paper")
        sns.set_palette("husl")

    def extract_embeddings(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings from autoencoder model.

        Args:
            model: Trained autoencoder model
            dataloader: Data loader for extraction

        Returns:
            tuple: (embeddings, window_ids, fire_ids, distances, true_labels) arrays
        """
        embeddings: List[np.ndarray] = []
        all_window_ids: List[np.ndarray] = []
        all_fire_ids: List[np.ndarray] = []
        all_distances: List[np.ndarray] = []

        self.logger.log_step("Extracting embeddings from model")

        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Extracting embeddings"):
                x, fire_id, distance, window_id = batch_data

                x = x.to(self.device)

                # Get embeddings using the model's get_embeddings method
                # Type assertion for autoencoder models that implement get_embeddings
                autoencoder_model = cast(model_base.AutoencoderBase, model)
                z = autoencoder_model.get_embeddings(x)
                embeddings.extend(z.cpu().numpy())
                all_window_ids.extend(window_id.cpu().numpy())
                all_fire_ids.extend(fire_id.cpu().numpy())
                all_distances.extend(distance.cpu().numpy())

        embeddings_array: np.ndarray = np.array(embeddings)
        all_window_ids_array: np.ndarray = np.array(all_window_ids)
        all_fire_ids_array: np.ndarray = np.array(all_fire_ids)
        all_distances_array: np.ndarray = np.array(all_distances)

        # Create true labels based on fire proximity
        true_labels: np.ndarray = self._create_true_labels(all_fire_ids_array, all_distances_array)

        self.logger.log_step(
            "Embeddings extracted",
            {
                "total_samples": len(embeddings_array),
                "embedding_dim": embeddings_array.shape[1],
                "fire_samples": int(np.sum(true_labels)),
                "normal_samples": int(np.sum(true_labels == 0)),
            },
        )

        return (
            embeddings_array,
            all_window_ids_array,
            all_fire_ids_array,
            all_distances_array,
            true_labels,
        )

    def _create_true_labels(self, all_fire_ids: np.ndarray, all_distances: np.ndarray) -> np.ndarray:
        """Create binary labels based on fire proximity."""
        # Fixed: Handle inf distances properly (inf means no fire)
        valid_fire_mask = (all_fire_ids > 0) & (all_distances != np.inf)
        distance_filtered_fire_mask = valid_fire_mask & (
            all_distances <= self.config.training.DISTANCE_FILTER_THRESHOLD_M
        )
        return distance_filtered_fire_mask.astype(int)

    def perform_kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int = 2) -> Tuple[np.ndarray, KMeans, Any]:
        """Perform K-Means clustering on embeddings."""
        self.logger.log_step(f"Performing K-Means clustering with {n_clusters} clusters")

        kmeans: KMeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.training.RANDOM_SEED,
            n_init=10,
        )
        cluster_labels: np.ndarray = kmeans.fit_predict(embeddings)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)

        self.logger.log_step(
            "K-Means clustering completed",
            {
                "n_clusters": n_clusters,
                "silhouette_score": float(silhouette_avg),
                "cluster_sizes": [int(np.sum(cluster_labels == i)) for i in range(n_clusters)],
            },
        )

        return cluster_labels, kmeans, silhouette_avg

    def perform_dbscan_clustering(
        self, embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 5
    ) -> Tuple[np.ndarray, DBSCAN, Any]:
        """Perform DBSCAN clustering on embeddings."""
        self.logger.log_step(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}")

        dbscan: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels: np.ndarray = dbscan.fit_predict(embeddings)

        # Count clusters (excluding noise labeled as -1)
        n_clusters: int = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise: int = np.sum(cluster_labels == -1)

        # Calculate silhouette score (excluding noise points)
        # Fixed: Better handling of edge cases
        silhouette_avg: Any = None
        if n_clusters > 1:
            non_noise_mask: np.ndarray = cluster_labels != -1
            non_noise_count: int = np.sum(non_noise_mask)
            if non_noise_count > 1:
                try:
                    silhouette_avg = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
                except ValueError:
                    # Handle case where all samples are in the same cluster
                    silhouette_avg = 0.0
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0

        self.logger.log_step(
            "DBSCAN clustering completed",
            {
                "n_clusters": n_clusters,
                "n_noise_points": int(n_noise),
                "silhouette_score": (float(silhouette_avg) if silhouette_avg is not None else "N/A"),
                "cluster_sizes": (
                    [int(np.sum(cluster_labels == i)) for i in range(n_clusters)] if n_clusters > 0 else []
                ),
            },
        )

        return (
            cluster_labels,
            dbscan,
            silhouette_avg if silhouette_avg is not None else 0.0,
        )

    def reduce_dimensionality(self, embeddings: np.ndarray, method: str = "tsne", n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality for visualization."""
        self.logger.log_step(f"Reducing dimensionality using {method.upper()} to {n_components} components")

        reducer: Union[TSNE, PCA]
        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=self.config.training.RANDOM_SEED,
                perplexity=30,
            )
        elif method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=self.config.training.RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        reduced_embeddings: np.ndarray = reducer.fit_transform(embeddings)
        return reduced_embeddings

    def visualize_clustering_results(
        self,
        reduced_embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray,
        distances: np.ndarray,
        clustering_method: str,
        silhouette_score: float,
        output_path: str,
    ) -> None:
        """Create comprehensive visualization of clustering results."""
        self.logger.log_step("Generating clustering visualization")

        # Set publication-ready style
        viz_style.set_publication_style()

        # Fixed: Correct figure layout - 4x2 grid for 7 plots (leaving one empty)
        fig: Figure
        axes: np.ndarray
        fig, axes = plt.subplots(4, 2, figsize=(16, 32))
        fig.suptitle(
            f"Embedding Clustering Analysis - {clustering_method}",
            fontsize=18,
            fontweight="bold",
        )

        # Cache unique clusters to avoid recalculation
        unique_clusters = np.unique(cluster_labels)

        # 1. Clustering results (colored by cluster)
        ax_cluster_results = axes[0, 0]
        scatter_cluster = ax_cluster_results.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=cluster_labels,
            cmap="viridis",
            alpha=0.7,
            s=30,
        )
        ax_cluster_results.set_title(
            f"Clustering Results (Silhouette: {silhouette_score:.3f})",
            fontweight="bold",
            fontsize=15,
        )
        ax_cluster_results.set_xlabel("Component 1", fontweight="bold")
        ax_cluster_results.set_ylabel("Component 2", fontweight="bold")
        plt.colorbar(scatter_cluster, ax=ax_cluster_results, label="Cluster Label")
        ax_cluster_results.grid(True, alpha=0.3, linewidth=0.6)

        # 2. Ground truth comparison (colored by true labels)
        ax_ground_truth = axes[0, 1]
        scatter_truth = ax_ground_truth.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=true_labels,
            cmap="coolwarm",
            alpha=0.7,
            s=30,
        )
        ax_ground_truth.set_title("Ground Truth Labels", fontweight="bold", fontsize=15)
        ax_ground_truth.set_xlabel("Component 1", fontweight="bold")
        ax_ground_truth.set_ylabel("Component 2", fontweight="bold")
        plt.colorbar(scatter_truth, ax=ax_ground_truth, label="True Label (0=Normal, 1=Fire)")
        ax_ground_truth.grid(True, alpha=0.3, linewidth=0.6)

        # 3. Cluster size distribution
        ax_cluster_sizes = axes[1, 0]

        fire_counts = []
        non_fire_counts = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            fire_in_cluster = np.sum(true_labels[cluster_mask] == 1)
            non_fire_in_cluster = np.sum(true_labels[cluster_mask] == 0)
            fire_counts.append(fire_in_cluster)
            non_fire_counts.append(non_fire_in_cluster)

        # Create stacked bar chart
        ax_cluster_sizes.bar(
            unique_clusters,
            non_fire_counts,
            label="Non-Fire",
            color=viz_style.VALID_COLORS["blue"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_cluster_sizes.bar(
            unique_clusters,
            fire_counts,
            bottom=non_fire_counts,
            label="Fire",
            color=viz_style.VALID_COLORS["red"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

        ax_cluster_sizes.set_title("Cluster Size Distribution", fontweight="bold", fontsize=15)
        ax_cluster_sizes.set_xlabel("Cluster Label", fontweight="bold")
        ax_cluster_sizes.set_ylabel("Number of Samples", fontweight="bold")
        ax_cluster_sizes.legend()
        ax_cluster_sizes.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for i, cluster_id in enumerate(unique_clusters):
            total_height = non_fire_counts[i] + fire_counts[i]
            ax_cluster_sizes.text(
                cluster_id,
                total_height + 0.01,
                f"{total_height}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # 4. Fire percentage per cluster
        ax_fire_distribution = axes[1, 1]
        fire_percentages = []
        cluster_ids_for_percentages = []

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
                cluster_ids_for_percentages.append(cluster_id)

        bars_fire_dist = ax_fire_distribution.bar(
            cluster_ids_for_percentages,
            fire_percentages,
            color=[
                (viz_style.VALID_COLORS["red"] if p > 10 else viz_style.VALID_COLORS["blue"]) for p in fire_percentages
            ],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_fire_distribution.set_title("Distribution of Fire Samples per Cluster", fontweight="bold", fontsize=15)
        ax_fire_distribution.set_xlabel("Cluster Label", fontweight="bold")
        ax_fire_distribution.set_ylabel("Percentage of Total Fire Samples (%)", fontweight="bold")
        ax_fire_distribution.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, percentage in zip(bars_fire_dist, fire_percentages):
            ax_fire_distribution.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # 5. Fire/Non-Fire Ratio per Cluster
        ax_fire_ratio = axes[2, 0]
        fire_ratios = []
        cluster_ids_for_ratios = []

        for cluster_id, fire, non_fire in zip(unique_clusters, fire_counts, non_fire_counts):
            total = fire + non_fire
            if total > 0:
                ratio = (fire / total) * 100
            else:
                ratio = 0.0
            fire_ratios.append(ratio)
            cluster_ids_for_ratios.append(cluster_id)

        bars_ratio = ax_fire_ratio.bar(
            cluster_ids_for_ratios,
            fire_ratios,
            color=[(viz_style.VALID_COLORS["red"] if r > 50 else viz_style.VALID_COLORS["blue"]) for r in fire_ratios],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_fire_ratio.set_title("Fire/Non-Fire Ratio per Cluster", fontweight="bold", fontsize=15)
        ax_fire_ratio.set_xlabel("Cluster Label", fontweight="bold")
        ax_fire_ratio.set_ylabel("Percentage of Fire Samples (%)", fontweight="bold")
        ax_fire_ratio.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, ratio in zip(bars_ratio, fire_ratios):
            ax_fire_ratio.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{ratio:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # 6. Cluster quality metrics
        # Fixed: Use correct subplot ax instead of ax5
        ax_quality_metrics = axes[2, 1]
        metrics = {
            "Silhouette Score": silhouette_score,
            "Number of Clusters": len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            "Noise Points": (np.sum(cluster_labels == -1) if -1 in unique_clusters else 0),
        }

        if len(np.unique(true_labels)) > 1 and len(np.unique(cluster_labels)) > 1:
            # Calculate Adjusted Rand Index if we have both true labels and multiple clusters
            ari = adjusted_rand_score(true_labels, cluster_labels)
            metrics["Adjusted Rand Index"] = ari

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars_metrics = ax_quality_metrics.bar(
            metric_names,
            metric_values,
            color=[
                viz_style.VALID_COLORS["green"],
                viz_style.VALID_COLORS["blue"],
                viz_style.VALID_COLORS["orange"],
                viz_style.VALID_COLORS["purple"],
            ][: len(metric_names)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_quality_metrics.set_title("Cluster Quality Metrics", fontweight="bold", fontsize=15)
        ax_quality_metrics.set_ylabel("Value", fontweight="bold")
        ax_quality_metrics.tick_params(axis="x", rotation=45)
        ax_quality_metrics.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, value in zip(bars_metrics, metric_values):
            ax_quality_metrics.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}" if isinstance(value, float) else f"{value}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # 7. Average distance to fire per cluster
        # Fixed: Only consider samples labeled as "fire" (within distance threshold)
        ax_avg_distance = axes[3, 0]
        average_distances = []
        cluster_ids_for_distances = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                # Only consider samples that are labeled as fire (true_labels == 1)
                # These are guaranteed to have distances ≤ DISTANCE_FILTER_THRESHOLD_M
                fire_samples_in_cluster_mask = (cluster_mask) & (true_labels == 1)

                if np.sum(fire_samples_in_cluster_mask) > 0:
                    fire_distances = distances[fire_samples_in_cluster_mask]
                    avg_distance = np.mean(fire_distances)
                else:
                    # No fire samples in this cluster
                    avg_distance = np.nan

                average_distances.append(avg_distance)
                cluster_ids_for_distances.append(cluster_id)

        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(average_distances)
        valid_cluster_ids = np.array(cluster_ids_for_distances)[valid_indices]
        valid_distances = np.array(average_distances)[valid_indices]

        if len(valid_distances) > 0:
            bars_distance = ax_avg_distance.bar(
                valid_cluster_ids,
                valid_distances,
                color=[
                    (viz_style.VALID_COLORS["orange"] if d < 10000 else viz_style.VALID_COLORS["blue"])
                    for d in valid_distances
                ],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
            )

            # Add value labels on bars
            for bar, value in zip(bars_distance, valid_distances):
                ax_avg_distance.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.0f}m",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        ax_avg_distance.set_title(
            "Average Distance to Fire per Cluster (Fire Samples Only)",
            fontweight="bold",
            fontsize=15,
        )
        ax_avg_distance.set_xlabel("Cluster Label", fontweight="bold")
        ax_avg_distance.set_ylabel("Average Distance (m)", fontweight="bold")
        ax_avg_distance.grid(True, alpha=0.3, linewidth=0.6)

        # Hide the unused subplot
        axes[3, 1].set_visible(False)

        plt.tight_layout()

        # Save outputs
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        self.logger.log_step(f"Clustering visualization saved to: {output_path}")

    def analyze_embeddings(
        self, dataloader: DataLoader, model: nn.Module, model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to perform complete embedding analysis.

        Args:
            dataloader: Data loader for analysis
            model: Pre-loaded autoencoder model (required)
            model_path: Path to trained model (optional, for logging purposes only)

        Returns:
            dict: Clustering results and metrics
        """
        self.logger.log_step("Starting embedding analysis")

        # Ensure model is on the correct device
        model.to(self.device)

        # Extract embeddings
        embeddings: np.ndarray
        window_ids: np.ndarray
        fire_ids: np.ndarray
        distances: np.ndarray
        true_labels: np.ndarray
        embeddings, window_ids, fire_ids, distances, true_labels = self.extract_embeddings(model, dataloader)

        # Reduce dimensionality for visualization
        reduced_embeddings: np.ndarray = self.reduce_dimensionality(
            embeddings, method=self.config.tuning.DIM_REDUCTION_METHOD
        )

        # Perform K-Means clustering
        kmeans_labels: np.ndarray
        kmeans_model: KMeans
        kmeans_silhouette: float
        kmeans_labels, kmeans_model, kmeans_silhouette = self.perform_kmeans_clustering(
            embeddings, n_clusters=self.config.tuning.KMEANS_N_CLUSTERS
        )

        # Perform DBSCAN clustering
        dbscan_labels: np.ndarray
        dbscan_model: DBSCAN
        dbscan_silhouette: float
        dbscan_labels, dbscan_model, dbscan_silhouette = self.perform_dbscan_clustering(
            embeddings,
            eps=self.config.tuning.DBSCAN_EPS,
            min_samples=self.config.tuning.DBSCAN_MIN_SAMPLES,
        )

        # Visualize results
        kmeans_output_path: str = os.path.join(self.config.paths.STATS_IMAGES_DIR, "kmeans_clustering_results.png")
        dbscan_output_path: str = os.path.join(self.config.paths.STATS_IMAGES_DIR, "dbscan_clustering_results.png")

        self.visualize_clustering_results(
            reduced_embeddings,
            kmeans_labels,
            true_labels,
            distances,
            "K-Means",
            kmeans_silhouette,
            kmeans_output_path,
        )
        self.visualize_clustering_results(
            reduced_embeddings,
            dbscan_labels,
            true_labels,
            distances,
            "DBSCAN",
            dbscan_silhouette,
            dbscan_output_path,
        )

        # Save clustering results
        results: Dict[str, Any] = {
            "embeddings": embeddings,
            "reduced_embeddings": reduced_embeddings,
            "window_ids": window_ids,
            "fire_ids": fire_ids,
            "distances": distances,
            "true_labels": true_labels,
            "kmeans_labels": kmeans_labels,
            "kmeans_silhouette": kmeans_silhouette,
            "dbscan_labels": dbscan_labels,
            "dbscan_silhouette": dbscan_silhouette,
            "kmeans_model": kmeans_model,
            "dbscan_model": dbscan_model,
        }

        # Save results to CSV
        results_df: pd.DataFrame = pd.DataFrame(
            {
                "window_id": window_ids,
                "fire_id": fire_ids,
                "distance": distances,
                "true_label": true_labels,
                "kmeans_cluster": kmeans_labels,
                "dbscan_cluster": dbscan_labels,
            }
        )

        os.makedirs(self.config.paths.STATS_CSV_DIR, exist_ok=True)
        results_df.to_csv(
            os.path.join(self.config.paths.STATS_CSV_DIR, "clustering_results.csv"),
            index=False,
        )

        self.logger.log_step("Embedding analysis completed successfully")
        self.logger.save_process_timeline()

        return results


# =========================================================================
# Main Function
# =========================================================================
def analyze_embeddings(
    config: Config,
    model: nn.Module,
    dataloader: Optional[DataLoader] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for embedding analysis.

    Args:
        config: Configuration object
        model: Pre-loaded autoencoder model (required)
        dataloader: Data loader for analysis (optional)
        model_path: Path to trained model (optional, for logging purposes only)

    Returns:
        dict: Clustering results from EmbeddingClusterAnalyzer
    """
    if dataloader is None:
        # If no dataloader is provided, create a default test loader
        _, _, dataloader = create_dataloaders(config, remove_fire_labels=False)

    analyzer: EmbeddingClusterAnalyzer = EmbeddingClusterAnalyzer(config)
    return analyzer.analyze_embeddings(dataloader, model, model_path)
