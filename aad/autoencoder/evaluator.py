"""Evaluation metrics and performance analysis for forest fire detection."""

import torch
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from tqdm.auto import tqdm
from . import model_base
from ..data.utils import create_dataloaders
from ..common import core_logging
from ..common import viz_style

from ..common.config import Config
from typing import Tuple, List, Dict, Any, Optional
from torch.utils.data import DataLoader


import torch.nn as nn


class ModelEvaluator:
    """Handles model performance evaluation and metric calculation."""

    def __init__(
        self,
        distance_filter_threshold_m: float,
        device: torch.device,
        logger: core_logging.ProcessLogger,
        model: nn.Module,
        test_loader: DataLoader,
        stats_images_dir: str,
        stats_csv_dir: str,
        eval_stats_img_path: str,
        eval_results_csv_path: str,
        eval_summary_json_path: str,
    ) -> None:
        """
        Initialize the model evaluator.

        Args:
            device: Torch device to use.
            logger: Logger instance.
            model: Trained model to evaluate.
            test_loader: DataLoader for test data.
            stats_images_dir: Directory for saving stats images.
            stats_csv_dir: Directory for saving stats CSVs.
            eval_stats_img_path: Path for evaluation statistics image.
            eval_results_csv_path: Path for evaluation results CSV.
            eval_summary_json_path: Path for evaluation summary JSON.
            distance_filter_threshold_m: Distance threshold for fire label.
        """
        self.device = device
        self.logger = logger
        self.model = model
        self.test_loader = test_loader
        self.stats_images_dir = stats_images_dir
        self.stats_csv_dir = stats_csv_dir
        self.eval_stats_img_path = eval_stats_img_path
        self.eval_results_csv_path = eval_results_csv_path
        self.eval_summary_json_path = eval_summary_json_path
        self.distance_filter_threshold_m = distance_filter_threshold_m
        plt.style.use("seaborn-v0_8-paper")
        sns.set_palette("husl")
        self.model.to(self.device)
        self.model.eval()
        self.logger.log_info(
            f"ModelEvaluator initialized with device={self.device}, model={type(self.model).__name__}, "
            f"test_loader={type(self.test_loader).__name__}"
        )

    def _log_evaluation_statistics(
        self,
        reconstruction_errors: np.ndarray,
        true_labels: np.ndarray,
        predicted_anomalies: np.ndarray,
        threshold: float,
        precision: float,
        recall: float,
        f1: float,
        accuracy: float,
    ) -> None:
        """
        Generate and save evaluation statistics visualizations.

        Args:
            reconstruction_errors (np.ndarray): Array of reconstruction error values.
            true_labels (np.ndarray): Array of ground truth labels.
            predicted_anomalies (np.ndarray): Array of predicted labels.
            threshold (float): Threshold value for anomaly classification.
            precision (float): Precision score.
            recall (float): Recall score.
            f1 (float): F1 score.
            accuracy (float): Accuracy score.

        Returns:
            None
        """

        # Set publication-ready style
        viz_style.set_publication_style()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Forest Fire Detection - Evaluation Statistics",
            fontsize=18,
            fontweight="bold",
        )

        # 1. Reconstruction error distribution
        ax1 = axes[0, 0]
        # Normal samples (true_labels == 0)
        normal_errors = reconstruction_errors[true_labels == 0]
        # Fire samples (true_labels == 1)
        fire_errors = reconstruction_errors[true_labels == 1]

        # Plot histograms
        if len(normal_errors) > 0:
            ax1.hist(
                normal_errors,
                bins=50,
                alpha=1,
                label=f"Normal Windows (n={len(normal_errors)})",
                color=viz_style.VALID_COLORS["blue"],
                edgecolor="black",
                linewidth=0.8,
            )

        if len(fire_errors) > 0:
            ax1.hist(
                fire_errors,
                bins=50,
                alpha=0.7,
                label=f"Fire Windows (n={len(fire_errors)})",
                color=viz_style.VALID_COLORS["red"],
                edgecolor="black",
                linewidth=0.8,
            )

        ax1.axvline(
            threshold,
            color=viz_style.VALID_COLORS["green"],
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {threshold:.4f}",
        )

        ax1.set_title("Reconstruction Error Distribution", fontweight="bold", fontsize=15)
        ax1.set_xlabel("Reconstruction Error", fontweight="bold")
        ax1.set_ylabel("Frequency", fontweight="bold")
        ax1.legend(frameon=True, fancybox=True, shadow=False)
        ax1.grid(True, alpha=0.3, linewidth=0.6)
        ax1.set_yscale("log")  # Log scale for better visibility

        # 2. Confusion Matrix
        ax2 = axes[0, 1]
        cm = confusion_matrix(true_labels, predicted_anomalies)

        # Rearrange confusion matrix so True Positive is in top-left
        cm_rearranged = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])  # [TP, FN]  # [FP, TN]

        # Create heatmap with seaborn
        sns.heatmap(
            cm_rearranged,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Fire", "Predicted Normal"],
            yticklabels=["Actual Fire", "Actual Normal"],
            ax=ax2,
            cbar_kws={"shrink": 0.8},
        )

        # Add text annotations for clarity
        ax2.text(
            0.5,
            0.2,
            f"TP\n{cm_rearranged[0,0]}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if cm_rearranged[0, 0] > cm_rearranged.max() / 2 else "black",
        )
        ax2.text(
            1.5,
            0.2,
            f"FN\n{cm_rearranged[0,1]}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if cm_rearranged[0, 1] > cm_rearranged.max() / 2 else "black",
        )
        ax2.text(
            0.5,
            1.2,
            f"FP\n{cm_rearranged[1,0]}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if cm_rearranged[1, 0] > cm_rearranged.max() / 2 else "black",
        )
        ax2.text(
            1.5,
            1.2,
            f"TN\n{cm_rearranged[1,1]}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if cm_rearranged[1, 1] > cm_rearranged.max() / 2 else "black",
        )

        ax2.set_title("Confusion Matrix (TP in top-left)", fontweight="bold", fontsize=15)
        ax2.set_xlabel("Predicted Label", fontweight="bold")
        ax2.set_ylabel("Actual Label", fontweight="bold")

        # 3. Metrics Summary
        ax3 = axes[0, 2]
        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Accuracy": accuracy,
        }

        bars = ax3.bar(
            metrics.keys(),
            metrics.values(),
            color=[
                viz_style.VALID_COLORS["blue"],
                viz_style.VALID_COLORS["orange"],
                viz_style.VALID_COLORS["green"],
                viz_style.VALID_COLORS["red"],
            ],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )
        ax3.set_title("Model Performance Metrics", fontweight="bold", fontsize=15)
        ax3.set_ylabel("Score", fontweight="bold")
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

        # 4. ROC Curve
        ax4 = axes[1, 0]
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
        roc_auc = auc(fpr, tpr)

        ax4.plot(
            fpr,
            tpr,
            color=viz_style.VALID_COLORS["blue"],
            linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        # Add diagonal reference line
        ax4.plot(
            [0, 1],
            [0, 1],
            color="#7f7f7f",
            linewidth=1.5,
            linestyle="--",
            label="Random Classifier",
        )

        ax4.set_title(f"ROC Curve (AUC = {roc_auc:.3f})", fontweight="bold", fontsize=15)
        ax4.set_xlabel("False Positive Rate", fontweight="bold")
        ax4.set_ylabel("True Positive Rate", fontweight="bold")
        ax4.legend(frameon=True, fancybox=True, shadow=False)
        ax4.grid(True, alpha=0.3, linewidth=0.6)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])

        # 5. Threshold Analysis
        ax5 = axes[1, 1]
        # Test different threshold percentiles
        percentiles = np.append(np.arange(1, 99.98, 5), (self.anomaly_threshold_percentile, 99.98))
        percentiles = np.unique(percentiles)  # avoid duplicate if 99.98 is already present
        f1_scores = []
        precisions = []
        recalls = []

        for p in percentiles:
            thresh = np.percentile(reconstruction_errors, p)
            pred = reconstruction_errors > thresh

            if np.sum(pred) > 0:  # Avoid division by zero
                f1_val = f1_score(true_labels, pred, zero_division=0)
                precision_val = precision_score(true_labels, pred, zero_division=0)
                recall_val = recall_score(true_labels, pred, zero_division=0)
            else:
                f1_val = precision_val = recall_val = 0

            f1_scores.append(f1_val)
            precisions.append(precision_val)
            recalls.append(recall_val)

        ax5.plot(
            percentiles,
            f1_scores,
            marker="o",
            label="F1-Score",
            linewidth=2,
            markersize=6,
        )
        ax5.plot(
            percentiles,
            precisions,
            marker="s",
            label="Precision",
            linewidth=2,
            markersize=6,
        )
        ax5.plot(percentiles, recalls, marker="^", label="Recall", linewidth=2, markersize=6)

        # Mark current threshold
        current_percentile = self.anomaly_threshold_percentile
        if current_percentile in percentiles:
            idx = np.where(percentiles == current_percentile)[0][0]
            ax5.axvline(
                current_percentile,
                color=viz_style.VALID_COLORS["green"],
                linestyle="--",
                linewidth=2,
                label=f"Current: {current_percentile}%",
            )

        ax5.set_title("Threshold Analysis: Metrics vs Percentile", fontweight="bold", fontsize=15)
        ax5.set_xlabel("Threshold Percentile", fontweight="bold")
        ax5.set_ylabel("Score", fontweight="bold")
        ax5.legend(frameon=True, fancybox=True, shadow=False)
        ax5.grid(True, alpha=0.3, linewidth=0.6)
        ax5.set_ylim([0, 1])

        # 6. Error Statistics
        ax6 = axes[1, 2]
        # Calculate statistics for normal and fire errors
        if len(normal_errors) > 0 and len(fire_errors) > 0:
            normal_stats = {
                "Mean": np.mean(normal_errors),
                "Median": np.median(normal_errors),
                "Std": np.std(normal_errors),
            }
            fire_stats = {
                "Mean": np.mean(fire_errors),
                "Median": np.median(fire_errors),
                "Std": np.std(fire_errors),
            }

            x = np.arange(len(normal_stats))
            width = 0.35

            ax6.bar(
                x - width / 2,
                normal_stats.values(),
                width,
                label="Normal Windows",
                color=viz_style.VALID_COLORS["blue"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
            )
            ax6.bar(
                x + width / 2,
                fire_stats.values(),
                width,
                label="Fire Windows",
                color=viz_style.VALID_COLORS["red"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
            )

            ax6.set_title("Error Statistics Comparison", fontweight="bold", fontsize=15)
            ax6.set_xlabel("Statistic", fontweight="bold")
            ax6.set_ylabel("Value", fontweight="bold")
            ax6.set_xticks(x)
            ax6.set_xticklabels(normal_stats.keys())
            ax6.legend(frameon=True, fancybox=True, shadow=False)
            ax6.grid(True, alpha=0.3, linewidth=0.6)
        else:
            ax6.text(
                0.5,
                0.5,
                "Insufficient data\nfor statistics",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=14,
                fontweight="bold",
            )
            ax6.set_title("Error Statistics Comparison", fontweight="bold", fontsize=15)

        plt.tight_layout()

        # Save outputs
        os.makedirs(self.stats_images_dir, exist_ok=True)
        plt.savefig(
            self.eval_stats_img_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()

        self.logger.log_step(f"Evaluation statistics saved to: {self.eval_stats_img_path}")

    def _find_optimal_threshold_internal(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        all_window_ids: np.ndarray,
        metric: str = "f1",
        fallback_percentile: float = 95.0
    ) -> Dict[str, Any]:
        """Internal helper method for finding optimal threshold."""
        if len(np.unique(true_labels)) < 2:
            self.logger.log_warning("Dataset contains only one class. Cannot compute metrics.")
            return {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "specificity": 0.0,
                "threshold_percentile": None,
                "threshold": None,
                "true_positive_window_ids": [],
            }

        percentiles_to_test = np.arange(1, 99.99, 0.1)
        best_val = -1
        optimal_percentile = None
        optimal_threshold_value = None
        best_metrics = {}
        found = False
        
        for p in percentiles_to_test:
            current_threshold = float(np.percentile(anomaly_scores, p))
            preds = anomaly_scores > current_threshold
            f1, precision, recall, accuracy, specificity = self.calculate_metrics(true_labels, preds)
            metric_map = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "specificity": specificity,
            }
            val = metric_map.get(metric, f1)
            if val > best_val:
                best_val = val
                optimal_percentile = p
                optimal_threshold_value = current_threshold
                best_metrics = {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "specificity": specificity,
                    "threshold": current_threshold,
                    "threshold_percentile": p,
                    "true_positive_window_ids": all_window_ids[(true_labels == 1) & (preds == 1)].tolist(),
                }
                found = True

        if found:
            self.logger.log_step(
                "Optimal threshold found",
                {
                    "optimal_percentile": optimal_percentile,
                    "optimal_threshold_value": optimal_threshold_value,
                    "best_metric": metric,
                    "best_metric_value": best_val,
                },
            )
            return best_metrics
        else:
            # fallback to fixed threshold
            threshold_float = float(np.percentile(anomaly_scores, fallback_percentile))
            preds = anomaly_scores > threshold_float
            f1, precision, recall, accuracy, specificity = self.calculate_metrics(true_labels, preds)
            self.logger.log_step(
                "Optimal threshold not found, fallback to fixed",
                {
                    "threshold_percentile": fallback_percentile,
                    "threshold_value": threshold_float,
                },
            )
            return {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "specificity": specificity,
                "threshold": threshold_float,
                "threshold_percentile": fallback_percentile,
                "true_positive_window_ids": all_window_ids[(true_labels == 1) & (preds == 1)].tolist(),
            }

    def evaluate_model(
        self,
        anomaly_scores: np.ndarray,
        all_window_ids: np.ndarray,
        all_fire_ids: np.ndarray,
        all_distances: np.ndarray,
        anomaly_threshold_percentile: float = 95.0,
        save_stats: Optional[bool] = False,
        threshold_mode: str = "fixed",
        metric: str = "f1"
    ) -> Dict[str, float]:
        """
        Execute model evaluation workflow using anomaly scores.

        Args:
            anomaly_scores: np.ndarray of anomaly scores.
            all_window_ids: np.ndarray of window IDs.
            all_fire_ids: np.ndarray of fire IDs.
            all_distances: np.ndarray of distances.
            anomaly_threshold_percentile: Percentile for fixed threshold.
            save_stats: Whether to save statistics.
            threshold_mode: "fixed" or "optimal".
            metric: Metric to optimize for "optimal" mode ("f1", "precision", "recall", "accuracy", "specificity").

        Returns:
            Dict[str, float]: Evaluation metrics including f1_score, precision, recall, accuracy, specificity, and threshold.
        """
        self.logger.log_step("Starting model evaluation")
        true_labels = self.create_true_labels(all_fire_ids, all_distances)

        threshold_float: float = float(np.percentile(anomaly_scores, anomaly_threshold_percentile))
        predicted_anomalies: np.ndarray = anomaly_scores > threshold_float

        if threshold_mode == "fixed":
            self.anomaly_threshold_percentile = anomaly_threshold_percentile
            threshold_float = float(np.percentile(anomaly_scores, anomaly_threshold_percentile))
            predicted_anomalies = anomaly_scores > threshold_float
            self.logger.log_step(
                "Anomaly threshold computed (fixed)",
                {
                    "threshold_percentile": anomaly_threshold_percentile,
                    "threshold_value": threshold_float,
                    "predicted_anomalies": int(np.sum(predicted_anomalies)),
                },
            )
        elif threshold_mode == "optimal":
            result = self._find_optimal_threshold_internal(
                anomaly_scores,
                true_labels,
                all_window_ids,
                metric,
                anomaly_threshold_percentile
            )
            self.anomaly_threshold_percentile = result["threshold_percentile"]
            threshold_float = result["threshold"]
            predicted_anomalies = anomaly_scores > threshold_float
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        f1, precision, recall, accuracy, specificity = self.calculate_metrics(true_labels, predicted_anomalies)
        true_positives_mask = (true_labels == 1) & (predicted_anomalies == 1)
        true_positive_window_ids = all_window_ids[true_positives_mask]

        if save_stats:
            self._log_evaluation_statistics(
                anomaly_scores,
                true_labels,
                predicted_anomalies,
                threshold_float,
                precision,
                recall,
                f1,
                accuracy,
            )
            self.save_evaluation_results(
                anomaly_scores,
                true_labels,
                predicted_anomalies,
                all_window_ids,
                threshold_float,
                precision,
                recall,
                f1,
                accuracy,
                specificity,
            )

        self.logger.save_process_timeline()
        self.logger.save_metrics_plot()

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "specificity": specificity,
            "threshold": threshold_float,
            "true_positive_window_ids": true_positive_window_ids.tolist(),
        }

    def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate anomaly scores for test data using model.compute_anomaly_score.

        Args:
            model (nn.Module): Trained autoencoder model.
            dataloader (DataLoader): Test data loader.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (scores, window_ids, fire_ids, distances) arrays.
        """
        model = self.model
        dataloader = self.test_loader
        anomaly_scores: List[float] = []
        all_window_ids: List[int] = []
        all_fire_ids: List[int] = []
        all_distances: List[float] = []

        self.logger.log_step("Computing anomaly scores")

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Computing anomaly scores"):
                x, fire_id, distance, window_id = batch_data

                x = x.to(self.device)
                # Removed excessive compute_anomaly_score call log
                if not hasattr(model, "compute_anomaly_score"):
                    raise AttributeError(
                        f"Model {type(model).__name__} missing required compute_anomaly_score method"
                    )
                
                compute_fn = model.compute_anomaly_score
                if not callable(compute_fn):
                    raise TypeError(
                        f"Model {type(model).__name__} compute_anomaly_score is not callable. "
                        f"Got type: {type(compute_fn)}"
                    )
                
                try:
                    scores = compute_fn(x)
                except Exception as e:
                    self.logger.log_error(
                        f"Failed to compute anomaly scores on {type(model).__name__}: {str(e)}"
                    )
                    raise ValueError(f"Anomaly score computation failed: {str(e)}") from e
                # Removed excessive compute_anomaly_score output log
                anomaly_scores.extend(scores.cpu().numpy())
                all_window_ids.extend(window_id.cpu().numpy())
                all_fire_ids.extend(fire_id.cpu().numpy())
                all_distances.extend(distance.cpu().numpy())

        anomaly_scores_arr: np.ndarray = np.array(anomaly_scores)
        all_window_ids_arr: np.ndarray = np.array(all_window_ids)
        all_fire_ids_arr: np.ndarray = np.array(all_fire_ids)
        all_distances_arr: np.ndarray = np.array(all_distances)

        self.logger.log_step(
            "Anomaly scores computed",
            {
                "total_samples": len(anomaly_scores_arr),
                "mean_score": float(np.mean(anomaly_scores_arr)),
                "std_score": float(np.std(anomaly_scores_arr)),
                "min_score": float(np.min(anomaly_scores_arr)),
                "max_score": float(np.max(anomaly_scores_arr)),
            },
        )

        return (
            anomaly_scores_arr,
            all_window_ids_arr,
            all_fire_ids_arr,
            all_distances_arr,
        )

    def create_true_labels(self, all_fire_ids: np.ndarray, all_distances: np.ndarray) -> np.ndarray:
        """
        Create binary labels based on fire proximity.

        Args:
            all_fire_ids (np.ndarray): Array of fire IDs.
            all_distances (np.ndarray): Array of distances to fires.

        Returns:
            np.ndarray: Binary labels (1=fire, 0=normal).
        """
        self.logger.log_step("Creating true labels with fire IDs and distance filtering")

        potential_fire_mask: np.ndarray = all_fire_ids > 0

        distance_filtered_fire_mask: np.ndarray = potential_fire_mask & (
            all_distances <= self.distance_filter_threshold_m
        )

        true_labels: np.ndarray = distance_filtered_fire_mask.astype(int)

        self.logger.log_step(
            "True labels created with distance filtering",
            {
                "total_samples": len(true_labels),
                "potential_fire_samples": int(np.sum(potential_fire_mask)),
                "distance_filtered_fire_samples": int(np.sum(distance_filtered_fire_mask)),
                "normal_samples": int(np.sum(true_labels == 0)),
                "distance_threshold_m": self.distance_filter_threshold_m,
            },
        )

        return true_labels

    def calculate_metrics(
        self, true_labels: np.ndarray, predicted_anomalies: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute evaluation metrics from labels and predictions.

        Args:
            true_labels (np.ndarray): Ground truth labels.
            predicted_anomalies (np.ndarray): Model predictions.

        Returns:
            Tuple[float, float, float, float, float]: (f1, precision, recall, accuracy, specificity) metrics.
        """
        true_positives: int = np.sum((true_labels == 1) & (predicted_anomalies == 1))
        false_positives: int = np.sum((true_labels == 0) & (predicted_anomalies == 1))
        false_negatives: int = np.sum((true_labels == 1) & (predicted_anomalies == 0))
        true_negatives: int = np.sum((true_labels == 0) & (predicted_anomalies == 0))

        precision: float = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        )
        recall: float = (
            true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        )
        f1: float = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy: float = (true_positives + true_negatives) / len(true_labels) if len(true_labels) > 0 else 0
        specificity: float = (
            true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        )

        # self.logger.log_step(
        #     "Evaluation metrics calculated",
        #     {
        #         "true_positives": int(true_positives),
        #         "false_positives": int(false_positives),
        #         "false_negatives": int(false_negatives),
        #         "true_negatives": int(true_negatives),
        #         "precision": float(precision),
        #         "recall": float(recall),
        #         "f1_score": float(f1),
        #         "accuracy": float(accuracy),
        #         "specificity": float(specificity),
        #     },
        # )

        # Removed excessive metrics summary log (TP, FP, FN, TN)
        # Removed excessive metrics summary logs (Precision, Recall, F1-Score, Accuracy, Specificity)

        return f1, precision, recall, accuracy, specificity

    def get_confusion_and_metrics(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Compute confusion matrix and main classification metrics.

        Args:
            true_labels (np.ndarray): Ground truth labels.
            predicted_labels (np.ndarray): Predicted labels.

        Returns:
            Tuple containing:
                - confusion matrix (np.ndarray)
                - precision (float)
                - recall (float)
                - accuracy (float)
                - f1_score (float)
        """
        cm = confusion_matrix(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        accuracy = (true_labels == predicted_labels).mean()
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        # Ensure all metrics are Python floats for type safety
        return (
            cm,
            float(precision),
            float(recall),
            float(accuracy),
            float(f1),
        )

    def save_evaluation_results(
        self,
        reconstruction_errors: np.ndarray,
        true_labels: np.ndarray,
        predicted_anomalies: np.ndarray,
        all_window_ids: np.ndarray,
        threshold: float,
        precision: float,
        recall: float,
        f1: float,
        accuracy: float,
        specificity: float,
    ) -> None:
        """
        Save evaluation results to CSV and JSON files.

        Args:
            reconstruction_errors (np.ndarray): Array of reconstruction errors.
            true_labels (np.ndarray): Ground truth labels.
            predicted_anomalies (np.ndarray): Model predictions.
            all_window_ids (np.ndarray): Window identifiers.
            threshold (float): Anomaly threshold value.
            precision (float): Precision score.
            recall (float): Recall score.
            f1 (float): F1 score.
            accuracy (float): Accuracy score.
            specificity (float): Specificity score.

        Returns:
            None
        """
        results_df: pd.DataFrame = pd.DataFrame(
            {
                "window_id": all_window_ids,
                "reconstruction_error": reconstruction_errors,
                "true_label": true_labels,
                "predicted_anomaly": predicted_anomalies.astype(int),
                "is_correct": (true_labels == predicted_anomalies.astype(int)).astype(int),
            }
        )

        os.makedirs(self.stats_csv_dir, exist_ok=True)

        results_df.to_csv(self.eval_results_csv_path, index=False)

        summary: Dict[str, Any] = {
            "threshold": float(threshold),
            "threshold_percentile": self.anomaly_threshold_percentile,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "specificity": float(specificity),
            "total_samples": len(true_labels),
            "fire_samples": int(np.sum(true_labels)),
            "normal_samples": len(true_labels) - int(np.sum(true_labels)),
            "predicted_anomalies": int(np.sum(predicted_anomalies)),
        }

        with open(self.eval_summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.log_step(
            f"Evaluation results saved to {self.eval_results_csv_path} and {self.eval_summary_json_path}"
        )

    def find_optimal_threshold(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        all_window_ids: np.ndarray,
        metric: str = "f1",
        fallback_percentile: float = 95.0
    ) -> Dict[str, Any]:
        """
        Finds the optimal anomaly threshold percentile that maximizes the selected metric.

        Args:
            anomaly_scores: np.ndarray of anomaly scores.
            true_labels: np.ndarray of true labels.
            all_window_ids: np.ndarray of window IDs.
            metric: Metric to optimize ("f1", "precision", "recall", "accuracy", "specificity").
            fallback_percentile: Percentile to use if no optimal threshold is found.

        Returns:
            Dictionary containing metrics and optimal threshold info.
        """
        self.logger.log_info("Starting search for optimal anomaly threshold.")
        return self._find_optimal_threshold_internal(
            anomaly_scores,
            true_labels,
            all_window_ids,
            metric,
            fallback_percentile
        )