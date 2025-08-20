"""Evaluation metrics and performance analysis for forest fire detection."""
import torch
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
)
from tqdm.auto import tqdm
import model_autoencoder
import model_utils
import core_logging
import viz_style
from pathlib import Path
from config import Config

class ModelEvaluator:
    """Handles model performance evaluation and metric calculation."""

    def __init__(self, config):
        self.config = config
        self.logger = core_logging.ProcessLogger(config, "Evaluation")
        self.device = torch.device(config.DEVICE)
        
        # Visualization Setup
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        # Load model and dataloaders
        _, _, self.test_loader = model_utils.create_dataloaders(self.config, remove_fire_labels=False)
        self.logger.log_step("Test data loaded", {
            'test_batches': len(self.test_loader),
            'test_samples': len(self.test_loader.dataset)
        })

        autoencoder_class = getattr(model_autoencoder, self.config.AUTOENCODER_CLASS)
        self.model = autoencoder_class(
            time_steps=self.config.WINDOW_SIZE,
            num_features=len(self.config.INPUT_COLUMNS),
            latent_dim=self.config.LATENT_DIM,
            hidden_dim=self.config.HIDDEN_DIM
        )

        self.model.load_state_dict(
            torch.load(self.config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.logger.log_step(
            "Model loaded successfully",
            {'model_path': self.config.BEST_MODEL_PATH}
        )

        self.model.to(self.device)
        self.model.eval()

    def _log_evaluation_statistics(
        self,
        reconstruction_errors: np.ndarray,
        true_labels: np.ndarray,
        predicted_anomalies: np.ndarray,
        threshold: float,
        precision: float,
        recall: float,
        f1: float,
        accuracy: float
    ) -> None:
        """Generate evaluation statistics visualizations. 
        
        Args:
            reconstruction_errors: Array of reconstruction error values
            true_labels: Array of ground truth labels
            predicted_anomalies: Array of predicted labels
            threshold: Threshold value for anomaly classification
            precision: Precision score
            recall: Recall score
            f1: F1 score
            accuracy: Accuracy score

        Returns:
            None: Saves visualization to output path
        """
        self.logger.log_step("Generating evaluation statistics")
        
        # Set publication-ready style
        viz_style.set_publication_style()
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forest Fire Detection - Evaluation Statistics', fontsize=18, fontweight='bold')
        
        # 1. Reconstruction error distribution
        ax1 = axes[0, 0]
        # Normal samples (true_labels == 0)
        normal_errors = reconstruction_errors[true_labels == 0]
        # Fire samples (true_labels == 1)
        fire_errors = reconstruction_errors[true_labels == 1]
        
        # Plot histograms
        if len(normal_errors) > 0:
            ax1.hist(normal_errors, bins=50, alpha=1, label=f'Normal Windows (n={len(normal_errors)})',
                    color=viz_style.VALID_COLORS['blue'], edgecolor='black', linewidth=0.8)
        
        if len(fire_errors) > 0:
            ax1.hist(fire_errors, bins=50, alpha=0.7, label=f'Fire Windows (n={len(fire_errors)})',
                    color=viz_style.VALID_COLORS['red'], edgecolor='black', linewidth=0.8)
        
        ax1.axvline(threshold, color=viz_style.VALID_COLORS['green'], linestyle='--', linewidth=2,
                    label=f'Threshold: {threshold:.4f}')
        
        ax1.set_title('Reconstruction Error Distribution', fontweight='bold', fontsize=15)
        ax1.set_xlabel('Reconstruction Error', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=False)
        ax1.grid(True, alpha=0.3, linewidth=0.6)
        ax1.set_yscale('log')  # Log scale for better visibility

        # 2. Confusion Matrix
        ax2 = axes[0, 1]
        cm = confusion_matrix(true_labels, predicted_anomalies)
        
        # Rearrange confusion matrix so True Positive is in top-left
        cm_rearranged = np.array([[cm[1,1], cm[1,0]],  # [TP, FN]
                                  [cm[0,1], cm[0,0]]])  # [FP, TN]
        
        # Create heatmap with seaborn
        sns.heatmap(cm_rearranged, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Fire', 'Predicted Normal'],
                    yticklabels=['Actual Fire', 'Actual Normal'], ax=ax2,
                    cbar_kws={'shrink': 0.8})
        
        # Add text annotations for clarity
        ax2.text(0.5, 0.2, f'TP\n{cm_rearranged[0,0]}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white' if cm_rearranged[0,0] > cm_rearranged.max()/2 else 'black')
        ax2.text(1.5, 0.2, f'FN\n{cm_rearranged[0,1]}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white' if cm_rearranged[0,1] > cm_rearranged.max()/2 else 'black')
        ax2.text(0.5, 1.2, f'FP\n{cm_rearranged[1,0]}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white' if cm_rearranged[1,0] > cm_rearranged.max()/2 else 'black')
        ax2.text(1.5, 1.2, f'TN\n{cm_rearranged[1,1]}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white' if cm_rearranged[1,1] > cm_rearranged.max()/2 else 'black')
        
        ax2.set_title('Confusion Matrix (TP in top-left)', fontweight='bold', fontsize=15)
        ax2.set_xlabel('Predicted Label', fontweight='bold')
        ax2.set_ylabel('Actual Label', fontweight='bold')
        
        # 3. Metrics Summary
        ax3 = axes[0, 2]
        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': accuracy
        }
        
        bars = ax3.bar(metrics.keys(), metrics.values(),
                        color=[viz_style.VALID_COLORS['blue'], viz_style.VALID_COLORS['orange'],
                               viz_style.VALID_COLORS['green'], viz_style.VALID_COLORS['red']],
                        alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_title('Model Performance Metrics', fontweight='bold', fontsize=15)
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, linewidth=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 4. ROC Curve
        ax4 = axes[1, 0]
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
        roc_auc = auc(fpr, tpr)
        
        ax4.plot(fpr, tpr, color=viz_style.VALID_COLORS['blue'], linewidth=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        # Add diagonal reference line
        ax4.plot([0, 1], [0, 1], color='#7f7f7f', linewidth=1.5, linestyle='--',
                label='Random Classifier')
        
        ax4.set_title(f'ROC Curve (AUC = {roc_auc:.3f})', fontweight='bold', fontsize=15)
        ax4.set_xlabel('False Positive Rate', fontweight='bold')
        ax4.set_ylabel('True Positive Rate', fontweight='bold')
        ax4.legend(frameon=True, fancybox=True, shadow=False)
        ax4.grid(True, alpha=0.3, linewidth=0.6)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        
        # 5. Threshold Analysis
        ax5 = axes[1, 1]
        # Test different threshold percentiles
        percentiles = np.arange(70, 99.95, 0.5)
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
        
        ax5.plot(percentiles, f1_scores, marker='o', label='F1-Score', linewidth=2, markersize=6)
        ax5.plot(percentiles, precisions, marker='s', label='Precision', linewidth=2, markersize=6)
        ax5.plot(percentiles, recalls, marker='^', label='Recall', linewidth=2, markersize=6)
        
        # Mark current threshold
        current_percentile = self.config.ANOMALY_THRESHOLD_PERCENTILE
        if current_percentile in percentiles:
            idx = np.where(percentiles == current_percentile)[0][0]
            ax5.axvline(current_percentile, color=viz_style.VALID_COLORS['green'], linestyle='--', linewidth=2,
                        label=f'Current: {current_percentile}%')
        
        ax5.set_title('Threshold Analysis: Metrics vs Percentile', fontweight='bold', fontsize=15)
        ax5.set_xlabel('Threshold Percentile', fontweight='bold')
        ax5.set_ylabel('Score', fontweight='bold')
        ax5.legend(frameon=True, fancybox=True, shadow=False)
        ax5.grid(True, alpha=0.3, linewidth=0.6)
        ax5.set_ylim([0, 1])
        
        # 6. Error Statistics
        ax6 = axes[1, 2]
        # Calculate statistics for normal and fire errors
        if len(normal_errors) > 0 and len(fire_errors) > 0:
            normal_stats = {
                'Mean': np.mean(normal_errors),
                'Median': np.median(normal_errors),
                'Std': np.std(normal_errors)
            }
            fire_stats = {
                'Mean': np.mean(fire_errors),
                'Median': np.median(fire_errors),
                'Std': np.std(fire_errors)
            }
            
            x = np.arange(len(normal_stats))
            width = 0.35
            
            ax6.bar(x - width/2, normal_stats.values(), width, label='Normal Windows',
                    color=viz_style.VALID_COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=0.8)
            ax6.bar(x + width/2, fire_stats.values(), width, label='Fire Windows',
                    color=viz_style.VALID_COLORS['red'], alpha=0.8, edgecolor='black', linewidth=0.8)
            
            ax6.set_title('Error Statistics Comparison', fontweight='bold', fontsize=15)
            ax6.set_xlabel('Statistic', fontweight='bold')
            ax6.set_ylabel('Value', fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(normal_stats.keys())
            ax6.legend(frameon=True, fancybox=True, shadow=False)
            ax6.grid(True, alpha=0.3, linewidth=0.6)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor statistics', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14, fontweight='bold')
            ax6.set_title('Error Statistics Comparison', fontweight='bold', fontsize=15)
        
        plt.tight_layout()
        
        # Save outputs
        os.makedirs(self.config.STATS_IMAGES_DIR, exist_ok=True)
        plt.savefig(self.config.EVALUATION_STATISTICS_IMAGE_PATH, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        self.logger.log_step(f"Evaluation statistics saved to: {self.config.EVALUATION_STATISTICS_IMAGE_PATH}")

    def evaluate_model(self):
        """
        Execute complete model evaluation workflow.
        
        Returns:
            tuple: (f1_score, precision, recall, threshold) evaluation metrics
        """
        self.logger.log_step("Starting model evaluation")

        reconstruction_errors, all_window_ids, all_fire_ids, all_distances = \
            self.compute_reconstruction_errors(self.model, self.test_loader)

        threshold = np.percentile(
            reconstruction_errors, self.config.ANOMALY_THRESHOLD_PERCENTILE
        )
        predicted_anomalies = reconstruction_errors > threshold

        self.logger.log_step("Anomaly threshold computed", {
            'threshold_percentile': self.config.ANOMALY_THRESHOLD_PERCENTILE,
            'threshold_value': float(threshold),
            'predicted_anomalies': int(np.sum(predicted_anomalies))
        })

        true_labels = self.create_true_labels(all_fire_ids, all_distances)

        f1, precision, recall, accuracy, specificity = \
            self.calculate_metrics(true_labels, predicted_anomalies)

        self._log_evaluation_statistics(
            reconstruction_errors, true_labels, predicted_anomalies,
            threshold, precision, recall, f1, accuracy
        )

        self.save_evaluation_results(
            reconstruction_errors, true_labels, predicted_anomalies,
            all_window_ids, threshold, precision, recall, f1, accuracy,
            specificity
        )

        self.logger.save_process_timeline()
        self.logger.save_metrics_plot()

        return f1, precision, recall, threshold

    def compute_reconstruction_errors(self, model, dataloader):
        """
        Calculate reconstruction errors for test data.
        
        Args:
            model: Trained autoencoder model
            dataloader: Test data loader
            
        Returns:
            tuple: (errors, window_ids, fire_ids, distances) arrays
        """
        reconstruction_errors = []
        all_window_ids = []
        all_fire_ids = []
        all_distances = []

        self.logger.log_step("Computing reconstruction errors")

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Computing reconstruction errors"):
                x, fire_id, distance, window_id = batch_data
                
                x = x.to(self.device)
                # Get reconstruction (first element) from standardized forward() output
                recon = model(x)[0]
                error = torch.mean((x - recon) ** 2, dim=(1, 2))
                reconstruction_errors.extend(error.cpu().numpy())
                all_window_ids.extend(window_id.cpu().numpy())
                all_fire_ids.extend(fire_id.cpu().numpy())
                all_distances.extend(distance.cpu().numpy())

        reconstruction_errors = np.array(reconstruction_errors)
        all_window_ids = np.array(all_window_ids)
        all_fire_ids = np.array(all_fire_ids)
        all_distances = np.array(all_distances)

        self.logger.log_step("Reconstruction errors computed", {
            'total_samples': len(reconstruction_errors),
            'mean_error': float(np.mean(reconstruction_errors)),
            'std_error': float(np.std(reconstruction_errors)),
            'min_error': float(np.min(reconstruction_errors)),
            'max_error': float(np.max(reconstruction_errors))
        })

        return reconstruction_errors, all_window_ids, all_fire_ids, all_distances

    def create_true_labels(self, all_fire_ids, all_distances):
        """
        Create binary labels based on fire proximity.
        
        Args:
            all_fire_ids: Array of fire IDs
            all_distances: Array of distances to fires
            
        Returns:
            np.array: Binary labels (1=fire, 0=normal)
        """
        self.logger.log_step("Creating true labels with fire IDs and distance filtering")
        
        potential_fire_mask = all_fire_ids > 0
        
        distance_filtered_fire_mask = (
            potential_fire_mask &
            (all_distances <= self.config.DISTANCE_FILTER_THRESHOLD_M)
        )
        
        true_labels = distance_filtered_fire_mask.astype(int)
        
        self.logger.log_step("True labels created with distance filtering", {
            'total_samples': len(true_labels),
            'potential_fire_samples': int(np.sum(potential_fire_mask)),
            'distance_filtered_fire_samples': int(np.sum(distance_filtered_fire_mask)),
            'normal_samples': int(np.sum(true_labels == 0)),
            'distance_threshold_m': self.config.DISTANCE_FILTER_THRESHOLD_M,
        })

        return true_labels

    def calculate_metrics(self, true_labels, predicted_anomalies):
        """
        Compute evaluation metrics from labels and predictions.
        
        Args:
            true_labels: Ground truth labels
            predicted_anomalies: Model predictions
            
        Returns:
            tuple: (f1, precision, recall, accuracy, specificity) metrics
        """
        true_positives = np.sum((true_labels == 1) & (predicted_anomalies == 1))
        false_positives = np.sum((true_labels == 0) & (predicted_anomalies == 1))
        false_negatives = np.sum((true_labels == 1) & (predicted_anomalies == 0))
        true_negatives = np.sum((true_labels == 0) & (predicted_anomalies == 0))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(true_labels) if len(true_labels) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

        self.logger.log_step("Evaluation metrics calculated", {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'specificity': float(specificity)
        })

        self.logger.log_info(
            f"TP: {true_positives}, FP: {false_positives}, "
            f"FN: {false_negatives}, TN: {true_negatives}"
        )
        self.logger.log_info(
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )
        self.logger.log_info(f"Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}")

        return f1, precision, recall, accuracy, specificity

    def save_evaluation_results(
            self, reconstruction_errors, true_labels, predicted_anomalies,
            all_window_ids, threshold, precision, recall, f1, accuracy,
            specificity):
        """
        Save evaluation results to CSV and JSON files.
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            true_labels: Ground truth labels
            predicted_anomalies: Model predictions
            all_window_ids: Window identifiers
            threshold: Anomaly threshold value
            precision: Precision score
            recall: Recall score
            f1: F1 score
            accuracy: Accuracy score
            specificity: Specificity score
        """
        results_df = pd.DataFrame({
            'window_id': all_window_ids,
            'reconstruction_error': reconstruction_errors,
            'true_label': true_labels,
            'predicted_anomaly': predicted_anomalies.astype(int),
            'is_correct': (
                true_labels == predicted_anomalies.astype(int)
            ).astype(int),
        })

        os.makedirs(self.config.STATS_CSV_DIR, exist_ok=True)

        results_df.to_csv(self.config.EVALUATION_RESULTS_CSV_PATH, index=False)

        summary = {
            'threshold': float(threshold),
            'threshold_percentile': self.config.ANOMALY_THRESHOLD_PERCENTILE,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'specificity': float(specificity),
            'total_samples': len(true_labels),
            'fire_samples': int(np.sum(true_labels)),
            'normal_samples': len(true_labels) - int(np.sum(true_labels)),
            'predicted_anomalies': int(np.sum(predicted_anomalies)),
        }

        with open(self.config.EVALUATION_SUMMARY_JSON_PATH, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.log_step(f"Evaluation results saved to {self.config.EVALUATION_RESULTS_CSV_PATH} and {self.config.EVALUATION_SUMMARY_JSON_PATH}")




# =========================================================================
# Main Function
# =========================================================================
def evaluate_model(config):
    """
    Main entry point for model evaluation.
    
    Args:
        config: Configuration object with evaluation parameters
        
    Returns:
        tuple: Evaluation metrics from ModelEvaluator
    """
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate_model()