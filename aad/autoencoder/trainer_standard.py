"""Training workflow implementation for forest fire detection autoencoder."""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
import matplotlib.container as mpl_container
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import logging
from tqdm.auto import tqdm
from aad.common.utils import set_seed
from aad.autoencoder import model_base
from aad.common import viz_style
from aad.common import core_logging
from aad.common.config import Config

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*gradient.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*NaN.*")


# =========================================================================
# Visualization Setup
# =========================================================================
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")

# =========================================================================
# Model Trainer Class
# =========================================================================
from aad.autoencoder.trainer_base import BaseTrainer


class StandardTrainer(BaseTrainer):
    """Standard training workflow for autoencoder models with comprehensive logging and visualization."""

    def __init__(
        self,
        random_seed: int,
        epochs: int,
        patience: int,
        learning_rate: float,
        loss_function_name: str,
        optimizer_name: str,
        batch_size: int,
        latent_dim: int,
        hidden_dim: int,
        window_size: int,
        num_features: int,
        device: str,
        stats_images_dir: str,
        training_statistics_image_path: str,
        best_model_path: str,
        loss_history_path: str,
        logger: core_logging.ProcessLogger,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(logger, device, callbacks)
        self.random_seed = random_seed
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.loss_function_name = loss_function_name
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_features = num_features
        self.stats_images_dir = stats_images_dir
        self.training_statistics_image_path = training_statistics_image_path
        self.best_model_path = best_model_path
        self.loss_history_path = loss_history_path
        plt.style.use("seaborn-v0_8-paper")
        sns.set_palette("husl")

    def _log_training_statistics(self, training_data: Dict[str, Any]) -> None:
        """Generate training statistics visualizations.

        Args:
            training_data: Dictionary containing training metrics
        """
        self.logger.log_step("Generating training statistics")

        # Set publication-ready style
        viz_style.set_publication_style()

        # Create figure with subplots
        fig: mpl_figure.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Forest Fire Detection - Training Statistics",
            fontsize=18,
            fontweight="bold",
        )

        # 1. Training and Validation Loss
        ax1: mpl_axes.Axes = axes[0, 0]
        epochs: List[int] = list(range(1, len(training_data["validation_losses"]) + 1))

        ax1.plot(
            epochs,
            training_data["validation_losses"],
            label="Validation Loss",
            marker="o",
            linewidth=2,
            markersize=6,
            color=viz_style.VALID_COLORS["blue"],
        )
        if "training_losses" in training_data and training_data["training_losses"]:
            ax1.plot(
                epochs,
                training_data["training_losses"],
                label="Training Loss",
                marker="s",
                linewidth=2,
                markersize=6,
                color=viz_style.VALID_COLORS["orange"],
            )

        # Mark best model point
        best_loss: float = training_data["best_loss"]
        best_epoch: int = training_data["validation_losses"].index(best_loss) + 1
        ax1.scatter(
            [best_epoch],
            [best_loss],
            color=viz_style.VALID_COLORS["red"],
            s=120,
            zorder=5,
            label=f"Best Model (Epoch {best_epoch})",
            marker="*",
            edgecolors="black",
            linewidth=0.8,
        )

        ax1.set_title("Model Training Progress", fontweight="bold", fontsize=15)
        ax1.set_xlabel("Epoch", fontweight="bold")
        ax1.set_ylabel("Loss", fontweight="bold")
        ax1.legend(frameon=True, fancybox=True, shadow=False, loc="upper right")
        ax1.grid(True, alpha=0.3, linewidth=0.6)

        # 2. Loss Distribution Histogram
        ax2: mpl_axes.Axes = axes[0, 1]
        val_losses: List[float] = training_data["validation_losses"]
        ax2.hist(
            val_losses,
            bins=min(20, len(val_losses) // 2 + 1),
            color=viz_style.VALID_COLORS["blue"],
            alpha=0.7,
            edgecolor=viz_style.VALID_COLORS["black"],
            linewidth=0.8,
        )
        ax2.axvline(
            best_loss,
            color=viz_style.VALID_COLORS["red"],
            linestyle="--",
            linewidth=2,
            label=f"Best Loss: {best_loss:.6f}",
        )
        ax2.set_title("Distribution of Validation Losses", fontweight="bold", fontsize=15)
        ax2.set_xlabel("Loss Value", fontweight="bold")
        ax2.set_ylabel("Frequency", fontweight="bold")
        ax2.legend(frameon=True, fancybox=True, shadow=False)
        ax2.grid(True, alpha=0.3, linewidth=0.6)

        # 3. Training Metrics
        ax3: mpl_axes.Axes = axes[1, 0]
        # Calculate statistics
        stats: Dict[str, Any] = {
            "Best Loss": training_data["best_loss"],
            "Final Loss": val_losses[-1],
            "Mean Loss": np.mean(val_losses),
            "Loss Std": np.std(val_losses),
        }

        bars: mpl_container.BarContainer = ax3.bar(
            list(stats.keys()),
            list(stats.values()),
            color=[
                viz_style.VALID_COLORS["blue"],
                viz_style.VALID_COLORS["orange"],
                viz_style.VALID_COLORS["green"],
                viz_style.VALID_COLORS["red"],
            ],
            alpha=0.8,
            edgecolor=viz_style.VALID_COLORS["black"],
            linewidth=0.8,
        )
        ax3.set_title("Training Summary Statistics", fontweight="bold", fontsize=15)
        ax3.set_ylabel("Value", fontweight="bold")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stats.values()) * 0.01,
                f"{value:.6f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        # 4. Loss Improvement
        ax4: mpl_axes.Axes = axes[1, 1]
        initial_loss: float = val_losses[0] if val_losses else 0.0
        improvement: float = ((initial_loss - best_loss) / initial_loss * 100) if initial_loss > 0 else 0.0

        metrics: Dict[str, Any] = {
            "Initial Loss": initial_loss,
            "Best Loss": best_loss,
            "Improvement %": improvement,
        }

        bars = ax4.bar(
            list(metrics.keys()),
            list(metrics.values()),
            color=[
                viz_style.VALID_COLORS["purple"],
                viz_style.VALID_COLORS["brown"],
                viz_style.VALID_COLORS["pink"],
            ],
            alpha=0.8,
            edgecolor=viz_style.VALID_COLORS["black"],
            linewidth=0.8,
        )
        ax4.set_title("Loss Improvement", fontweight="bold", fontsize=15)
        ax4.set_ylabel("Value", fontweight="bold")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3, linewidth=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            if bar.get_x() == 2:  # Improvement percentage
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(metrics.values()) * 0.01,
                    f"{value:.2f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )
            else:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(metrics.values()) * 0.01,
                    f"{value:.6f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )

        plt.tight_layout()

        # Save outputs
        os.makedirs(self.stats_images_dir, exist_ok=True)
        plt.savefig(
            self.training_statistics_image_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()

        self.logger.log_step(f"Training statistics saved to: {self.training_statistics_image_path}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: Optional[int] = None,
    ) -> float:
        """
        Execute one training epoch using model's train_step method, with callback hooks.
        """
        model.train()
        epoch_loss: float = 0.0
        num_batches: int = 0

        # Check if model has train_step method (required interface)
        try:
            model.train_step
        except AttributeError:
            self.logger.log_error(f"Model class {type(model).__name__} does not implement train_step().")
            raise AttributeError(f"Model class {type(model).__name__} does not implement train_step().")

        for batch_idx, (X, _, _, _) in enumerate(dataloader):
            X = X.to(self.device)
            optimizer.zero_grad()

            step_result: Tuple[Any, ...] = model.train_step(X, optimizer, loss_fn)  # type: ignore
            if len(step_result) == 5:
                loss_value, _, _, _, grad_stats = step_result
            else:
                # Backward compatibility: if grad_stats missing, use empty dict
                loss_value, _, _, grad_stats = step_result
                if not isinstance(grad_stats, dict):
                    grad_stats = {}

            if math.isnan(loss_value) or math.isinf(loss_value):
                if epoch is not None:
                    self.on_nan_detected(epoch, batch_idx, {"loss": loss_value})
                continue

            epoch_loss += loss_value
            num_batches += 1
            self.on_batch_end(batch_idx, {"loss": loss_value})

            if grad_stats and (
                grad_stats.get("grad_norm_before", 0) > 1000 or grad_stats.get("grad_max_before", 0) > 1000
            ):
                self.logger.log_info(
                    f"Gradient stats - Norm: {grad_stats['grad_norm_before']:.4f}, "
                    f"Max: {grad_stats['grad_max_before']:.4f}"
                )

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _val_epoch(self, dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> float:
        """
        Execute one validation epoch using model's val_step method.
        """
        model.eval()
        epoch_loss: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            for X, _, _, _ in dataloader:
                X = X.to(self.device)
                step_result: Tuple[Any, ...] = model.val_step(X, loss_fn)  # type: ignore
                if len(step_result) == 5:
                    loss_tensor, _, _, _, _ = step_result
                elif len(step_result) == 4:
                    loss_tensor, _, _, _ = step_result
                elif len(step_result) == 3:
                    # Unpack for the case where only loss, reconstruction, and anomaly scores are returned.
                    loss_tensor, _, _ = step_result
                else:
                    raise ValueError(f"Unexpected number of values returned by validation step: {len(step_result)}")

                epoch_loss += loss_tensor
                num_batches += 1

            return epoch_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_stats: Optional[bool] = False,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train the model using the provided data loaders, with callback hooks.
        """
        self.logger.log_step("Starting standard model training workflow")

        set_seed(self.random_seed)
        self.logger.log_step("Random seed set", {"seed": self.random_seed})

        # Ensure data loaders are provided
        if val_loader is None:
            raise ValueError("Validation loader must be provided")

        # Log provided data loaders
        self.logger.log_step(
            "Using provided data loaders",
            {
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "batch_size": self.batch_size,
            },
        )

        self._log_model_architecture(model)
        self.logger.log_step(
            "Model initialized",
            {
                "time_steps": self.window_size,
                "num_features": self.num_features,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
            },
        )

        self.logger.log_info(f"Training on: {self.device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.log_step(
                "GPU available",
                {"device_name": gpu_name, "memory_gb": f"{gpu_memory:.1f}"},
            )

        model.to(self.device)

        # Dynamically get loss function from config
        loss_fn = getattr(nn, self.loss_function_name)()

        # Dynamically get optimizer from config
        optimizer_class = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_class(model.parameters(), lr=self.learning_rate)

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.patience // 2, min_lr=1e-6
        )

        self.logger.log_step(
            "Optimizer configured",
            {
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "loss_function": self.loss_function_name,
                "scheduler": "ReduceLROnPlateau",
            },
        )

        best_loss: float = float("inf")
        loss_history: List[float] = []
        train_losses: List[float] = []
        no_improvement_count: int = 0
        nan_count: int = 0
        max_nan_allowed: int = 3

        self.logger.log_step("Training started", {"total_epochs": self.epochs})

        pbar: tqdm = tqdm(range(self.epochs), desc="Training Epochs", file=sys.stdout)

        epoch: int = 0
        val_loss: float = float("inf")

        for epoch in pbar:
            self.on_epoch_start(epoch)
            train_loss: float = self.train_epoch(train_loader, model, loss_fn, optimizer, epoch=epoch)
            train_losses.append(train_loss)

            val_loss = self._val_epoch(val_loader, model, loss_fn)
            loss_history.append(val_loss)

            postfix_data: Dict[str, str] = {
                "Train Loss": f"{train_loss:.6f}",
                "Val Loss": f"{val_loss:.6f}",
                "Best Loss": f"{best_loss:.6f}",
                "No Improve": str(no_improvement_count),
            }

            pbar.set_postfix(postfix_data)

            # Handle NaN losses - if validation loss is NaN, handle gracefully
            if math.isnan(val_loss) or math.isinf(val_loss):
                nan_count += 1
                self.logger.log_warning(
                    f"NaN/Inf validation loss detected at epoch {epoch} (count: {nan_count}/{max_nan_allowed})"
                )
                self.on_nan_detected(epoch, -1, {"val_loss": val_loss})
                if nan_count >= max_nan_allowed:
                    self.logger.log_error("Too many NaN losses - stopping training")
                    postfix_data = {
                        "Train Loss": f"{train_loss:.6f}",
                        "Val Loss": f"{val_loss:.6f}",
                        "Best Loss": f"{best_loss:.6f}",
                        "No Improve": str(no_improvement_count),
                        "Status": "Too many NaN!",
                    }
                    pbar.set_postfix(postfix_data)
                    break
                continue

            if val_loss < best_loss:
                best_loss = val_loss
                os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
                torch.save(model.state_dict(), self.best_model_path)
                no_improvement_count = 0
                postfix_data = {
                    "Train Loss": f"{train_loss:.6f}",
                    "Val Loss": f"{val_loss:.6f}",
                    "Best Loss": f"{best_loss:.6f}",
                    "No Improve": str(no_improvement_count),
                    "Status": "New Best!",
                }
                pbar.set_postfix(postfix_data)
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.patience:
                postfix_data = {
                    "Train Loss": f"{train_loss:.6f}",
                    "Val Loss": f"{val_loss:.6f}",
                    "Best Loss": f"{best_loss:.6f}",
                    "No Improve": str(no_improvement_count),
                    "Status": "Early Stop",
                }
                pbar.set_postfix(postfix_data)
                break

            # Update learning rate scheduler
            scheduler.step(val_loss)

            self.on_epoch_end(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_loss": best_loss,
                    "no_improvement_count": no_improvement_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )

        self.logger.log_step(
            "Training completed",
            {
                "final_epochs": epoch + 1,
                "best_validation_loss": best_loss,
                "final_validation_loss": (val_loss if not math.isnan(val_loss) else float("nan")),
                "improvement_from_start": (
                    f"{( (loss_history[0] - best_loss) / loss_history[0] * 100):.2f}%"
                    if not math.isnan(loss_history[0])
                    else "N/A"
                ),
                "nan_count": nan_count,
            },
        )

        training_data = {
            "validation_losses": loss_history,
            "training_losses": train_losses,
            "best_loss": best_loss,
            "total_epochs": len(loss_history),
            "nan_count": nan_count,
            "config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "time_steps": self.window_size,
                "num_features": self.num_features,
            },
        }

        if save_stats:
            os.makedirs(os.path.dirname(self.loss_history_path), exist_ok=True)
            with open(self.loss_history_path, "w") as f:
                json.dump(training_data, f, indent=2)

            self._log_training_statistics(training_data)

        self.logger.save_process_timeline()
        self.logger.save_metrics_plot()

        self.logger.log_info("Training completed! Check the output directory for detailed logs and visualizations.")

        return model, training_data

    def validate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Validate the model on the provided data loader.

        Args:
            model: The neural network model to validate
            dataloader: DataLoader for validation data

        Returns:
            dict: Validation metrics including loss
        """
        self.logger.log_step("Starting model validation")

        model.to(self.device)
        loss_fn = getattr(nn, self.loss_function_name)()

        val_loss = self._val_epoch(dataloader, model, loss_fn)

        return {
            "validation_loss": val_loss,
            "device": str(self.device),
            "loss_function": self.loss_function_name,
        }