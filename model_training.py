"""Training workflow implementation for forest fire detection autoencoder."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import json
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from tqdm.auto import tqdm
import model_utils
import model_autoencoder
import viz_style
import core_logging
from config import Config
from pathlib import Path

# =========================================================================
# Visualization Setup
# =========================================================================
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# =========================================================================
# Model Trainer Class
# =========================================================================
class ModelTrainer:
    """Handles complete training lifecycle for autoencoder model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = core_logging.ProcessLogger(config, "Training")
        self.device = torch.device(config.DEVICE)
        
        # Visualization Setup
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

    def _log_model_architecture(self, model: nn.Module) -> None:
        """Log model architecture details.
        
        Args:
            model: The model to log details for
        """
        self.logger.log_info("Model Architecture:")
        self.logger.log_info(f"Model type: {type(model).__name__}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        self.logger.log_info(f"Total parameters: {total_params:,}")
        self.logger.log_info(f"Trainable parameters: {trainable_params:,}")

        # Log layer details
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                if hasattr(module, 'weight'):
                    self.logger.log_info(
                        f"Layer {name}: {module}, "
                        f"Parameters: {module.weight.numel():,}"
                    )
                else:
                    self.logger.log_info(f"Layer {name}: {module}")

    def _log_training_statistics(self, training_data: Dict[str, Any]) -> None:
        """Generate training statistics visualizations.
        
        Args:
            training_data: Dictionary containing training metrics
        """
        self.logger.log_step("Generating training statistics")
        
        # Set publication-ready style
        viz_style.set_publication_style()
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Forest Fire Detection - Training Statistics', fontsize=18, fontweight='bold')
        
        # 1. Training and Validation Loss
        ax1 = axes[0, 0]
        epochs = list(range(1, len(training_data['validation_losses']) + 1))
        
        ax1.plot(epochs, training_data['validation_losses'],
                label='Validation Loss', marker='o', linewidth=2, markersize=6,
                color=viz_style.VALID_COLORS['blue'])
        if 'training_losses' in training_data and training_data['training_losses']:
            ax1.plot(epochs, training_data['training_losses'],
                    label='Training Loss', marker='s', linewidth=2, markersize=6,
                    color=viz_style.VALID_COLORS['orange'])
        
        # Mark best model point
        best_loss = training_data['best_loss']
        best_epoch = training_data['validation_losses'].index(best_loss) + 1
        ax1.scatter([best_epoch], [best_loss], color=viz_style.VALID_COLORS['red'], s=120, zorder=5,
                   label=f'Best Model (Epoch {best_epoch})', marker='*', edgecolors='black', linewidth=0.8)
        
        ax1.set_title('Model Training Progress', fontweight='bold', fontsize=15)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=False, loc='upper right')
        ax1.grid(True, alpha=0.3, linewidth=0.6)
        
        # 2. Loss Distribution Histogram
        ax2 = axes[0, 1]
        val_losses = training_data['validation_losses']
        ax2.hist(val_losses, bins=min(20, len(val_losses)//2 + 1),
                color=viz_style.VALID_COLORS['blue'], alpha=0.7, edgecolor=viz_style.VALID_COLORS['black'], linewidth=0.8)
        ax2.axvline(best_loss, color=viz_style.VALID_COLORS['red'], linestyle='--', linewidth=2,
                   label=f'Best Loss: {best_loss:.6f}')
        ax2.set_title('Distribution of Validation Losses', fontweight='bold', fontsize=15)
        ax2.set_xlabel('Loss Value', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.legend(frameon=True, fancybox=True, shadow=False)
        ax2.grid(True, alpha=0.3, linewidth=0.6)
        
        # 3. Training Metrics
        ax3 = axes[1, 0]
        # Calculate statistics
        stats = {
            'Best Loss': training_data['best_loss'],
            'Final Loss': val_losses[-1],
            'Mean Loss': np.mean(val_losses),
            'Loss Std': np.std(val_losses)
        }
        
        bars = ax3.bar(stats.keys(), stats.values(),
                      color=[viz_style.VALID_COLORS['blue'], viz_style.VALID_COLORS['orange'],
                             viz_style.VALID_COLORS['green'], viz_style.VALID_COLORS['red']],
                      alpha=0.8, edgecolor=viz_style.VALID_COLORS['black'], linewidth=0.8)
        ax3.set_title('Training Summary Statistics', fontweight='bold', fontsize=15)
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, linewidth=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats.values())*0.01,
                    f'{value:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 4. Loss Improvement
        ax4 = axes[1, 1]
        initial_loss = val_losses[0] if val_losses else 0
        improvement = ((initial_loss - best_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        metrics = {
            'Initial Loss': initial_loss,
            'Best Loss': best_loss,
            'Improvement %': improvement
        }
        
        bars = ax4.bar(metrics.keys(), metrics.values(),
                      color=[viz_style.VALID_COLORS['purple'], viz_style.VALID_COLORS['brown'],
                             viz_style.VALID_COLORS['pink']],
                      alpha=0.8, edgecolor=viz_style.VALID_COLORS['black'], linewidth=0.8)
        ax4.set_title('Loss Improvement', fontweight='bold', fontsize=15)
        ax4.set_ylabel('Value', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, linewidth=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            if bar.get_x() == 2:  # Improvement percentage
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values())*0.01,
                        f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
            else:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values())*0.01,
                        f'{value:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        
        # Save outputs
        os.makedirs(self.config.STATS_IMAGES_DIR, exist_ok=True)
        plt.savefig(self.config.TRAINING_STATISTICS_IMAGE_PATH, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        self.logger.log_step(f"Training statistics saved to: {self.config.TRAINING_STATISTICS_IMAGE_PATH}")

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Execute one training epoch using model's train_step method.
        
        Args:
            dataloader: Training data loader
            model: Model to train
            loss_fn: Loss function
            optimizer: Optimization algorithm
            
        Returns:
            float: Average training loss for the epoch
        """
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for X, _, _, _ in dataloader:
            X = X.to(self.device)
            loss, _, _, _ = model.train_step(X, optimizer, loss_fn)
                
            epoch_loss += loss
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _val_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_fn: nn.Module
    ) -> float:
        """
        Execute one validation epoch using model's val_step method.
        
        Args:
            dataloader: Validation data loader
            model: Model to evaluate
            loss_fn: Loss function
            
        Returns:
            float: Average validation loss for the epoch
        """
        model.eval()
        epoch_loss = 0.0
        num_batches = 0

        for X, _, _, _ in dataloader:
            X = X.to(self.device)
            loss, _, _, _ = model.val_step(X, loss_fn)
                
            epoch_loss += loss
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def train_model(self) -> Tuple[nn.Module, Dict[str, Union[List[float], Dict[str, Any]]]]:
        """
        Execute complete training process.
        
        Returns:
            tuple: (trained_model, training_metrics) where:
                trained_model: Best performing model
                training_metrics: Dictionary containing loss history and config
        """
        self.logger.log_step("Starting model training")

        model_utils.set_seed(self.config.RANDOM_SEED)
        self.logger.log_step("Random seed set", {'seed': self.config.RANDOM_SEED})

        train_loader, val_loader, _ = model_utils.create_dataloaders(self.config, remove_fire_labels=True)

        self.logger.log_step("Data loaders created", {
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'batch_size': self.config.BATCH_SIZE,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset)
        })

        # Dynamically get the autoencoder class from config
        autoencoder_class = getattr(model_autoencoder, self.config.AUTOENCODER_CLASS)
        model = autoencoder_class(
            time_steps=self.config.WINDOW_SIZE,
            num_features=len(self.config.INPUT_COLUMNS),
            latent_dim=self.config.LATENT_DIM,
            hidden_dim=self.config.HIDDEN_DIM
        )

        self._log_model_architecture(model)
        self.logger.log_step("Model initialized", {
            'time_steps': self.config.WINDOW_SIZE,
            'num_features': len(self.config.INPUT_COLUMNS),
            'latent_dim': self.config.LATENT_DIM,
            'hidden_dim': self.config.HIDDEN_DIM
        })

        self.logger.log_info(f"Training on: {self.device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.log_step("GPU available", {
                'device_name': gpu_name,
                'memory_gb': f"{gpu_memory:.1f}"
            })

        model.to(self.device)
        # Dynamically get loss function from config
        loss_fn = getattr(nn, self.config.LOSS_FUNCTION)()
        
        # Dynamically get optimizer from config
        optimizer_class = getattr(torch.optim, self.config.OPTIMIZER)
        optimizer = optimizer_class(model.parameters(), lr=self.config.LEARNING_RATE)

        self.logger.log_step("Optimizer configured", {
            'optimizer': 'Adam',
            'learning_rate': self.config.LEARNING_RATE,
            'loss_function': 'MSELoss'
        })

        best_loss = float('inf')
        loss_history = []
        train_losses = []
        no_improvement_count = 0

        self.logger.log_step("Training started", {'total_epochs': self.config.EPOCHS})

        pbar = tqdm(range(self.config.EPOCHS), desc="Training Epochs", file=sys.stdout)

        for epoch in pbar:
            train_loss = self.train_epoch(train_loader, model, loss_fn, optimizer)
            train_losses.append(train_loss)

            val_loss = self._val_epoch(val_loader, model, loss_fn)
            loss_history.append(val_loss)

            pbar.set_postfix({
                'Train Loss': f'{train_loss:.6f}',
                'Val Loss': f'{val_loss:.6f}',
                'Best Loss': f'{best_loss:.6f}',
                'No Improve': no_improvement_count
            })

            if val_loss < best_loss:
                best_loss = val_loss
                os.makedirs(os.path.dirname(self.config.BEST_MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), self.config.BEST_MODEL_PATH)
                no_improvement_count = 0
                pbar.set_postfix({
                    'Train Loss': f'{train_loss:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'Best Loss': f'{best_loss:.6f}',
                    'No Improve': no_improvement_count,
                    'Status': 'New Best!'
                })
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.config.PATIENCE:
                pbar.set_postfix({
                    'Train Loss': f'{train_loss:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'Best Loss': f'{best_loss:.6f}',
                    'No Improve': no_improvement_count,
                    'Status': 'Early Stop'
                })
                break

        self.logger.log_step("Training completed", {
            'final_epochs': epoch + 1,
            'best_validation_loss': best_loss,
            'final_validation_loss': val_loss,
            'improvement_from_start':
                f"{( (loss_history[0] - best_loss) / loss_history[0] * 100):.2f}%"
        })

        training_data = {
            'validation_losses': loss_history,
            'training_losses': train_losses,
            'best_loss': best_loss,
            'total_epochs': len(loss_history),
            'config': {
                'learning_rate': self.config.LEARNING_RATE,
                'batch_size': self.config.BATCH_SIZE,
                'latent_dim': self.config.LATENT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'time_steps': self.config.WINDOW_SIZE,
                'num_features': len(self.config.INPUT_COLUMNS)
            }
        }

        os.makedirs(os.path.dirname(self.config.LOSS_HISTORY_PATH), exist_ok=True)
        with open(self.config.LOSS_HISTORY_PATH, 'w') as f:
            json.dump(training_data, f, indent=2)

        self._log_training_statistics(training_data)

        self.logger.save_process_timeline()
        self.logger.save_metrics_plot()

        self.logger.log_info(
            "Training completed! Check the output directory for detailed logs and visualizations."
        )

        return model, training_data

# =========================================================================
# Main Function
# =========================================================================
def train_autoencoder(config: Config) -> Tuple[nn.Module, Dict[str, Union[List[float], Dict[str, Any]]]]:
    """
    Main entry point for model training.
    
    Args:
        config: Configuration object with training parameters
        
    Returns:
        tuple: (trained_model, training_metrics) from ModelTrainer
    """
    trainer = ModelTrainer(config)
    return trainer.train_model()