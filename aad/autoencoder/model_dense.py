from .model_base import AutoencoderBase
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn


class DenseAutoencoder(AutoencoderBase):
    """Dense autoencoder model for sequence data.

    Args:
        time_steps: Number of timesteps in sequence
        num_features: Features per timestep
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, time_steps: int, num_features: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__(time_steps, num_features, latent_dim, hidden_dim)
        self.input_dim = time_steps * num_features  # Flattened size

        # Dense encoder/decoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent space.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            Latent representation (batch, latent_dim)
        """
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed sequence.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Reconstructed tensor (batch, time_steps, num_features)
        """
        recon_flat = self.decoder(z)
        recon = recon_flat.view(-1, self.time_steps, self.num_features)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            Reconstructed tensor with same shape as input
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def train_step(
        self, x: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, beta: float = 0.001, **kwargs
    ) -> Tuple[float, float, float, torch.Tensor, dict]:
        """Perform a single training step.

        Args:
            x: Input tensor
            optimizer: Optimizer for training
            loss_fn: Loss function
            beta: Unused parameter (kept for interface consistency)

        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
                  kl_loss is always 0 for standard autoencoder
        """
        import logging

        self.train()
        optimizer.zero_grad()

        # Forward returns tuple (recon,)
        recon = self.forward(x)
        recon_loss = loss_fn(recon, x)
        total_loss = recon_loss

        # Logging for interface validation
        logging.debug(
            f"[Dense train_step] total_loss={total_loss.item() if hasattr(total_loss,'item') else total_loss}, recon_loss={recon_loss.item() if hasattr(recon_loss,'item') else recon_loss}, kl_loss=0.0, beta={beta}"
        )

        total_loss.backward()
        optimizer.step()

        grad_stats = {}
        return (
            total_loss.item() if hasattr(total_loss, "item") else float(total_loss),
            recon_loss.item() if hasattr(recon_loss, "item") else float(recon_loss),
            0.0,
            recon,
            grad_stats,
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input tensor.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            torch.Tensor: Embeddings from encoder output
        """
        x_flat = x.view(x.size(0), -1)
        return self.encoder(x_flat)

    def val_step(
        self, x: torch.Tensor, loss_fn: nn.Module, beta: float = 0.001, **kwargs
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Perform a single validation step.

        Args:
            x: Input tensor
            loss_fn: Loss function
            beta: Unused parameter (kept for interface consistency)

        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
                  kl_loss is always 0 for standard autoencoder
        """
        import logging

        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            recon_loss = loss_fn(recon, x)
            total_loss = recon_loss

            # Logging for interface validation
            logging.debug(
                f"[Dense val_step] total_loss={total_loss.item() if hasattr(total_loss,'item') else total_loss}, recon_loss={recon_loss.item() if hasattr(recon_loss,'item') else recon_loss}, kl_loss=0.0, beta={beta}"
            )

        return (
            total_loss.item() if hasattr(total_loss, "item") else float(total_loss),
            recon_loss.item() if hasattr(recon_loss, "item") else float(recon_loss),
            0.0,
            recon,
        )

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for forest fire detection (DenseAutoencoder).

        High reconstruction error = Anomaly

        Args:
            x: Input sensor data (batch_size, time_steps, num_features)

        Returns:
            Anomaly scores (batch_size,) - higher scores indicate anomalies
        """
        self.eval()
        with torch.no_grad():
            # Get reconstruction
            recon = self.forward(x)

            # Reconstruction error (per sample)
            recon_error = torch.mean((recon - x) ** 2, dim=(1, 2))  # (batch_size,)

            # KL divergence is not used in DenseAutoencoder, set to zeros
            kl_div = torch.zeros_like(recon_error)

            # Combined anomaly score (normalize both components)
            recon_error_norm = (recon_error - recon_error.mean()) / (recon_error.std() + 1e-8)
            kl_div_norm = torch.zeros_like(recon_error_norm)

            # Weight reconstruction error more heavily for sensor data
            anomaly_scores = 0.7 * recon_error_norm + 0.3 * kl_div_norm

        return anomaly_scores
