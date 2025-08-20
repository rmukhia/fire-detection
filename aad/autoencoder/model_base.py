"""Autoencoder models for forest fire anomaly detection.

Includes:
- AutoencoderBase: Abstract base class for all autoencoders
- DenseAutoencoder: Simple fully connected autoencoder
- VariationalAutoencoder: VAE with probabilistic latent space
- ConvolutionalVariationalAutoencoder: Advanced VAE with convolutional layers,
  attention mechanisms, residual connections, and feature matching loss
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn


class AutoencoderBase(nn.Module, ABC):
    """Abstract base class for all autoencoder models with a generic interface."""

    def __init__(self, time_steps: int, num_features: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    # --- Core encode/decode/forward interface ---
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Any:
        """
        Encode input tensor to latent representation or parameters.
        Args:
            x: Input tensor (batch, time_steps, num_features)
        Returns:
            Latent representation (AE) or (mu, logvar) tuple (VAE)
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed sequence.
        Args:
            z: Latent vector
        Returns:
            Reconstructed tensor (batch, time_steps, num_features)
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass through autoencoder.
        Args:
            x: Input tensor (batch, time_steps, num_features)
        Returns:
            Reconstructed tensor or tuple with additional outputs
        """
        pass

    # --- Embedding extraction ---
    @abstractmethod
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from input tensor.
        Args:
            x: Input tensor (batch, time_steps, num_features)
        Returns:
            torch.Tensor: Embeddings from encoder output
        """
        pass

    # --- Training/validation steps ---
    @abstractmethod
    def train_step(
        self, x: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, beta: float = 0.001, **kwargs
    ) -> Tuple[float, float, float, torch.Tensor, dict]:
        """
        Perform a single training step.
        Args:
            x: Input tensor
            optimizer: Optimizer for training
            loss_fn: Loss function
            beta: KL divergence weight (for VAEs, ignored for standard AEs)
            **kwargs: Additional arguments for extensibility
        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
        """
        pass

    @abstractmethod
    def val_step(
        self, x: torch.Tensor, loss_fn: nn.Module, beta: float = 0.001, **kwargs
    ) -> Tuple[float, float, float, torch.Tensor]:
        """
        Perform a single validation step.
        Args:
            x: Input tensor
            loss_fn: Loss function
            beta: KL divergence weight (for VAEs, ignored for standard AEs)
            **kwargs: Additional arguments for extensibility
        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
        """
        pass

    def initialize_weights(self):
        """
        Optional weight initialization for autoencoder models.
        Override in subclasses if custom initialization is needed.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
