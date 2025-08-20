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


from .model_blocks import ConvEncoderBlock, ConvDecoderBlock, ResidualAdd, LatentAttention, TransformerEncoderBlock


class AutoencoderBase(nn.Module, ABC):
    """Abstract base class for all autoencoder models."""

    def __init__(self, time_steps: int, num_features: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through autoencoder.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            Reconstructed tensor or tuple with additional outputs
        """
        pass

    @abstractmethod
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input tensor.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            torch.Tensor: Embeddings from encoder output
        """
        pass

    @abstractmethod
    def train_step(
        self, x: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, beta: float = 0.001
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Perform a single training step.

        Args:
            x: Input tensor
            optimizer: Optimizer for training
            loss_fn: Loss function
            beta: KL divergence weight (for VAEs)

        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction, ...)
        """
        pass

    @abstractmethod
    def val_step(
        self, x: torch.Tensor, loss_fn: nn.Module, beta: float = 0.001
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Perform a single validation step.

        Args:
            x: Input tensor
            loss_fn: Loss function
            beta: KL divergence weight (for VAEs)

        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction, ...)
        """
        pass


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            Reconstructed tensor with same shape as input
        """
        # Flatten to (batch, input_dim)
        x_flat = x.view(x.size(0), -1)

        # Encode & decode
        z = self.encoder(x_flat)
        recon_flat = self.decoder(z)

        # Reshape back to (batch, time_steps, num_features)
        recon = recon_flat.view(-1, self.time_steps, self.num_features)
        return recon

    def train_step(
        self, x: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, beta: float = 0.001
    ) -> Tuple[float, float, float, torch.Tensor]:
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
        self.train()
        optimizer.zero_grad()

        # Forward returns tuple (recon,)
        recon = self.forward(x)
        recon_loss = loss_fn(recon, x)
        total_loss = recon_loss

        total_loss.backward()
        optimizer.step()

        return total_loss, recon_loss.item(), 0.0, recon

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
        self, x: torch.Tensor, loss_fn: nn.Module, beta: float = 0.001
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
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            recon_loss = loss_fn(recon, x)
            total_loss = recon_loss

        return total_loss, recon_loss.item(), 0.0, recon


class VariationalAutoencoder(AutoencoderBase):
    """Variational Autoencoder (VAE) with probabilistic latent space.

    Args:
        time_steps: Number of timesteps in sequence
        num_features: Features per timestep
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, time_steps: int, num_features: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__(time_steps, num_features, latent_dim, hidden_dim)
        self.input_dim = time_steps * num_features  # Flattened size

        # Encoder after transformer
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * time_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, time_steps, num_features)
        x_embed = self.input_proj(x)  # (batch, time_steps, embed_dim)

        # Transformer encoder stack
        x_trans = self.transformer(x_embed)  # (batch, time_steps, embed_dim)

        # Flatten before encoder
        x_flat = x_trans.reshape(x.size(0), -1)

        # Encode
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize + Decode
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decoder(z)
        recon = recon_flat.view(-1, self.time_steps, self.num_features)

        return recon, mu, logvar


class ConvolutionalVariationalAutoencoder(AutoencoderBase):
    """Advanced VAE with convolutional layers, attention mechanisms, residual connections, and feature matching loss.

    Args:
        time_steps: Number of timesteps in sequence
        num_features: Features per timestep
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of convolutional layers
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        use_attention: Whether to use attention mechanism
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        time_steps: int,
        num_features: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        use_attention: bool,
        use_residual: bool,
    ) -> None:
        super().__init__(time_steps, num_features, latent_dim, hidden_dim)

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_residual_layers = nn.ModuleList()  # For residual connections
        in_channels = num_features
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - i - 1))
            self.encoder_layers.append(ConvEncoderBlock(in_channels, out_channels, kernel_size, dropout))
            if use_residual:
                self.encoder_residual_layers.append(ResidualAdd(in_channels, out_channels))
            in_channels = out_channels

        # Attention mechanism in latent space
        if use_attention:
            self.attention = LatentAttention(latent_dim, num_heads=4)

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim * time_steps, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * time_steps, latent_dim)

        # Projection from latent space back to decoder input
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * time_steps)

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        self.decoder_residual_layers = nn.ModuleList()  # For residual connections
        in_channels = hidden_dim
        for i in range(num_layers):
            out_channels = hidden_dim // (2**i)
            self.decoder_layers.append(ConvDecoderBlock(in_channels, out_channels, kernel_size, dropout))
            if use_residual:
                # The residual connection for decoder comes from the encoder's output
                # The in_channels for ResidualAdd here should be the encoder_feature's channels
                # which is out_channels of the corresponding encoder layer
                encoder_out_channels = hidden_dim // (2 ** (num_layers - (num_layers - i - 1) - 1))
                self.decoder_residual_layers.append(ResidualAdd(encoder_out_channels, out_channels))
            in_channels = out_channels

        # Final output layer
        self.final_layer = nn.Conv1d(in_channels, num_features, kernel_size, padding=kernel_size // 2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through convolutional VAE.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            tuple: (reconstructed, mu, logvar)
        """
        batch_size = x.size(0)

        # Permute to (batch, channels, time_steps) for conv1d
        x_permuted = x.permute(0, 2, 1)

        # Encoder forward pass
        encoder_features = []
        h = x_permuted

        for i, layer in enumerate(self.encoder_layers):
            h_input_to_block = h  # Store input for residual connection
            h = layer(h)
            if self.use_residual:
                h = self.encoder_residual_layers[i](h, h_input_to_block)
            encoder_features.append(h)

        # Flatten for latent space
        h_flat = h.view(batch_size, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        # Apply attention in latent space if enabled
        if self.use_attention:
            z = self.reparameterize(mu, logvar)
            z = self.attention(z)
        else:
            z = self.reparameterize(mu, logvar)

        # Project back to decoder input shape using linear layer
        x_flat = self.decoder_fc(z)
        decoder_input = x_flat.reshape(batch_size, -1, self.time_steps)

        # Decoder forward pass
        h = decoder_input
        for i, layer in enumerate(self.decoder_layers):
            if self.use_residual and i < len(encoder_features):
                residual_idx = len(encoder_features) - i - 1
                encoder_feature = encoder_features[residual_idx]
                h = self.decoder_residual_layers[i](layer(h), encoder_feature)
            else:
                h = layer(h)

        # Final output
        recon_permuted = self.final_layer(h)
        recon = recon_permuted.permute(0, 2, 1)  # Back to (batch, time_steps, num_features)

        return recon, mu, logvar

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings (mean of latent distribution) from input tensor.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            torch.Tensor: Mean of latent distribution
        """
        batch_size = x.size(0)
        x_permuted = x.permute(0, 2, 1)

        # Encoder forward pass
        h = x_permuted
        for layer in self.encoder_layers:
            h = layer(h)

        # Get latent mean
        h_flat = h.view(batch_size, -1)
        return self.fc_mu(h_flat)

    def train_step(
        self, x: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, beta: float = 0.001
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Perform a single training step.

        Args:
            x: Input tensor
            optimizer: Optimizer for training
            loss_fn: Loss function
            beta: KL divergence weight

        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
        """
        self.train()
        optimizer.zero_grad()

        # Forward pass
        recon, mu, logvar = self.forward(x)

        # Calculate losses
        recon_loss = loss_fn(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss

        total_loss.backward()
        optimizer.step()

        return total_loss, recon_loss.item(), kl_loss.item(), recon

    def val_step(
        self, x: torch.Tensor, loss_fn: nn.Module, beta: float = 0.001
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Perform a single validation step.

        Args:
            x: Input tensor
            loss_fn: Loss function
            beta: KL divergence weight

        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
        """
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)

            recon_loss = loss_fn(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss.item(), kl_loss.item(), recon
