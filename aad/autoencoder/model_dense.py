from .model_base import AutoencoderBase
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import math


class Autoencoder(AutoencoderBase):
    """
    Standard Autoencoder for sequence reconstruction,
    using a Transformer for the encoder and a simple MLP for the decoder.
    """

    def __init__(
        self,
        time_steps: int,
        num_features: int,
        latent_dim: int,
        d_model: int = 8,
        num_heads: int = 4,  # must be a divisor of d_model
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the Autoencoder model.

        Args:
            time_steps (int): The number of time steps in the input sequence.
            num_features (int): The number of features per time step.
            latent_dim (int): The dimensionality of the latent space.
            d_model (int): The embedding dimension for the transformer.
            num_heads (int): The number of attention heads in the transformer.
            num_layers (int): The number of transformer encoder layers.
            dropout (float): The dropout rate.
        """
        super().__init__(time_steps, num_features, latent_dim, num_features)
        
        self.output_dim = time_steps * num_features

        # POSITIONAL ENCODING
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=time_steps)
        
        # TRANSFORMER ENCODER (operating on 3-dimensional feature space)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              
            nhead=num_heads,
            dim_feedforward= d_model * 4, # normally set to 4 times d_model
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ATTENTION POOLING
        self.attention_pooling = AttentionPooling(d_model)
        
        # LATENT SPACE MAPPING
        # A single layer to map the pooled output to the latent dimension
        self.fc_latent = nn.Linear(d_model, latent_dim)

        # DECODER (Simple MLP from latent back to full sequence)
        hidden_dim = max(64, latent_dim * 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

        self.initialize_weights()


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to a single latent vector.
        
        Args:
            x: [batch_size, time_steps, num_features] - e.g., [batch, 60, 3]
        Returns:
            latent_vector: [batch_size, latent_dim]
        """
        # Add positional encoding to input
        x = self.pos_encoding(x)

        # Transform with attention
        x = self.transformer(x)

        # Attention pooling to get a single vector per sequence
        x = self.attention_pooling(x)

        # Map to the latent space
        latent_vector = self.fc_latent(x)
        return latent_vector

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to sequence.
        
        Args:
            z: [batch_size, latent_dim]
        Returns:
            recon: [batch_size, time_steps, num_features]
        """
        batch_size = z.size(0)
        x = self.decoder(z)
        x = x.view(batch_size, self.time_steps, self.num_features)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Autoencoder."""
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings (the latent representation)."""
        latent = self.encode(x)
        return latent

    def train_step(
        self, 
        x: torch.Tensor, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module, 
        **kwargs
    ) -> Tuple[float, float, torch.Tensor, dict]:
        """Training step for the Autoencoder."""
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        recon, _ = self.forward(x)
        
        # Reconstruction loss is the only loss
        total_loss = loss_fn(recon, x)
        
        # Backward pass
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        grad_stats = {
            "grad_norm": float(grad_norm),
        }
        
        return total_loss.item(), 0.0, recon, grad_stats

    def val_step(
        self, 
        x: torch.Tensor, 
        loss_fn: nn.Module, 
        **kwargs
    ) -> Tuple[float, float, torch.Tensor]:
        """Validation step for the Autoencoder."""
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            total_loss = loss_fn(recon, x)
        
        return total_loss.item(), 0.0, recon

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error.
        
        Args:
            x: [batch_size, time_steps, num_features]
        Returns:
            anomaly_scores: [batch_size] - higher scores indicate anomalies
        """
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            # The anomaly score is simply the Mean Squared Error of reconstruction.
            # We're no longer using KL divergence, as this is not a VAE.
            anomaly_scores = torch.mean((recon - x) ** 2, dim=(1, 2))
        return anomaly_scores


class AttentionPooling(nn.Module):
    """Learnable attention pooling for sequence aggregation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        hidden_dim = max(d_model // 2, 1)
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to sequence.
        
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            pooled: [batch_size, d_model]
        """
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        pooled = torch.sum(x * attention_weights, dim=1)
        
        return pooled


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding applied directly to input features."""

    def __init__(self, num_features: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_features = num_features
        
        pe = torch.zeros(max_len, num_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, num_features, 1).float() * (-math.log(10000.0) / num_features))
        
        for i in range(num_features):
            if i % 2 == 0:
                pe[:, i] = torch.sin(position.squeeze() * div_term[i])
            else:
                pe[:, i] = torch.cos(position.squeeze() * div_term[i])
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input features.
        
        Args:
            x: [batch_size, seq_len, num_features]
        Returns:
            x with positional encoding added
        """
        pe = getattr(self, "pe")
        x = x + pe[:, :x.size(1), :]
        return self.dropout(x)
