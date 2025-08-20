from .model_base import AutoencoderBase
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import math


class VariationalAutoencoder(AutoencoderBase):
    """
    Variational Autoencoder with a transformer-based encoder and decoder.

    The encoder uses attention pooling to aggregate the transformer's output
    into a latent vector, while the decoder uses a transformer to reconstruct
    the sequence from this latent vector.
    """

    def __init__(
        self,
        time_steps: int,
        num_features: int,
        latent_dim: int,
        d_model: int = 8,
        num_heads: int = 4,  # must be divisor of d_model
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(time_steps, num_features, latent_dim, num_features)

        self.output_dim = time_steps * num_features
        self.d_model = d_model

        # --- ENCODER ARCHITECTURE ---
        # Add projection from num_features to d_model for the encoder
        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding for the encoder input
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=time_steps)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              
            nhead=num_heads, 
            dim_feedforward= d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling to create a single vector from the sequence
        self.attention_pooling = AttentionPooling(d_model)
        
        # Latent space mapping from pooled vector
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # --- DECODER ARCHITECTURE ---
        # Projection from latent vector to a sequence for the decoder
        # The latent vector [batch, latent_dim] is projected to a sequence-like shape
        # [batch, time_steps, d_model]. The decoder then works on this sequence.
        self.latent_to_sequence = nn.Linear(latent_dim, time_steps * d_model)

        # Positional encoding for the decoder input
        self.decoder_pos_encoding = PositionalEncoding(d_model, dropout, max_len=time_steps)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The transformer decoder will operate on the sequence projected from the latent space
        # and has a memory of the encoded input sequence, but for a VAE it's simpler to
        # make it a single-stream decoder, so we pass no memory.
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection from d_model back to num_features
        self.output_projection = nn.Linear(d_model, num_features)

        self.initialize_weights()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent parameters.
        
        Args:
            x: [batch_size, time_steps, num_features]
        Returns:
            mu, logvar: [batch_size, latent_dim]
        """
        # Reproject to d_model
        x = self.input_projection(x)  # [batch, 60, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)  # [batch, 60, d_model]

        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch, 60, d_model]

        # Attention pooling to get a single vector
        x = self.attention_pooling(x)  # [batch, d_model]

        # Map to latent space parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to sequence using a transformer decoder.
        
        Args:
            z: [batch_size, latent_dim]
        Returns:
            recon: [batch_size, time_steps, num_features]
        """
        batch_size = z.size(0)
        
        # Project latent vector to a sequence format for the decoder
        x = self.latent_to_sequence(z)  # [batch, time_steps * d_model]
        x = x.view(batch_size, self.time_steps, self.d_model) # [batch, time_steps, d_model]

        # Add positional encoding to the decoder input
        x = self.decoder_pos_encoding(x)

        # Transformer decoder to generate the sequence
        # We pass the input to both tgt and memory as this is a simple autoencoder setup
        # and the decoder is effectively creating the output sequence from the latent features.
        recon_sequence = self.transformer_decoder(tgt=x, memory=x)
        
        # Project back to the original feature dimension
        recon = self.output_projection(recon_sequence)  # [batch, time_steps, num_features]

        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings (latent means)."""
        mu, _ = self.encode(x)
        return mu

    def train_step(
        self, 
        x: torch.Tensor, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module, 
        beta: float = 0.01,
        **kwargs
    ) -> Tuple[float, float, float, torch.Tensor, dict]:
        """Training step with KL annealing."""
        import logging

        self.train()
        optimizer.zero_grad()
        
        # Get training parameters
        kl_annealing = kwargs.get("kl_annealing", True)
        epoch = kwargs.get("epoch", 0)
        max_epochs = kwargs.get("max_epochs", 100)
        
        # Forward pass
        recon, mu, logvar = self.forward(x)
        
        # Reconstruction loss
        recon_loss = loss_fn(recon, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Beta scheduling for KL annealing
        if kl_annealing:
            annealing_factor = min(1.0, 2.0 * epoch / max_epochs)
            effective_beta = beta * annealing_factor
        else:
            effective_beta = beta
        
        # Total loss
        total_loss = recon_loss + effective_beta * kl_loss
        
        # Backward pass
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging and stats
        logging.debug(
            f"[VAE train_step] total_loss={total_loss.item():.6f}, "
            f"recon_loss={recon_loss.item():.6f}, kl_loss={kl_loss.item():.6f}, "
            f"beta={effective_beta:.6f}"
        )
        
        grad_stats = {
            "grad_norm": float(grad_norm),
            "effective_beta": effective_beta,
        }
        
        return total_loss.item(), recon_loss.item(), kl_loss.item(), recon, grad_stats

    def val_step(
        self, 
        x: torch.Tensor, 
        loss_fn: nn.Module, 
        beta: float = 0.01,
        **kwargs
    ) -> Tuple[float, float, float, torch.Tensor]:
        """Validation step."""
        import logging

        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            
            recon_loss = loss_fn(recon, x)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            total_loss = recon_loss + beta * kl_loss

            logging.debug(
                f"[VAE val_step] total_loss={total_loss.item():.6f}, "
                f"recon_loss={recon_loss.item():.6f}, kl_loss={kl_loss.item():.6f}"
            )

        return total_loss.item(), recon_loss.item(), kl_loss.item(), recon

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for forest fire detection.
        
        Combines reconstruction error with KL divergence and feature weighting
        for PM2.5, CO, and RH sensors.
        
        Args:
            x: [batch_size, time_steps, num_features] where features are [PM2.5, CO, RH]
        Returns:
            anomaly_scores: [batch_size] - higher scores indicate anomalies
        """
        import logging
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            logging.debug(f"[anomaly_score] recon shape: {recon.shape}, x shape: {x.shape}")
            # 1. Overall reconstruction error (MSE)
            mse_error = torch.mean((recon - x) ** 2, dim=(1, 2))  # [batch_size]
            
            # 2. Feature-weighted reconstruction error
            # Weight PM2.5 and CO more heavily (fire indicators)
            feature_weights = torch.tensor([1.0, 0.3, 0.7], device=x.device)  # [PM2.5, CO, RH]
            if x.shape[-1] != 3:
                logging.error(f"[anomaly_score] Expected 3 features, got {x.shape[-1]}")
            weighted_error = torch.mean(
                torch.mean((recon - x) ** 2, dim=1) * feature_weights, dim=1
            )
            
            # 3. KL divergence (latent space anomaly)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Normalize components using robust statistics
            def robust_normalize(tensor):
                median = torch.median(tensor)
                mad = torch.median(torch.abs(tensor - median))
                return (tensor - median) / (mad + 1e-8)
            
            mse_norm = robust_normalize(mse_error)
            weighted_norm = robust_normalize(weighted_error)
            kl_norm = robust_normalize(kl_div)
            
            # Combined anomaly score
            anomaly_scores = 0.5 * mse_norm + 0.3 * weighted_norm + 0.2 * kl_norm
        
        return anomaly_scores


class AttentionPooling(nn.Module):
    """Learnable attention pooling for sequence aggregation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Attention mechanism (works with any d_model size)
        hidden_dim = max(d_model // 2, 1)  # Ensure at least 1
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
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch, seq_len, 1]
        
        # Apply attention weights
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch, d_model]
        
        return pooled


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding applied directly to input features."""

    def __init__(self, num_features: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_features = num_features
        
        # Create positional encoding for the feature dimension
        pe = torch.zeros(max_len, num_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use different frequencies for different features
        div_term = torch.exp(torch.arange(0, num_features, 1).float() * (-math.log(10000.0) / num_features))
        
        # Apply sin/cos to all features (alternating pattern)
        for i in range(num_features):
            if i % 2 == 0:
                pe[:, i] = torch.sin(position.squeeze() * div_term[i])
            else:
                pe[:, i] = torch.cos(position.squeeze() * div_term[i])
        
        # Register as buffer for batch_first: [1, max_len, num_features]
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
