"""Variational Autoencoder (VAE) model for forest fire anomaly detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config



class DenseAutoencoder(nn.Module):
    """Dense autoencoder model for sequence data.
    
    Args:
        time_steps: Number of timesteps in sequence
        num_features: Features per timestep
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, time_steps=Config.WINDOW_SIZE, num_features=len(Config.INPUT_COLUMNS), latent_dim=Config.LATENT_DIM, hidden_dim=Config.HIDDEN_DIM):
        super().__init__()
        self.time_steps = time_steps
        self.num_features = num_features
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

    def forward(self, x):
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

    def train_step(self, x, optimizer, loss_fn, beta=0.001):
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
        
        return total_loss.item(), recon_loss.item(), 0.0, recon

    def get_embeddings(self, x):
        """Extract embeddings from input tensor.
        
        Args:
            x: Input tensor (batch, time_steps, num_features)
            
        Returns:
            torch.Tensor: Embeddings from encoder output
        """
        x_flat = x.view(x.size(0), -1)
        return self.encoder(x_flat)

    def val_step(self, x, loss_fn, beta=0.001):
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
            
        return total_loss.item(), recon_loss.item(), 0.0, recon



class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for sequence data with probabilistic latent space.
    
    Args:
        time_steps: Number of timesteps in sequence (default: from Config)
        num_features: Features per timestep (default: from Config)
        latent_dim: Latent space dimension (default: from Config)
        hidden_dim: Hidden layer dimension (default: from Config)
    """

    def __init__(
        self,
        time_steps: int = Config.WINDOW_SIZE,
        num_features: int = len(Config.INPUT_COLUMNS),
        latent_dim: int = Config.LATENT_DIM,
        hidden_dim: int = Config.HIDDEN_DIM
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.num_features = num_features
        self.input_dim = time_steps * num_features
        self.latent_dim = latent_dim

        # Encoder layers for mean and log variance
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def encode(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into mean and log variance of latent distribution."""
        h = self.encoder_fc(x_flat)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, num_features)
            
        Returns:
            reconstructed: Reconstructed tensor with same shape as input
            mu: Latent space mean
            logvar: Latent space log variance
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        # Validate input dimensions
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, timesteps, features), got {x.dim()}D")
            
        batch_size = x.size(0)
        if x.size(1) != self.time_steps or x.size(2) != self.num_features:
            raise ValueError(
                f"Input shape {x.shape[1:]} incompatible with "
                f"model configuration ({self.time_steps}, {self.num_features})"
            )

        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # Encode to latent distribution parameters
        mu, logvar = self.encode(x_flat)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstructed input
        reconstructed_flat = self.decode(z)
        
        # Reshape to original dimensions
        reconstructed = reconstructed_flat.view(batch_size, self.time_steps, self.num_features)
        return reconstructed, mu, logvar

    def train_step(self, x, optimizer, loss_fn, beta=0.001):
        """Perform a single training step for VAE.
        
        Args:
            x: Input tensor
            optimizer: Optimizer for training
            loss_fn: Loss function for reconstruction
            beta: Weight for KL divergence loss (default=0.001)
            
        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
        """
        self.train()
        optimizer.zero_grad()
        
        recon, mu, logvar = self.forward(x)
        recon_loss = loss_fn(recon, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.mean(kl_loss)
        
        total_loss = recon_loss + beta * kl_loss
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), recon_loss.item(), kl_loss.item(), recon

    def val_step(self, x, loss_fn, beta=0.001):
        """Perform a single validation step for VAE.
        
        Args:
            x: Input tensor
            loss_fn: Loss function for reconstruction
            beta: Weight for KL divergence loss (default=0.001)
            
        Returns:
            tuple: (total_loss, recon_loss, kl_loss, reconstruction)
        """
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            recon_loss = loss_fn(recon, x)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = torch.mean(kl_loss)
            
            total_loss = recon_loss + beta * kl_loss
            
        return total_loss.item(), recon_loss.item(), kl_loss.item(), recon

    def get_embeddings(self, x):
        """Extract embeddings from input tensor.
        
        Args:
            x: Input tensor (batch, time_steps, num_features)
            
        Returns:
            torch.Tensor: Embeddings (mean of latent distribution for VAE)
        """
        # Flatten input
        x_flat = x.view(x.size(0), -1)
        
        # Encode to latent distribution parameters
        mu, logvar = self.encode(x_flat)
        
        # Use mean as embedding
        return mu
