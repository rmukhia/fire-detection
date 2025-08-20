"""Model loading utilities for forest fire detection."""

import torch
import torch.nn as nn
from typing import Optional
from .common.config import Config
from .autoencoder import model_base


def load_autoencoder_model(config: Config, model_path: Optional[str] = None) -> nn.Module:
    """
    Load a trained autoencoder model from disk.

    Args:
        config: Configuration object
        model_path: Path to model file (optional, uses config.paths.BEST_MODEL_PATH if not provided)

    Returns:
        Loaded autoencoder model instance
    """
    if model_path is None:
        model_path = config.paths.BEST_MODEL_PATH

    # Get the autoencoder class from config
    autoencoder_class = getattr(model_base, config.tuning.AUTOENCODER_CLASS)

    # Create model instance with proper parameters
    if config.tuning.AUTOENCODER_CLASS == "ConvolutionalVariationalAutoencoder":
        model = autoencoder_class(
            time_steps=config.data_pipeline.WINDOW_SIZE,
            num_features=len(config.data_pipeline.INPUT_COLUMNS),
            latent_dim=config.tuning.LATENT_DIM,
            hidden_dim=config.tuning.HIDDEN_DIM,
            num_layers=config.tuning.CONV_NUM_LAYERS,
            kernel_size=config.tuning.CONV_KERNEL_SIZE,
            dropout=config.tuning.CONV_DROPOUT_RATE,
            use_attention=config.tuning.CONV_USE_ATTENTION,
            use_residual=config.tuning.CONV_USE_RESIDUAL,
        )
    else:
        model = autoencoder_class(
            time_steps=config.data_pipeline.WINDOW_SIZE,
            num_features=len(config.data_pipeline.INPUT_COLUMNS),
            latent_dim=config.tuning.LATENT_DIM,
            hidden_dim=config.tuning.HIDDEN_DIM,
        )

    # Load state dict
    device = torch.device(config.training.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model
