"""Abstract base class for all trainers in the forest fire detection system."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, List, Optional
from aad.common import core_logging
from aad.common.config import Config


class BaseTrainer(ABC):
    """Abstract base class for model trainers providing common infrastructure."""

    def __init__(self, logger: core_logging.ProcessLogger, device: str, callbacks: Optional[List[Any]] = None) -> None:
        """
        Initialize the base trainer with logger, device, and optional callbacks.
        Args:
            logger: Logger instance
            device: torch.device
            callbacks: List of callback objects implementing hook methods
        """
        self.logger: core_logging.ProcessLogger = logger
        self.device: torch.device = torch.device(device)
        self.callbacks: List[Any] = callbacks if callbacks is not None else []

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train the model using the provided data loaders.

        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data

        Returns:
            tuple: (trained_model, training_metrics)
        """
        pass

    @abstractmethod
    def validate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Validate the model on the provided data loader.

        Args:
            model: The neural network model to validate
            dataloader: DataLoader for validation data

        Returns:
            dict: Validation metrics
        """
        pass

    def _log_model_architecture(self, model: nn.Module) -> None:
        """
        Log model architecture details.

        Args:
            model: The model to log details for
        """
        self.logger.log_info("Model Architecture:")
        self.logger.log_info(f"Model type: {type(model).__name__}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.log_info(f"Total parameters: {total_params:,}")
        self.logger.log_info(f"Trainable parameters: {trainable_params:,}")

        # Log layer details
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                if hasattr(module, "weight"):
                    self.logger.log_info(
                        f"Layer {name}: {module}, "
                        f"Parameters: {module.weight.numel():,}"  # type: ignore
                    )
                else:
                    self.logger.log_info(f"Layer {name}: {module}")

    # ==========================
    # Callback Hook Definitions
    # ==========================
    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for cb in self.callbacks:
            if hasattr(cb, "on_epoch_start"):
                cb.on_epoch_start(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for cb in self.callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(epoch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        for cb in self.callbacks:
            if hasattr(cb, "on_batch_end"):
                cb.on_batch_end(batch, logs)

    def on_nan_detected(self, epoch: int, batch: int, logs: Optional[Dict[str, Any]] = None):
        for cb in self.callbacks:
            if hasattr(cb, "on_nan_detected"):
                cb.on_nan_detected(epoch, batch, logs)
