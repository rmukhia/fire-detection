#!/usr/bin/env python3
"""
Fire Detection Model Training Script

This script provides a command-line interface for training the fire detection model.
It can be used as an alternative to the Jupyter notebook for production environments.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from aad.common.config import Config
from aad.common.core_logging import ProcessLogger
from aad.model_autoencoder import AutoencoderTrainer


def main():
    parser = argparse.ArgumentParser(description='Train fire detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only (no training)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override configuration with command-line arguments
    if args.epochs is not None:
        config.training.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.training.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.training.LEARNING_RATE = args.learning_rate
    
    # Device selection
    if args.device:
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        config.training.DEVICE = device
    
    # Setup logging
    logger = ProcessLogger(config, 'training')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting model training with configuration: {args.config}")
    logger.info(f"Device: {config.training.DEVICE}")
    logger.info(f"Epochs: {config.training.EPOCHS}")
    logger.info(f"Batch size: {config.training.BATCH_SIZE}")
    logger.info(f"Learning rate: {config.training.LEARNING_RATE}")
    
    try:
        # Initialize trainer
        trainer = AutoencoderTrainer(config, logger)
        
        if args.validate_only:
            logger.info("Running validation only...")
            if args.resume:
                trainer.load_checkpoint(args.resume)
            else:
                logger.error("Validation-only mode requires --resume checkpoint")
                sys.exit(1)
            
            val_loss = trainer.validate()
            logger.info(f"Validation loss: {val_loss:.6f}")
        else:
            logger.info("Starting training...")
            if args.resume:
                logger.info(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            
            trainer.train()
            logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()