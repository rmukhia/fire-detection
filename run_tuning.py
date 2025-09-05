#!/usr/bin/env python3
"""
Fire Detection Model Hyperparameter Tuning Script

This script provides a command-line interface for hyperparameter optimization.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from aad.common.config import Config
from aad.common.core_logging import ProcessLogger
from aad.model_tuning import HyperparameterTuner


def main():
    parser = argparse.ArgumentParser(description='Optimize fire detection model hyperparameters')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['basic', 'advanced'], default='basic',
                       help='Tuning mode: basic or advanced grid search')
    parser.add_argument('--n-trials', type=int, default=None,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for optimization')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from previous study')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    logger = ProcessLogger(config, 'tuning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting hyperparameter tuning with configuration: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    
    try:
        # Initialize tuner
        tuner = HyperparameterTuner(config, logger)
        
        # Run optimization
        best_params = tuner.optimize(
            mode=args.mode,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            resume_from=args.resume
        )
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Best parameters: {best_params}")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()