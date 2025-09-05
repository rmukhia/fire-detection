#!/usr/bin/env python3
"""
Fire Detection System Setup and Optimization Script

This script performs initial setup, optimization, and system checks.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from aad.common.config import Config
from aad.common.core_logging import ProcessLogger
from aad.common.performance import (
    log_system_info,
    configure_environment_variables,
    optimize_pytorch_settings,
    create_performance_config,
    check_system_requirements
)
from aad.common.data_validation import validate_data_pipeline


def setup_directories(config: Config, logger: logging.Logger):
    """Create necessary directories."""
    directories = [
        config.paths.OUTPUT_DIR,
        config.paths.PROCESSED_DATA_DIR,
        config.paths.ANNOTATED_DATA_DIR,
        config.paths.INTERMEDIATE_DIR,
        config.paths.STATS_DIR,
        config.paths.STATS_IMAGES_DIR,
        config.paths.STATS_CSV_DIR,
        config.paths.STATS_HTML_DIR,
        config.paths.LOGS_DIR,
        config.paths.MODEL_DIR,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")
    
    logger.info(f"Created {len(directories)} directories")


def check_dependencies(logger: logging.Logger):
    """Check that all required dependencies are available."""
    required_packages = [
        'pandas', 'torch', 'sklearn', 'joblib', 'geopandas', 
        'shapely', 'tqdm', 'plotly', 'numpy', 'matplotlib', 
        'seaborn', 'dask', 'distributed'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    else:
        logger.info("✓ All required packages are available")
        return True


def optimize_configuration(config: Config, logger: logging.Logger):
    """Apply performance optimizations to configuration."""
    logger.info("Applying performance optimizations...")
    
    # Configure environment variables
    configure_environment_variables()
    
    # Optimize PyTorch settings
    optimize_pytorch_settings()
    
    # Get performance recommendations
    perf_config = create_performance_config()
    
    logger.info("Performance recommendations:")
    for key, value in perf_config.items():
        logger.info(f"  {key}: {value}")
    
    # Update config with recommendations if not explicitly set
    if config.training.DEVICE == 'auto':
        logger.info(f"Setting device to: {perf_config['device']}")
    
    return perf_config


def run_system_checks(logger: logging.Logger):
    """Run comprehensive system checks."""
    logger.info("Running system checks...")
    
    # Check system requirements
    requirements = check_system_requirements()
    
    failed_requirements = [req for req, met in requirements.items() if not met]
    if failed_requirements:
        logger.warning(f"System requirements not met: {failed_requirements}")
    else:
        logger.info("✓ All system requirements met")
    
    # Log detailed system information
    log_system_info(logger)
    
    return len(failed_requirements) == 0


def validate_data(config: Config, logger: logging.Logger):
    """Validate input data if available."""
    logger.info("Validating input data...")
    
    try:
        validator = validate_data_pipeline(config, logger)
        
        # Check if validation found any critical issues
        total_issues = sum(
            len(results.get('issues', []))
            for results in validator.validation_results.values()
        )
        
        if total_issues == 0:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"Data validation found {total_issues} issues")
        
        return total_issues == 0
    
    except Exception as e:
        logger.warning(f"Data validation skipped: {e}")
        return True  # Don't fail setup if data isn't available yet


def create_sample_config(config_path: Path, logger: logging.Logger):
    """Create a sample configuration file with optimized settings."""
    if config_path.exists():
        logger.info(f"Configuration file already exists: {config_path}")
        return
    
    perf_config = create_performance_config()
    
    sample_config = f"""# Fire Detection System Configuration
# Generated with optimized settings for this system

paths:
  DATA_DIR: data
  OUTPUT_DIR: output
  MODEL_DIR: models
  LOGS_DIR: logs

data_pipeline:
  NUM_SAMPLES: 100000
  RESAMPLE_INTERVAL: 2min
  WINDOW_DURATION_MINUTES: 120
  SAMPLING_INTERVAL_SECONDS: 120
  EPSG_ZONE: EPSG:32647
  INPUT_COLUMNS:
    - PM2.5
    - Carbon dioxide (CO2)
    - Relative humidity
  LOCAL_OFFSET_MINUTES: 420
  NUM_WORKERS: {perf_config['num_workers']}
  RESAMPLE_TOLERANCE_FACTOR: 0.5
  WINDOW_STEP_SIZE: 1
  DURATION_FILTER_TOLERANCE: 5s
  SMA_MULTIPLIERS: [4, 16, 32, 64]

training:
  EPOCHS: 50
  BATCH_SIZE: {perf_config['batch_size']}
  LEARNING_RATE: 0.001
  RANDOM_SEED: 42
  TRAIN_SPLIT: 0.7
  VAL_SPLIT: 0.15
  PATIENCE: 25
  LOSS_FUNCTION: MSELoss
  OPTIMIZER: Adam
  USE_BETA_SCHEDULE: true
  BETA_SCHEDULE_TYPE: linear
  USE_DISTRIBUTED: false
  DISTRIBUTED_BACKEND: nccl
  USE_DATA_PARALLEL: false
  USE_MIXED_PRECISION: {str(perf_config['use_mixed_precision']).lower()}
  DISTANCE_FILTER_THRESHOLD_M: 5000
  ANOMALY_THRESHOLD_PERCENTILE: 99.95
  LOG_FILENAME_PATTERN: "%Y-%m-%d_%H-%M-%S.log"
  DEVICE: {perf_config['device']}

tuning:
  AUTOENCODER_CLASS: VariationalAutoencoder
  LATENT_DIM: 12
  HIDDEN_DIM: 360
  CONV_NUM_LAYERS: 4
  CONV_KERNEL_SIZE: 5
  CONV_DROPOUT_RATE: 0.1
  CONV_USE_ATTENTION: true
  CONV_USE_RESIDUAL: true
  KMEANS_N_CLUSTERS: 32
  DBSCAN_EPS: 0.3
  DBSCAN_MIN_SAMPLES: 5
  DIM_REDUCTION_METHOD: tsne
  HYPERPARAMETER_GRID:
    latent_dims: [18, 36]
    hidden_dims: [180, 360]
    anomaly_threshold_percentiles: [99.9, 99.98]
  ADVANCED_HYPERPARAMETER_GRID:
    latent_dims: [18, 36, 54]
    hidden_dims: [180, 360, 540]
    learning_rates: [0.001, 0.0005, 0.0001]
    batch_sizes: [16, 32, 64]
    anomaly_thresholds: [99.9, 99.95, 99.98]
"""
    
    with open(config_path, 'w') as f:
        f.write(sample_config)
    
    logger.info(f"Created optimized configuration: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Setup and optimize fire detection system')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create optimized configuration file')
    parser.add_argument('--skip-data-validation', action='store_true',
                       help='Skip data validation step')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('setup')
    
    logger.info("=== Fire Detection System Setup ===")
    
    config_path = Path(args.config)
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config(config_path, logger)
        return
    
    # Check dependencies first
    if not check_dependencies(logger):
        sys.exit(1)
    
    # Load configuration
    try:
        config = Config()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        if not config_path.exists():
            logger.info("Consider running with --create-config to generate a sample configuration")
        sys.exit(1)
    
    logger.info(f"Loaded configuration: {config_path}")
    
    # Setup directories
    setup_directories(config, logger)
    
    # Run system checks
    system_ok = run_system_checks(logger)
    if not system_ok:
        logger.warning("System checks failed, but continuing...")
    
    # Optimize configuration
    perf_config = optimize_configuration(config, logger)
    
    # Validate data if available
    if not args.skip_data_validation:
        validate_data(config, logger)
    
    logger.info("=== Setup Complete ===")
    logger.info("System is ready for fire detection processing!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Place your data files in the data/ directory")
    logger.info("2. Run preprocessing: python run_preprocessing_pipeline.py")
    logger.info("3. Train models: python run_training.py")
    logger.info("4. Or use Jupyter: jupyter lab ForestFireDetection.ipynb")
    

if __name__ == "__main__":
    main()