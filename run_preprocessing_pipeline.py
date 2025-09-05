#!/usr/bin/env python3
"""
Fire Detection Data Preprocessing Pipeline

This script runs the complete data preprocessing pipeline including:
1. Data loading and cleaning
2. Preprocessing and resampling
3. Ground truth collection and annotation
4. Sequence generation for model training
"""

import os
import sys
import tempfile
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import geopandas as gpd
import dask.config
from dask.distributed import Client

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from aad.common.config import Config
from aad.data.loader import DataLoader
from aad.data.preprocessing import DataPreprocessor
from aad.data.annotation import DataAnnotator
from aad.data.sequences import DataSequencer
from aad.data.groundtruth import GroundTruthCollector
from aad.common.core_logging import ProcessLogger


def setup_dask_config():
    """Setup optimized Dask configuration."""
    # Use system temp directory instead of hardcoded Windows path
    temp_dir = os.environ.get('DASK_TEMP_DIR', tempfile.gettempdir())
    dask.config.set({'temporary_directory': temp_dir})
    
    # Optimize Dask settings for better performance
    dask.config.set({
        'array.chunk-size': '256MB',
        'dataframe.query-planning': True,
        'optimization.fuse': {},
    })


def main():
    parser = argparse.ArgumentParser(description='Run fire detection preprocessing pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of Dask workers (overrides config)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Skip annotation step')
    parser.add_argument('--skip-sequencing', action='store_true',
                       help='Skip sequencing step')
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
    logger = ProcessLogger(config, 'preprocessing_pipeline')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine number of workers
    if args.n_workers:
        n_workers = args.n_workers
    else:
        n_workers = min(mp.cpu_count(), config.data_pipeline.NUM_WORKERS)
    
    logger.info(f"Starting preprocessing pipeline with {n_workers} workers")
    logger.info(f"Configuration: {args.config}")
    
    # Setup Dask configuration
    setup_dask_config()
    
    try:
        # Load data
        logger.info("Loading raw data...")
        loader = DataLoader(config)
        df_sensor, _, df_locations = loader.load_raw_data(label_load=False, location_load=True)
        logger.info(f"Loaded {len(df_sensor)} sensor records")
        
        # Use a single Dask client for the entire pipeline for better resource management
        with Client(n_workers=n_workers, threads_per_worker=2, 
                   memory_limit='4GB', dashboard_address=None) as client:
            logger.info(f"Dask client created: {client.dashboard_link}")
            
            # Step 1: Preprocessing
            if not args.skip_preprocessing:
                logger.info("Starting data preprocessing...")
                preprocessor = DataPreprocessor(config, df_sensor=df_sensor)
                preprocessor.preprocess_data(client=client)
                logger.info("Preprocessing completed")
            
            # Step 2: Ground Truth Processing and Annotation
            if not args.skip_annotation:
                logger.info("Starting ground truth collection...")
                groundtruth_collector = GroundTruthCollector(config)
                df_groundtruth = groundtruth_collector.collect_groundtruth(start_end_offset_min=180)
                logger.info(f"Collected {len(df_groundtruth)} ground truth records")
                
                logger.info("Starting data annotation...")
                annotator = DataAnnotator(config, df_labels=df_groundtruth, df_locations=df_locations)
                annotator.annotate_data(client=client)
                logger.info("Annotation completed")
        
        # Step 3: Sequence creation (outside Dask client to avoid conflicts)
        if not args.skip_sequencing:
            logger.info("Starting sequence creation...")
            sequencer = DataSequencer(config, ProcessLogger(config, 'sequencer'))
            sequencer.create_dataset(fit_scaler=True)
            logger.info("Sequence creation completed")
        
        logger.info("Preprocessing pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    