import unittest
from typing import Tuple, Any, Optional
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import geopandas as gpd
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from aad.common.config import Config
from aad.data.loader import DataLoader


class TestUtils(unittest.TestCase):

    def test_config_loading(self) -> None:
        """Test that config loads properly."""
        config = Config()
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.data_pipeline)
        self.assertIsNotNone(config.paths)

    def test_device_configuration(self) -> None:
        """Test device configuration is reasonable."""
        config = Config()
        device = config.training.DEVICE
        self.assertIn(device, ['cpu', 'cuda', 'mps'])

    def test_worker_count_configuration(self) -> None:
        """Test worker count is reasonable."""
        config = Config()
        workers = config.data_pipeline.NUM_WORKERS
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, 32)  # Reasonable upper bound

    @patch('aad.data.loader.pd.read_csv')
    @patch('os.path.exists')
    def test_data_loader_initialization(self, mock_exists: MagicMock, mock_read_csv: MagicMock) -> None:
        """Test that DataLoader initializes properly."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'id': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        config = Config()
        loader = DataLoader(config)
        self.assertIsNotNone(loader)
        self.assertEqual(loader.config, config)

    def test_paths_configuration(self) -> None:
        """Test that paths are configured correctly."""
        config = Config()
        
        # Check that paths are strings
        self.assertIsInstance(config.paths.DATA_DIR, str)
        self.assertIsInstance(config.paths.OUTPUT_DIR, str)
        self.assertIsInstance(config.paths.MODEL_DIR, str)
        
        # Check that raw data path is constructed correctly
        self.assertTrue(config.paths.RAW_DATA_PATH.endswith('.csv'))
        
    def test_training_parameters(self) -> None:
        """Test that training parameters are reasonable."""
        config = Config()
        
        self.assertGreater(config.training.EPOCHS, 0)
        self.assertGreater(config.training.BATCH_SIZE, 0)
        self.assertGreater(config.training.LEARNING_RATE, 0)
        self.assertLessEqual(config.training.LEARNING_RATE, 1)
        self.assertGreaterEqual(config.training.TRAIN_SPLIT, 0)
        self.assertLessEqual(config.training.TRAIN_SPLIT, 1)
        self.assertGreaterEqual(config.training.VAL_SPLIT, 0)
        self.assertLessEqual(config.training.VAL_SPLIT, 1)

    def test_data_pipeline_parameters(self) -> None:
        """Test that data pipeline parameters are reasonable."""
        config = Config()
        
        self.assertGreater(config.data_pipeline.WINDOW_DURATION_MINUTES, 0)
        self.assertGreater(config.data_pipeline.SAMPLING_INTERVAL_SECONDS, 0)
        self.assertIsInstance(config.data_pipeline.INPUT_COLUMNS, list)
        self.assertGreater(len(config.data_pipeline.INPUT_COLUMNS), 0)
        
        # Test computed properties
        window_size = config.data_pipeline.WINDOW_SIZE
        self.assertGreater(window_size, 0)
        
        input_dim = config.data_pipeline.INPUT_DIM
        self.assertEqual(input_dim, len(config.data_pipeline.INPUT_COLUMNS) * window_size)


if __name__ == '__main__':
    unittest.main()
