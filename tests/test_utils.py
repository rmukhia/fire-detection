import unittest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import model_utils
import data_loader
import config


class TestUtils(unittest.TestCase):

    def test_set_seed(self):
        """Test that set_seed properly initializes random number generators."""
        model_utils.set_seed(42)
        a = torch.rand(1)
        b = np.random.rand(1)
        model_utils.set_seed(42)
        c = torch.rand(1)
        d = np.random.rand(1)
        self.assertEqual(a.item(), c.item())
        self.assertEqual(b[0], d[0])

    @patch('data_loader.pd.read_parquet')
    @patch('data_loader.pd.read_csv')
    @patch('data_loader.gpd.GeoDataFrame')
    def test_load_data(self, mock_gdf, mock_read_csv, mock_read_parquet):
        """Test that load_data correctly loads and returns sensor data."""
        mock_df = pd.DataFrame({'id': [1, 2, 3]})
        mock_df_labels = pd.DataFrame(
            {'start_time': [pd.to_datetime('2023-01-01')], 'end_time': [pd.to_datetime('2023-01-02')], 'geometry': ['POINT (1 1)']}
        )
        mock_df_locations = pd.DataFrame(
            {'GPS_Lon': [], 'GPS_Lat': [], 'Sensor_Id': []}
        )

        mock_read_parquet.return_value = mock_df
        mock_df_sensor = pd.DataFrame({'Datetime': pd.to_datetime(['2023-01-01']), 'Sensor_Id': [1], 'feature': [10]})
        mock_read_csv.side_effect = [mock_df_sensor, mock_df_labels, mock_df_locations]
        mock_gdf.side_effect = [MagicMock(), MagicMock()]

        test_config = config.Config()
        data_loader_instance = data_loader.DataLoader(test_config)
        df_sensor, df_labels, df_locations = data_loader_instance.load_raw_data()

        self.assertIsNotNone(df_sensor)
        self.assertIsNotNone(df_labels)
        self.assertIsNotNone(df_locations)

    @patch('model_utils.torch.load')
    def test_create_dataloaders(self, mock_torch_load):
        """Test that dataloaders are created with proper train/val/test splits."""
        # Mock the torch.load calls for sequences and window_ids
        mock_sequences = torch.randn(100, 10, 3) # Added num_features dimension
        mock_fire_ids = torch.randint(0, 2, (100,))
        # Ensure some distances are above and some below the threshold
        mock_distances = torch.cat([
            torch.rand(70) * 100000 + config.Config.SPLIT_DISTANCE_THRESHOLD + 1000, # Eligible
            torch.rand(30) * config.Config.SPLIT_DISTANCE_THRESHOLD * 0.5 # Ineligible
        ])
        mock_window_ids = torch.arange(100)
        mock_torch_load.return_value = (mock_sequences, mock_fire_ids, mock_distances, mock_window_ids)

        train_loader, val_loader, test_loader = model_utils.create_dataloaders(config.Config(), remove_fire_labels=False)

        total_samples = len(mock_sequences)
        eligible_samples = (mock_distances > config.Config.SPLIT_DISTANCE_THRESHOLD).sum().item()
        ineligible_samples = (mock_distances <= config.Config.SPLIT_DISTANCE_THRESHOLD).sum().item()

        # Calculate expected sizes based on the logic in create_dataloaders
        expected_train_size = int(config.Config.TRAIN_SPLIT * total_samples)
        expected_val_size = int(config.Config.VAL_SPLIT * total_samples)

        if eligible_samples < (expected_train_size + expected_val_size):
            scale = eligible_samples / (expected_train_size + expected_val_size)
            expected_train_size = int(expected_train_size * scale)
            expected_val_size = eligible_samples - expected_train_size

        expected_test_size = total_samples - expected_train_size - expected_val_size

        self.assertEqual(len(train_loader.dataset), expected_train_size)
        self.assertEqual(len(val_loader.dataset), expected_val_size)
        self.assertEqual(len(test_loader.dataset), expected_test_size)
        self.assertEqual(len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset), total_samples)


if __name__ == '__main__':
    unittest.main()
