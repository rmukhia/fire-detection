"""Transforms annotated sensor data into training sequences for fire detection."""

# =============================================================================
# IMPORT STATEMENTS - ORGANIZED BY CATEGORY
# =============================================================================

# Standard Library Imports
import os
from typing import Any, Dict, List, Optional, Union

# Third-Party Scientific Computing
import numpy as np
import pandas as pd
import torch
import joblib
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from torch.utils.data import TensorDataset
from dask.distributed import Client
import multiprocessing as mp
from config import Config
import core_logging
import viz_style
import matplotlib.pyplot as plt
import seaborn as sns

class DataSequencer:
    """Creates and manages data sequences for model training."""
    def __init__(self, config: Config, logger: Optional[core_logging.ProcessLogger] = None):
        """
        Initializes the DataSequencer with a configuration and an optional logger.

        Args:
            config (Config): The configuration object containing paths and parameters.
            logger (Optional[core_logging.ProcessLogger]): The logger for process tracking.
        """
        self.config = config
        self.logger = logger or core_logging.ProcessLogger(config, "Sequence_Creation")
        self.scaler = None  # Will hold the fitted scaler
        self.logger.log_step("DataSequencer initialized")
        
        # Visualization Setup
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

    def _fit_or_load_scaler(self, df: dd.DataFrame, fit_new: bool = False) -> None:
        """
        Fit scaler on data or load existing scaler.
        
        Args:
            df: DataFrame to fit scaler on (if fitting new)
            fit_new: Whether to fit a new scaler or load existing one
        """
        if fit_new or not os.path.exists(self.config.SCALER_PATH):
            self.logger.log_step("Fitting new scaler on data")
            self.scaler = DaskStandardScaler()
            self.scaler.fit(df[self.config.INPUT_COLUMNS])
            
            # Save the fitted scaler
            os.makedirs(os.path.dirname(self.config.SCALER_PATH), exist_ok=True)
            joblib.dump(self.scaler, self.config.SCALER_PATH)
            self.logger.log_step(f"Scaler fitted and saved to {self.config.SCALER_PATH}")
        else:
            self.logger.log_step(f"Loading existing scaler from {self.config.SCALER_PATH}")
            self.scaler = joblib.load(self.config.SCALER_PATH)

    def _log_sequence_creation_statistics(self, dataset: TensorDataset) -> None:
        """
        Generate visualizations of sequence statistics.
        
        Args:
            dataset: TensorDataset containing sequences and metadata
        """
        self.logger.log_step("Generating sequence creation statistics")

        # Apply publication-ready style to plots
        viz_style.set_publication_style()
        
        # Extract tensors from the dataset for visualization
        sequences, _, distances, _ = dataset.tensors

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Sequence Creation Statistics', fontsize=16, fontweight='bold')

        # --- 1. Feature distribution (scaled) ---
        # Plot distribution of a subset of scaled features from the first timestep
        num_features_to_plot = min(len(self.config.INPUT_COLUMNS), 10)
        first_timestep_features = sequences[:, 0, :].numpy()

        for feature_idx in range(num_features_to_plot):
            feature_name = self.config.INPUT_COLUMNS[feature_idx]
            sns.histplot(first_timestep_features[:, feature_idx], ax=axes[0], kde=True,
                         label=feature_name, alpha=0.6, linewidth=0.8)
        axes[0].set_title('Feature Distribution (Scaled, First Timestep)', fontweight='bold')
        axes[0].set_xlabel('Value', fontweight='bold')
        axes[0].set_ylabel('Density', fontweight='bold')
        axes[0].legend(title='Features', loc='upper right')

        # --- 2. Distance distribution ---
        # Plot histogram of distances to fire to understand proximity distribution
        sns.histplot(distances.numpy(), bins=30, ax=axes[1], color=viz_style.VALID_COLORS['blue'], alpha=0.7, edgecolor='black', linewidth=0.8)
        axes[1].set_title('Distance to Fire Distribution', fontweight='bold')
        axes[1].set_xlabel('Distance (m)', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')

        plt.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs(self.config.STATS_IMAGES_DIR, exist_ok=True)
        plot_path = os.path.join(self.config.STATS_IMAGES_DIR, 'sequence_creation_statistics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show() 
        plt.close() # Close the plot to free up memory resources

        self.logger.log_step(f"Statistics plot saved to: {plot_path}")

    def create_sequences_and_labels(self, df_annotated: dd.DataFrame, fit_scaler: bool = False) -> TensorDataset:
        """
        Transform annotated data into training sequences.
        
        Args:
            df_annotated: DataFrame with window_id, fire_id, distance_to_fire_m,
                         Datetime and feature columns
            fit_scaler: Whether to fit scaler (True for training data, False for val/test)
                         
        Returns:
            TensorDataset: Contains sequences, fire_ids, distances, window_ids
        """
        self.logger.log_step("Starting sequence creation and feature scaling")

        # Validate presence of essential metadata columns
        required_meta_cols = ['window_id', 'fire_id', 'distance_to_fire_m']
        for col in required_meta_cols:
            if col not in df_annotated.columns:
                raise KeyError(f"Required column '{col}' not found in dataframe: {df_annotated.columns.tolist()}")

        # --- Fit or load scaler ---
        self._fit_or_load_scaler(df_annotated, fit_new=fit_scaler)

        # --- Scale only feature columns ---
        self.logger.log_step("Scaling input features using fitted scaler")
        
        # Fit and transform on the Dask DataFrame
        df_scaled_dask = df_annotated.copy()
        df_scaled_dask[self.config.INPUT_COLUMNS] = self.scaler.transform(
            df_scaled_dask[self.config.INPUT_COLUMNS]
        )

        # --- Create sequences in (num_windows, time_steps, num_features) format ---
        self.logger.log_step("Grouping data by window_id to create sequences using Dask")

        def _extract_sequence_from_group(group: pd.DataFrame) -> np.ndarray:
            # Sort by Datetime to ensure correct temporal order within each window
            group = group.sort_values('Datetime')
            return group[self.config.INPUT_COLUMNS].values

        # Apply the function to each group. The meta argument is crucial for Dask to infer the output type.
        # The output of apply will be a Dask Series where each element is a NumPy array (the sequence).
        sequences_dask_series = df_scaled_dask.groupby('window_id').apply(
            _extract_sequence_from_group,
            meta=('x', 'object') # 'object' because the output is a numpy array, not a simple type
        )
        
        # Compute the sequences and stack them into a single 3D numpy array
        sequences_list = sequences_dask_series.compute().tolist()
        sequences_array = np.stack(sequences_list, axis=0)
        
        # Convert numpy array to PyTorch tensor for model compatibility
        sequences = torch.tensor(sequences_array, dtype=torch.float32)
        self.logger.log_step(f"Created {sequences.shape[0]} sequences, each with shape {sequences.shape[1:]}")

        # --- Extract metadata (unscaled) ---
        self.logger.log_step("Extracting unscaled metadata (fire_id, distance_to_fire_m, window_id) using Dask")
        # Retrieve the first occurrence of metadata for each unique window_id
        # .compute() is called here to get a Pandas DataFrame from the Dask DataFrame
        meta_df = df_annotated.groupby('window_id').first()[['fire_id', 'distance_to_fire_m']].compute()
        
        # Convert metadata to PyTorch tensors
        fire_ids = torch.tensor(meta_df['fire_id'].values, dtype=torch.long)
        distances = torch.tensor(meta_df['distance_to_fire_m'].values, dtype=torch.float32)
        window_ids = torch.tensor(meta_df.index.values, dtype=torch.long)

        # --- Create unified TensorDataset ---
        # Combine sequences and metadata into a single PyTorch TensorDataset
        dataset = TensorDataset(sequences, fire_ids, distances, window_ids)
        # Save the dataset tensors to the specified path for later use in training/evaluation
        torch.save(dataset.tensors, self.config.DATASET_PATH)
        self.logger.log_step(f"Full TensorDataset saved to {self.config.DATASET_PATH}")

        return dataset

    def create_dataset(self, fit_scaler: bool = False) -> TensorDataset:
        """
        Execute complete dataset creation workflow.
        
        Args:
            fit_scaler: Whether to fit scaler (True for training, False for val/test)
        
        Returns:
            TensorDataset: Final dataset ready for training
        """
        self.logger.log_step("Starting comprehensive dataset creation workflow")

        n_workers = min(mp.cpu_count(), self.config.NUM_WORKERS)
        with Client(n_workers=n_workers, threads_per_worker=1) as client:
            self.logger.log_step(f"Started Dask cluster with {n_workers} workers: {client.dashboard_link}")

            # Load the previously annotated data (multiple sensor files)
            # Use dask.dataframe.read_parquet to read all parquet files in the directory
            df_window = dd.read_parquet(os.path.join(self.config.ANNOTATED_DATA_DIR, "sensor_*.parquet"))
            self.logger.log_step(f"Loaded processed data from {self.config.ANNOTATED_DATA_DIR} as Dask DataFrame")
            
            # Create sequences and labels from the loaded data
            dataset = self.create_sequences_and_labels(df_window, fit_scaler=fit_scaler)
            # Log and visualize statistics of the created sequences
            self._log_sequence_creation_statistics(dataset)

        self.logger.log_step("Dataset creation completed successfully")
        self.logger.save_process_timeline() # Save the timeline for this process for debugging/monitoring
        return dataset


# =========================================================================
# Main Function
# =========================================================================
def create_dataset(config: Config, fit_scaler: bool = False) -> TensorDataset:
    """
    Main function to create dataset from config.
    
    Args:
        config: Configuration object
        fit_scaler: Whether to fit scaler (True for training, False for val/test)
        
    Returns:
        TensorDataset: Created dataset
    """
    sequencer = DataSequencer(config)
    return sequencer.create_dataset(fit_scaler=fit_scaler)