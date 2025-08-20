"""Transforms annotated sensor data into training sequences for fire detection."""

# =============================================================================
# IMPORT STATEMENTS - ORGANIZED BY CATEGORY
# =============================================================================

# Standard Library Imports
import gc
import os
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Scientific Computing
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from dask.distributed import Client
import multiprocessing as mp
from aad.common.config import Config
from aad.common import core_logging
from aad.common import viz_style
import matplotlib.pyplot as plt
import seaborn as sns


from dask.delayed import delayed
from aad.data.dask_pipeline_base import DaskPipelineBase
from pathlib import Path

class DataSequencer(DaskPipelineBase):
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
        plt.style.use("seaborn-v0_8-paper")
        sns.set_palette("husl")

    def _fit_scaler_on_normal_data_only(self, df: pd.DataFrame, columns : List[str]) -> None:
        """
        CRITICAL FOR ANOMALY DETECTION: Fit scaler only on non-fire data.
        This establishes the 'normal' baseline for anomaly detection.
        """
        self.logger.log_step("Fitting scaler ONLY on normal (non-fire) data for anomaly detection")
        
        # Filter to non-fire data only
        normal_data = df[df['fire_id'] == -1]
        normal_count = len(normal_data)
        total_count = len(df)
        
        if normal_count == 0:
            raise ValueError("No non-fire data found for scaler fitting")
        
        self.logger.log_step(
            f"Fitting scaler on {normal_count} normal samples ({normal_count/total_count*100:.1f}% of total)"
        )
        
        # Fit scaler only on normal data
        self.scaler = StandardScaler()
        self.scaler.fit(normal_data[columns])
        
        # Save the normal-data-fitted scaler

    def _log_sequence_creation_statistics(self, dataset: TensorDataset, columns : List[str]) -> None:
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
        fig.suptitle("Sequence Creation Statistics", fontsize=16, fontweight="bold")

        # --- 1. Feature distribution (scaled) ---
        # Plot distribution of a subset of scaled features from the first timestep
        num_features_to_plot = min(len(columns), 10)
        first_timestep_features = sequences[:, 0, :].numpy()

        for feature_idx in range(num_features_to_plot):
            feature_name = columns[feature_idx]
            sns.histplot(
                first_timestep_features[:, feature_idx],
                ax=axes[0],
                kde=True,
                label=feature_name,
                alpha=0.6,
                linewidth=0.8,
            )
        axes[0].set_title("Feature Distribution (Scaled, First Timestep)", fontweight="bold")
        axes[0].set_xlabel("Value", fontweight="bold")
        axes[0].set_ylabel("Density", fontweight="bold")
        axes[0].legend(title="Features", loc="upper right")

        # --- 2. Distance distribution ---
        # Plot histogram of distances to fire to understand proximity distribution
        sns.histplot(
            distances.numpy(),
            bins=30,
            ax=axes[1],
            color=viz_style.VALID_COLORS["blue"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.8,
        )
        axes[1].set_title("Distance to Fire Distribution", fontweight="bold")
        axes[1].set_xlabel("Distance (m)", fontweight="bold")
        axes[1].set_ylabel("Frequency", fontweight="bold")

        plt.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs(self.config.paths.STATS_IMAGES_DIR, exist_ok=True)
        plot_path = os.path.join(self.config.paths.STATS_IMAGES_DIR, "sequence_creation_statistics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()  # Close the plot to free up memory resources

        self.logger.log_step(f"Statistics plot saved to: {plot_path}")

    def create_sequences_and_labels(self, df_annotated: pd.DataFrame, columns:  List[str], fit_scaler: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        required_meta_cols = ["window_id", "fire_id", "distance_to_fire_m"]
        for col in required_meta_cols:
            if col not in df_annotated.columns:
                raise KeyError(f"Required column '{col}' not found in dataframe: {df_annotated.columns.tolist()}")

        # --- Scale only feature columns ---
        self.logger.log_step("Scaling input features using fitted scaler")

        # Validate that scaler was successfully initialized
        if self.scaler is None:
            raise ValueError("Scaler was not properly initialized. Check _fit_or_load_scaler method.")

        meta_df = df_annotated.groupby("window_id").first()[["fire_id", "distance_to_fire_m"]]
        # Fit and transform on the DataFrame

        df_annotated[columns] = self.scaler.transform(
            df_annotated[columns]
        )

        # --- Create sequences in (num_windows, time_steps, num_features) format ---
        self.logger.log_step("Grouping data by window_id to create sequences [LOG_BEFORE_GROUPBY_APPLY]")

        def _extract_sequence_from_group(group: pd.DataFrame) -> np.ndarray:
            # Sort by Datetime to ensure correct temporal order within each window
            group = group.sort_values("Datetime")
            return group[columns].values

        sequences_list = [
            _extract_sequence_from_group(group)
            for _, group in df_annotated.groupby("window_id")
        ]

        self.logger.log_step("Completed groupby for sequence creation [LOG_AFTER_GROUPBY]")


        # Stack the sequences into a single 3D numpy array
        sequences_array = np.stack(sequences_list, axis=0)
        sequences = torch.tensor(sequences_array, dtype=torch.float32)
        self.logger.log_step(f"Created {sequences.shape[0]} sequences, each with shape {sequences.shape[1:]}")

        # --- Extract metadata (unscaled) ---
        self.logger.log_step("Extracting unscaled metadata (fire_id, distance_to_fire_m, window_id)")
        fire_ids = torch.tensor(meta_df["fire_id"].values, dtype=torch.long)
        distances = torch.tensor(meta_df["distance_to_fire_m"].values, dtype=torch.float32)
        window_ids = torch.tensor(meta_df.index.values, dtype=torch.long)

        return sequences, fire_ids, distances, window_ids

    def create_dataset(self, fit_scaler: bool = True) -> TensorDataset:
        """
        Execute complete dataset creation workflow on all data at once (no Dask multi-tasking).

        Args:
            fit_scaler: Whether to fit scaler (True for training, False for val/test)
            client: (unused, kept for compatibility)

        Returns:
            TensorDataset: Final dataset ready for training
        """
        self.logger.log_step("Starting dataset creation workflow (low memory, sequential processing)")

        # Find all annotated sensor files
        sensor_files = list(Path(self.config.paths.ANNOTATED_DATA_DIR).glob("sensor_*.parquet"))
        if not sensor_files:
            raise ValueError("No annotated sensor data found.")

        columns = []
        for col in self.config.data_pipeline.INPUT_COLUMNS:
            columns.append(col)
            for m in self.config.data_pipeline.SMA_MULTIPLIERS:
                columns.append(f"{col}_sma_{m}x")
                columns.append(f"{col}_max_{m}x")
                columns.append(f"{col}_min_{m}x")

        self.logger.log_step(f"Columns for tensors: {columns}")
        print(f"Columns for tensors: {columns}, {len(columns)} total")

        self.scaler = StandardScaler()
        # Fit scaler if needed, using all normal data from all sensor files
        if fit_scaler:
            for sensor_file in sensor_files:
                df = pd.read_parquet(sensor_file)
                if 'fire_id' in df.columns:
                    normal_df = df[df['fire_id'] == -1]
                    self.scaler.partial_fit(normal_df[columns])
            gc.collect()
            os.makedirs(os.path.dirname(self.config.paths.SCALER_PATH), exist_ok=True)
            joblib.dump(self.scaler, self.config.paths.SCALER_PATH)
            self.logger.log_step(f"Normal-data-only scaler fitted and saved to {self.config.paths.SCALER_PATH}")


        # Sequentially process each sensor file and collect tensors
        all_sequences = []
        all_fire_ids = []
        all_distances = []
        all_window_ids = []

        for sensor_file in sensor_files:
            df_annotated = pd.read_parquet(sensor_file)
            sequences, fire_ids, distances, window_ids = self.create_sequences_and_labels(df_annotated, columns, fit_scaler=fit_scaler)
            all_sequences.append(sequences)
            all_fire_ids.append(fire_ids)
            all_distances.append(distances)
            all_window_ids.append(window_ids)

        # Concatenate tensors along the first dimension
        sequences = torch.cat(all_sequences, dim=0)
        fire_ids = torch.cat(all_fire_ids, dim=0)
        distances = torch.cat(all_distances, dim=0)
        window_ids = torch.cat(all_window_ids, dim=0)

        dataset = TensorDataset(sequences, fire_ids, distances, window_ids)
        torch.save(dataset.tensors, self.config.paths.DATASET_PATH)

        # Log and visualize statistics of the created sequences
        self._log_sequence_creation_statistics(dataset, columns)
        self.logger.log_step("Dataset creation completed successfully")
        self.logger.save_process_timeline()
        return dataset
