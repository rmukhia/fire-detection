"""Dataset helper utilities for forest fire detection."""

from venv import logger
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import Tuple, Optional, Union, List
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from aad.common.core_logging import ProcessLogger
from aad.common.config import Config


def parse_geometry_column(df: pd.DataFrame, geometry_col: str = "geometry", crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Parse WKT geometry column and return a GeoDataFrame.
    """
    df = df.copy()
    df[geometry_col] = [wkt.loads(geom_str) for geom_str in df[geometry_col]]
    return gpd.GeoDataFrame(df, geometry=geometry_col, crs=crs)


def localize_datetime_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Remove timezone info from specified datetime columns.
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].dt.tz_localize(None)
    return df


def create_window_views(data: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
    """
    Create windowed views of the data using stride tricks.
    """
    if data.shape[0] < window_size:
        return np.array([]).reshape(0, window_size, data.shape[1])
    n_windows = (data.shape[0] - window_size) // step_size + 1
    shape = (n_windows, window_size, data.shape[1])
    strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)


def _load_dataset(
    dataset_path: str, num_samples: Optional[int], random_seed: int, logger: ProcessLogger
) -> Union[TensorDataset, Subset]:
    """Load and optionally sample dataset."""
    sequences, fire_ids, distances, window_ids = torch.load(dataset_path)
    dataset = TensorDataset(sequences, fire_ids, distances, window_ids)
    logger.log_step("Dataset loaded successfully")

    # Apply random sampling if specified
    if num_samples is not None and num_samples < len(dataset):
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
        dataset = Subset(dataset, indices.tolist())
        logger.log_step("Applied random sampling", {"num_samples": num_samples})

    return dataset

import torch
from torch.utils.data import TensorDataset, Subset
from typing import Union

def _filter_fire_labels(
    dataset: Union[TensorDataset, Subset], 
    logger, 
    fire_threshold_distance_min: int
) -> TensorDataset:
    """Filter dataset based on fire label and distance threshold.
    Rules:
      1. Keep all non-fire samples.
      2. Relabel fire samples with distance > threshold as non-fire (fire_id=-1, distance=inf).
      3. Discard remaining fire samples (distance <= threshold).
    Always returns a new TensorDataset.
    """
    logger.log_step(f"Removing fire-labeled data, {len(dataset)} samples before filtering")

    # --- Extract base tensors and indices ---
    if isinstance(dataset, TensorDataset):
        sequences, fire_ids, distances, window_ids = dataset.tensors
        indices = torch.arange(len(dataset))
    else:  # Subset
        base_dataset = dataset.dataset
        while isinstance(base_dataset, Subset):  # unwrap nested subsets
            base_dataset = base_dataset.dataset
        if not isinstance(base_dataset, TensorDataset):
            raise TypeError(f"Expected TensorDataset, got {type(base_dataset)}")

        sequences, fire_ids, distances, window_ids = base_dataset.tensors
        indices = torch.tensor(dataset.indices)

    # --- Select only the subset we’re working with ---
    selected_fire_ids = fire_ids[indices].clone()
    selected_distances = distances[indices].clone()
    selected_sequences = sequences[indices]
    selected_window_ids = window_ids[indices]

    # --- Count total fire samples before filtering ---
    total_fire = (selected_fire_ids != -1).sum().item()

    # --- Step 2: Relabel too-far fire samples ---
    too_far_mask = (selected_fire_ids != -1) & (selected_distances > fire_threshold_distance_min)
    relabelled = too_far_mask.sum().item()
    selected_fire_ids[too_far_mask] = -1
    selected_distances[too_far_mask] = float("inf")

    # --- Step 3: Discard remaining fire samples ---
    keep_mask = selected_fire_ids == -1
    discarded = total_fire - relabelled

    filtered_dataset = TensorDataset(
        selected_sequences[keep_mask],
        selected_fire_ids[keep_mask],
        selected_distances[keep_mask],
        selected_window_ids[keep_mask],
    )

    # --- Logging ---
    logger.log_step(f"Total fire samples before filtering: {total_fire}")
    logger.log_step(f"Relabelled fire samples (> {fire_threshold_distance_min}): {relabelled}")
    logger.log_step(f"Discarded fire samples (<= {fire_threshold_distance_min}): {discarded}")
    logger.log_step(f"Final dataset size: {len(filtered_dataset)}")
    logger.log_step(f"Unique fire_ids after filtering: {selected_fire_ids[keep_mask].unique().tolist()}")
    logger.log_step("Returning TensorDataset")

    return filtered_dataset




def create_dataloaders(
    dataset_path: str,
    num_samples: Optional[int],
    random_seed: int,
    train_split: float,
    val_split: float,
    batch_size: int,
    logger: Optional[ProcessLogger] = None,
    remove_fire_labels: bool = False,
    fire_threshold_distance_min: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with random split."""
    print(f"[DEBUG] logger argument: {logger}")
    if logger is None:
        print("[DEBUG] Initializing ProcessLogger with config=None")
    logger = logger or ProcessLogger(None, "DataLoader_Creation")

    dataset = _load_dataset(dataset_path, num_samples, random_seed, logger)

    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    logger.log_step(
        "Dataset split",
        {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    )
    logger.log_step(f"Train, val, and test datasets created with {len(train_dataset)} , {len(val_dataset)} , {len(test_dataset)} samples")

    # Remove fire labels from train and val sets if requested
    if remove_fire_labels:
        train_dataset = _filter_fire_labels(train_dataset, logger, fire_threshold_distance_min=fire_threshold_distance_min)
        val_dataset = _filter_fire_labels(val_dataset, logger, fire_threshold_distance_min=fire_threshold_distance_min)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_full_dataloader(
    dataset_path: str,
    num_samples: Optional[int],
    random_seed: int,
    batch_size: int,
    logger: Optional[ProcessLogger] = None,
    remove_fire_labels: bool = False,
    fire_threshold_distance_min: int = 0
) -> DataLoader:
    """Create a single dataloader for the entire dataset."""
    logger = logger or ProcessLogger(None, "Full_DataLoader_Creation")

    dataset = _load_dataset(dataset_path, num_samples, random_seed, logger)

    if remove_fire_labels:
        dataset = _filter_fire_labels(dataset, logger, fire_threshold_distance_min=fire_threshold_distance_min)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
