"""Utilities for model training and data loading in forest fire detection."""
import torch
import numpy as np
import random
import os
from typing import Tuple, Optional
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset, Subset
from core_logging import ProcessLogger


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed value for all random number generators
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(config, logger: Optional[ProcessLogger] = None, remove_fire_labels: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders with a simple naive split.
    
    This function loads the preprocessed dataset, performs a simple random split,
    and optionally removes fire-labeled data from the training and validation sets.
    
    Args:
        config: Configuration object with dataset parameters
        logger: Optional ProcessLogger instance for structured logging.
        remove_fire_labels: If True, removes fire-labeled data from train and val sets.
                
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader instances
        
    Raises:
        FileNotFoundError: If the dataset file cannot be found
        ValueError: If required configuration parameters are missing or invalid
    """
    # Initialize logger if not provided
    logger = logger or ProcessLogger(config, "DataLoader_Creation")
    
    # Validate required configuration parameters
    required_params = ['DATASET_PATH', 'TRAIN_SPLIT', 'VAL_SPLIT', 'BATCH_SIZE', 'RANDOM_SEED']
    for param in required_params:
        if not hasattr(config, param):
            raise ValueError(f"Missing required configuration parameter: {param}")
    
    logger.log_step("Starting dataloader creation process")
    
    # Load dataset with error handling
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {config.DATASET_PATH}")
    
    try:
        sequences, fire_ids, distances, window_ids = torch.load(config.DATASET_PATH)
        dataset = TensorDataset(sequences, fire_ids, distances, window_ids)
        logger.log_step("Dataset loaded successfully", {'dataset_path': config.DATASET_PATH})
    except Exception as e:
        logger.log_error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Simple naive split
    total_size = len(dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    logger.log_step("Dataset statistics", {
        'total_samples': total_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    })

    if remove_fire_labels:
        logger.log_step("Removing fire-labeled data from train and validation sets")

        def filter_fire_labels(dataset_subset):
            original_indices = dataset_subset.indices
            original_dataset = dataset_subset.dataset
            
            subset_fire_ids = original_dataset.tensors[1][original_indices]
            
            non_fire_mask = subset_fire_ids == -1
            
            non_fire_indices = np.array(original_indices)[non_fire_mask]
            
            return Subset(original_dataset, non_fire_indices)

        train_dataset = filter_fire_labels(train_dataset)
        val_dataset = filter_fire_labels(val_dataset)

        logger.log_step("Dataset sizes after removing fire labels", {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
        })

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    logger.log_step("Dataloaders created successfully", {
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader),
        'batch_size': config.BATCH_SIZE
    })

    return train_loader, val_loader, test_loader

def create_full_dataloader(config, logger: Optional[ProcessLogger] = None, remove_fire_labels: bool = False) -> DataLoader:
    """
    Create a single dataloader for the entire dataset.
    
    Args:
        config: Configuration object with dataset parameters
        logger: Optional ProcessLogger instance for structured logging.
                
    Returns:
        DataLoader: A DataLoader instance for the entire dataset.
        
    Raises:
        FileNotFoundError: If the dataset file cannot be found
    """
    # Initialize logger if not provided
    logger = logger or ProcessLogger(config, "DataLoader_Creation")
    
    # Load dataset with error handling
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {config.DATASET_PATH}")
    
    try:
        sequences, fire_ids, distances, window_ids = torch.load(config.DATASET_PATH)
        dataset = TensorDataset(sequences, fire_ids, distances, window_ids)
        logger.log_step("Dataset loaded successfully", {'dataset_path': config.DATASET_PATH})
    except Exception as e:
        logger.log_error(f"Failed to load dataset: {str(e)}")
        raise

    if remove_fire_labels:
        logger.log_step("Removing fire-labeled data from the dataset")
        non_fire_indices = (fire_ids == -1).nonzero(as_tuple=True)[0]
        dataset = Subset(dataset, non_fire_indices)
        logger.log_step(f"Dataset size after removing fire labels: {len(dataset)}")

    full_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    logger.log_step("Full dataloader created successfully", {
        'total_samples': len(dataset),
        'batch_size': config.BATCH_SIZE,
        'num_batches': len(full_loader)
    })

    return full_loader