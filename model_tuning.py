"""
Hyperparameter tuning utilities for the forest fire detection system.

Provides functions for grid search and advanced hyperparameter optimization
of autoencoder models. Includes early stopping and performance tracking.
"""
import torch
import torch.nn as nn
import pandas as pd
import copy
from tqdm.auto import tqdm
import model_autoencoder
import model_utils
import model_evaluator

# =========================================================================
# Hyperparameter Tuning Functions
# =========================================================================
def tune_hyperparameters(config: object) -> pd.DataFrame:
    """
    Comprehensive hyperparameter tuning for the autoencoder model.
    
    Performs grid search over latent dimensions, hidden dimensions and
    anomaly thresholds. Tracks F1-score, precision and recall metrics.
    
    Args:
        config: Configuration object containing:
            - HYPERPARAMETER_GRID: Dictionary of parameter ranges
            - Other training/evaluation parameters
            
    Returns:
        pd.DataFrame: Contains columns:
            - latent_dim: Tested latent dimensions
            - hidden_dim: Tested hidden dimensions
            - anomaly_threshold_percentile: Tested thresholds
            - f1_score: Best F1-score for each combination
            - precision: Best precision for each combination
            - recall: Best recall for each combination
            - threshold: Optimal threshold for each combination
            - best_val_loss: Best validation loss achieved
    """
    model_utils.set_seed(config.RANDOM_SEED)

    device = torch.device(config.DEVICE)

    # Define hyperparameter grids
    latent_dims = config.HYPERPARAMETER_GRID['latent_dims']
    hidden_dims = config.HYPERPARAMETER_GRID['hidden_dims']
    anomaly_threshold_percentiles = config.HYPERPARAMETER_GRID['anomaly_threshold_percentiles']

    results = []

    for ld in tqdm(latent_dims, desc="Tuning Latent Dims"):
        for hd in tqdm(
            hidden_dims, desc=f"Tuning Hidden Dims for LD={ld}", leave=False
        ):
            for atp in tqdm(
                anomaly_threshold_percentiles,
                desc=f"Tuning Thresholds for LD={ld}, HD={hd}",
                leave=False,
            ):
                

                # Create a copy of config for this run to avoid mutations
                config_copy = copy.deepcopy(config)
                config_copy.LATENT_DIM = ld
                config_copy.HIDDEN_DIM = hd
                config_copy.ANOMALY_THRESHOLD_PERCENTILE = atp

                train_loader, val_loader, _ = model_utils.create_dataloaders(config_copy, remove_fire_labels=False)

                # Re-initialize model with new dimensions
                # Dynamically get the autoencoder class from config
                autoencoder_class = getattr(model_autoencoder, config_copy.AUTOENCODER_CLASS)
                model = autoencoder_class(
                            time_steps=config_copy.WINDOW_SIZE,
                            num_features=len(config_copy.INPUT_COLUMNS),
                            latent_dim=config_copy.LATENT_DIM,
                            hidden_dim=config_copy.HIDDEN_DIM,
                        )
                model.to(device)

                loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config_copy.LEARNING_RATE
                )
                best_loss = float('inf')

                # Training loop for this hyperparameter combination
                for epoch in tqdm(
                    range(config_copy.EPOCHS), desc="Epochs", leave=False
                ):
                    model.train_step(train_loader, optimizer, loss_fn)
                    val_loss, _, _, _ = model.val_step(val_loader, loss_fn)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        # Save model temporarily for evaluation
                        torch.save(
                            model.state_dict(), config_copy.BEST_MODEL_PATH
                        )

                # Evaluate the model with the current config copy
                f1, precision, recall, _ = model_evaluator.evaluate_model(config_copy)

                results.append({
                    'latent_dim': ld,
                    'hidden_dim': hd,
                    'anomaly_threshold_percentile': atp,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'best_val_loss': best_loss
                })

                print(
                    f"Results: F1={f1:.4f}, Precision={precision:.4f}, "
                    f"Recall={recall:.4f}"
                )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_result = results_df.loc[results_df['f1_score'].idxmax()]
        print("\n--- Hyperparameter tuning complete ---")
        print(f"Best F1-score found: {best_result['f1_score']:.4f}")
        print(f"Best parameters: \n{best_result}")
        
        # Save results
        results_df.to_csv(config.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH, index=False)
        print(f"Results saved to: {config.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH}")
    else:
        print("No results found from hyperparameter tuning.")

    return results_df


def tune_hyperparameters_advanced(config: object, param_grid: dict = None) -> pd.DataFrame:
    """
    Advanced hyperparameter tuning with custom parameter grid and early stopping.
    
    Supports tuning of learning rates, batch sizes in addition to model architecture.
    Implements early stopping based on validation loss.
    
    Args:
        config: Configuration object containing training parameters
        param_grid: Dictionary with parameter ranges containing:
            - latent_dims: List of latent dimensions to test
            - hidden_dims: List of hidden dimensions to test
            - learning_rates: List of learning rates to test
            - batch_sizes: List of batch sizes to test
            - anomaly_thresholds: List of anomaly thresholds to test
            
    Returns:
        pd.DataFrame: Contains columns:
            - latent_dim: Tested latent dimensions
            - hidden_dim: Tested hidden dimensions
            - learning_rate: Tested learning rates
            - batch_size: Tested batch sizes
            - anomaly_threshold_percentile: Tested thresholds
            - f1_score: Best F1-score for each combination
            - precision: Best precision for each combination
            - recall: Best recall for each combination
            - threshold: Optimal threshold for each combination
            - best_val_loss: Best validation loss achieved
            - epochs_trained: Number of epochs completed
            - status: 'success' or 'failure'
    """
    if param_grid is None:
        param_grid = config.ADVANCED_HYPERPARAMETER_GRID
    
    model_utils.set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    
    results = []
    total_combinations = (
        len(param_grid['latent_dims']) * 
        len(param_grid['hidden_dims']) * 
        len(param_grid['learning_rates']) * 
        len(param_grid['batch_sizes']) * 
        len(param_grid['anomaly_thresholds'])
    )
    
    combination_count = 0
    
    for ld in param_grid['latent_dims']:
        for hd in param_grid['hidden_dims']:
            for lr in param_grid['learning_rates']:
                for bs in param_grid['batch_sizes']:
                    for at in param_grid['anomaly_thresholds']:
                        combination_count += 1
                        
                        print(f"\n--- Combination {combination_count}/{total_combinations} ---")
                        print(f"Latent: {ld}, Hidden: {hd}, LR: {lr}, Batch: {bs}, Threshold: {at}")
                        
                        # Create config copy
                        config_copy = copy.deepcopy(config)
                        config_copy.LATENT_DIM = ld
                        config_copy.HIDDEN_DIM = hd
                        config_copy.LEARNING_RATE = lr
                        config_copy.BATCH_SIZE = bs
                        config_copy.ANOMALY_THRESHOLD_PERCENTILE = at
                        
                        # Train and evaluate
                        train_loader, val_loader, _ = model_utils.create_dataloaders(config_copy, remove_fire_labels=False)
                        
                        # Dynamically get the autoencoder class from config
                        autoencoder_class = getattr(model_autoencoder, config_copy.AUTOENCODER_CLASS)
                        model = autoencoder_class(
                                            time_steps=config_copy.WINDOW_SIZE,
                                            num_features=len(config_copy.INPUT_COLUMNS),
                                            latent_dim=config_copy.LATENT_DIM,
                                            hidden_dim=config_copy.HIDDEN_DIM
                                        )
                        model.to(device)
                        
                        loss_fn = nn.MSELoss()
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=config_copy.LEARNING_RATE
                        )
                        
                        best_loss = float('inf')
                        patience = 5
                        no_improvement = 0
                        
                        # Training with early stopping
                        for epoch in range(min(config_copy.EPOCHS, 50)):  # Limit epochs for tuning
                            model.train_step(train_loader, optimizer, loss_fn)
                            val_loss, _, _, _ = model.val_step(val_loader, loss_fn)
                            
                            if val_loss < best_loss:
                                best_loss = val_loss
                                no_improvement = 0
                                torch.save(model.state_dict(), config_copy.BEST_MODEL_PATH)
                            else:
                                no_improvement += 1
                                
                            if no_improvement >= patience:
                                print(f"Early stopping at epoch {epoch+1}")
                                break
                        
                        # Evaluate
                        f1, precision, recall, threshold = model_evaluator.evaluate_model(config_copy)
                        
                        result = {
                            'latent_dim': ld,
                            'hidden_dim': hd,
                            'learning_rate': lr,
                            'batch_size': bs,
                            'anomaly_threshold_percentile': at,
                            'f1_score': f1 if f1 is not None else 0.0,
                            'precision': precision if precision is not None else 0.0,
                            'recall': recall if recall is not None else 0.0,
                            'threshold': threshold if threshold is not None else 0.0,
                            'best_val_loss': best_loss,
                            'epochs_trained': epoch + 1,
                            'status': 'success'
                        }
                        
                        print(f"F1: {result['f1_score']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}")
                        
                        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Find best result
        valid_results = results_df[results_df['status'] == 'success']
        if not valid_results.empty:
            best_result = valid_results.loc[valid_results['f1_score'].idxmax()]
            print(f"\n=== BEST HYPERPARAMETERS FOUND ===")
            print(f"Best F1-Score: {best_result['f1_score']:.4f}")
            print(f"Parameters:")
            for key in ['latent_dim', 'hidden_dim', 'learning_rate', 'batch_size', 'anomaly_threshold_percentile']:
                print(f"  {key}: {best_result[key]}")
        
        # Save detailed results
        results_df.to_csv(config.ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH, index=False)
        print(f"\nDetailed results saved to: {config.ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH}")
    
    return results_df
