import optuna
import torch
import torch.nn as nn
import pandas as pd
import copy
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union
from .autoencoder import model_base
from .common.utils import set_seed
from .data.utils import create_dataloaders
from .autoencoder import evaluator
from .common.config import Config


def optuna_tune_hyperparameters(config: Config, n_trials: int = 50) -> pd.DataFrame:
    """
    Optuna-based hyperparameter optimization for the autoencoder model.
    Args:
        config: Configuration object.
        n_trials: Number of Optuna trials.
    Returns:
        pd.DataFrame: DataFrame of trial results.
    """
    device = torch.device(config.training.DEVICE)
    results = []

    def objective(trial):
        config_copy = copy.deepcopy(config)
        config_copy.tuning.LATENT_DIM = trial.suggest_int("latent_dim", 2, 32)
        config_copy.tuning.HIDDEN_DIM = trial.suggest_int("hidden_dim", 8, 128)
        config_copy.training.LEARNING_RATE = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        config_copy.training.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128])
        config_copy.training.ANOMALY_THRESHOLD_PERCENTILE = trial.suggest_categorical(
            "anomaly_threshold_percentile", [90, 95, 97, 99]
        )

        set_seed(config_copy.training.RANDOM_SEED)
        train_loader, val_loader, _ = create_dataloaders(
            config_copy.paths.DATASET_PATH,
            getattr(config_copy.data_pipeline, "NUM_SAMPLES", None),
            config_copy.training.RANDOM_SEED,
            config_copy.training.TRAIN_SPLIT,
            config_copy.training.VAL_SPLIT,
            config_copy.training.BATCH_SIZE,
            remove_fire_labels=False,
        )
        autoencoder_class = getattr(model_base, config_copy.tuning.AUTOENCODER_CLASS)
        model = autoencoder_class(
            time_steps=config_copy.data_pipeline.WINDOW_SIZE,
            num_features=len(config_copy.data_pipeline.INPUT_COLUMNS),
            latent_dim=config_copy.tuning.LATENT_DIM,
            hidden_dim=config_copy.tuning.HIDDEN_DIM,
        )
        model.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config_copy.training.LEARNING_RATE)
        best_loss = float("inf")
        patience = 5
        no_improvement = 0

        for epoch in range(min(config_copy.training.EPOCHS, 50)):
            model.train()
            for X, _, _, _ in train_loader:
                X = X.to(device)
                loss, _, _, _ = model.train_step(X, optimizer, loss_fn)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, _, _, _ in val_loader:
                    X = X.to(device)
                    loss, _, _, _ = model.val_step(X, loss_fn)
                    val_loss += loss
            val_loss /= len(val_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= patience:
                break

        f1, precision, recall, _ = evaluator.evaluate_model(config_copy, model=model)
        trial.set_user_attr("f1_score", f1)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("best_val_loss", best_loss)
        results.append(
            {
                "latent_dim": config_copy.tuning.LATENT_DIM,
                "hidden_dim": config_copy.tuning.HIDDEN_DIM,
                "learning_rate": config_copy.training.LEARNING_RATE,
                "batch_size": config_copy.training.BATCH_SIZE,
                "anomaly_threshold_percentile": config_copy.training.ANOMALY_THRESHOLD_PERCENTILE,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "best_val_loss": best_loss,
            }
        )
        return -f1 if f1 is not None else 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_result = results_df.loc[results_df["f1_score"].idxmax()]
        print("\n--- Optuna tuning complete ---")
        print(f"Best F1-score found: {best_result['f1_score']:.4f}")
        print(f"Best parameters: \n{best_result}")
        results_df.to_csv(config.paths.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH, index=False)
        print(f"Results saved to: {config.paths.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH}")
    else:
        print("No results found from Optuna tuning.")
    return results_df


# =========================================================================
# Hyperparameter Tuning Functions
# =========================================================================
def tune_hyperparameters(config: Config) -> pd.DataFrame:
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
    set_seed(config.training.RANDOM_SEED)

    device = torch.device(config.training.DEVICE)

    # Define hyperparameter grids
    latent_dims = config.tuning.HYPERPARAMETER_GRID["latent_dims"]
    hidden_dims = config.tuning.HYPERPARAMETER_GRID["hidden_dims"]
    anomaly_threshold_percentiles = config.tuning.HYPERPARAMETER_GRID["anomaly_threshold_percentiles"]

    results = []

    for ld in tqdm(latent_dims, desc="Tuning Latent Dims"):
        for hd in tqdm(hidden_dims, desc=f"Tuning Hidden Dims for LD={ld}", leave=False):
            for atp in tqdm(
                anomaly_threshold_percentiles,
                desc=f"Tuning Thresholds for LD={ld}, HD={hd}",
                leave=False,
            ):

                # Create a copy of config for this run to avoid mutations
                config_copy = copy.deepcopy(config)
                config_copy.tuning.LATENT_DIM = ld
                config_copy.tuning.HIDDEN_DIM = hd
                config_copy.training.ANOMALY_THRESHOLD_PERCENTILE = atp

                train_loader, val_loader, _ = create_dataloaders(
                    config_copy.paths.DATASET_PATH,
                    getattr(config_copy.data_pipeline, "NUM_SAMPLES", None),
                    config_copy.training.RANDOM_SEED,
                    config_copy.training.TRAIN_SPLIT,
                    config_copy.training.VAL_SPLIT,
                    config_copy.training.BATCH_SIZE,
                    remove_fire_labels=False,
                )

                # Re-initialize model with new dimensions
                # Dynamically get the autoencoder class from config
                autoencoder_class = getattr(model_base, config_copy.tuning.AUTOENCODER_CLASS)
                model = autoencoder_class(
                    time_steps=config_copy.data_pipeline.WINDOW_SIZE,
                    num_features=len(config_copy.data_pipeline.INPUT_COLUMNS),
                    latent_dim=config_copy.tuning.LATENT_DIM,
                    hidden_dim=config_copy.tuning.HIDDEN_DIM,
                )
                model.to(device)

                loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config_copy.training.LEARNING_RATE)
                best_loss = float("inf")

                # Training loop for this hyperparameter combination
                for epoch in tqdm(range(config_copy.training.EPOCHS), desc="Epochs", leave=False):
                    model.train()
                    for X, _, _, _ in train_loader:
                        X = X.to(device)
                        loss, _, _, _ = model.train_step(X, optimizer, loss_fn)

                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X, _, _, _ in val_loader:
                            X = X.to(device)
                            loss, _, _, _ = model.val_step(X, loss_fn)
                            val_loss += loss
                    val_loss /= len(val_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        # Save model temporarily for evaluation
                        torch.save(model.state_dict(), config_copy.paths.BEST_MODEL_PATH)

                # Evaluate the model with the current config copy
                f1, precision, recall, _ = evaluator.evaluate_model(config_copy, model=model)

                results.append(
                    {
                        "latent_dim": ld,
                        "hidden_dim": hd,
                        "anomaly_threshold_percentile": atp,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall,
                        "best_val_loss": best_loss,
                    }
                )

                print(f"Results: F1={f1:.4f}, Precision={precision:.4f}, " f"Recall={recall:.4f}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_result = results_df.loc[results_df["f1_score"].idxmax()]
        print("\n--- Hyperparameter tuning complete ---")
        print(f"Best F1-score found: {best_result['f1_score']:.4f}")
        print(f"Best parameters: \n{best_result}")

        # Save results
        results_df.to_csv(config.paths.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH, index=False)
        print(f"Results saved to: {config.paths.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH}")
    else:
        print("No results found from hyperparameter tuning.")

    return results_df


def tune_hyperparameters_advanced(config: Config, param_grid: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
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
        param_grid = config.tuning.ADVANCED_HYPERPARAMETER_GRID

    set_seed(config.training.RANDOM_SEED)
    device = torch.device(config.training.DEVICE)

    results = []
    total_combinations = (
        len(param_grid["latent_dims"])
        * len(param_grid["hidden_dims"])
        * len(param_grid["learning_rates"])
        * len(param_grid["batch_sizes"])
        * len(param_grid["anomaly_thresholds"])
    )

    combination_count = 0

    for ld in param_grid["latent_dims"]:
        for hd in param_grid["hidden_dims"]:
            for lr in param_grid["learning_rates"]:
                for bs in param_grid["batch_sizes"]:
                    for at in param_grid["anomaly_thresholds"]:
                        combination_count += 1

                        print(f"\n--- Combination {combination_count}/{total_combinations} ---\n")
                        print(f"Latent: {ld}, Hidden: {hd}, LR: {lr}, Batch: {bs}, Threshold: {at}")

                        # Create config copy
                        config_copy = copy.deepcopy(config)
                        config_copy.tuning.LATENT_DIM = ld
                        config_copy.tuning.HIDDEN_DIM = hd
                        config_copy.training.LEARNING_RATE = lr
                        config_copy.training.BATCH_SIZE = bs
                        config_copy.training.ANOMALY_THRESHOLD_PERCENTILE = at

                        # Train and evaluate
                        train_loader, val_loader, _ = create_dataloaders(
                            config_copy.paths.DATASET_PATH,
                            getattr(config_copy.data_pipeline, "NUM_SAMPLES", None),
                            config_copy.training.RANDOM_SEED,
                            config_copy.training.TRAIN_SPLIT,
                            config_copy.training.VAL_SPLIT,
                            config_copy.training.BATCH_SIZE,
                            remove_fire_labels=False,
                        )

                        # Dynamically get the autoencoder class from config
                        autoencoder_class = getattr(model_base, config_copy.tuning.AUTOENCODER_CLASS)
                        model = autoencoder_class(
                            time_steps=config_copy.data_pipeline.WINDOW_SIZE,
                            num_features=len(config_copy.data_pipeline.INPUT_COLUMNS),
                            latent_dim=config_copy.tuning.LATENT_DIM,
                            hidden_dim=config_copy.tuning.HIDDEN_DIM,
                        )
                        model.to(device)

                        loss_fn = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=config_copy.LEARNING_RATE)

                        best_loss = float("inf")
                        patience = 5
                        no_improvement = 0
                        epochs_completed = 0  # Track epochs completed

                        # Training with early stopping
                        for epoch in range(min(config_copy.training.EPOCHS, 50)):  # Limit epochs for tuning
                            model.train()
                            for X, _, _, _ in train_loader:
                                X = X.to(device)
                                loss, _, _, _ = model.train_step(X, optimizer, loss_fn)

                            model.eval()
                            val_loss = 0.0
                            with torch.no_grad():
                                for X, _, _, _ in val_loader:
                                    X = X.to(device)
                                    loss, _, _, _ = model.val_step(X, loss_fn)
                                    val_loss += loss
                            val_loss /= len(val_loader)

                            if val_loss < best_loss:
                                best_loss = val_loss
                                no_improvement = 0
                                torch.save(model.state_dict(), config_copy.BEST_MODEL_PATH)
                            else:
                                no_improvement += 1

                            if no_improvement >= patience:
                                print(f"Early stopping at epoch {epoch+1}")
                                epochs_completed = epoch + 1
                                break
                        else:
                            # Loop completed without break (no early stopping)
                            epochs_completed = min(config_copy.EPOCHS, 50)

                        # Evaluate
                        f1, precision, recall, threshold = evaluator.evaluate_model(config_copy, model=model)

                        result = {
                            "latent_dim": ld,
                            "hidden_dim": hd,
                            "learning_rate": lr,
                            "batch_size": bs,
                            "anomaly_threshold_percentile": at,
                            "f1_score": f1 if f1 is not None else 0.0,
                            "precision": precision if precision is not None else 0.0,
                            "recall": recall if recall is not None else 0.0,
                            "threshold": threshold if threshold is not None else 0.0,
                            "best_val_loss": best_loss,
                            "epochs_trained": epochs_completed,
                            "status": "success",
                        }

                        print(
                            f"F1: {result['f1_score']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}"
                        )

                        results.append(result)

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Find best result
        valid_results = results_df[results_df["status"] == "success"]
        if not valid_results.empty:
            best_result = valid_results.loc[valid_results["f1_score"].idxmax()]
            print(f"\n=== BEST HYPERPARAMETERS FOUND ===")
            print(f"Best F1-Score: {best_result['f1_score']:.4f}")
            print(f"Parameters:")
            for key in [
                "latent_dim",
                "hidden_dim",
                "learning_rate",
                "batch_size",
                "anomaly_threshold_percentile",
            ]:
                print(f"  {key}: {best_result[key]}")

        # Save detailed results
        results_df.to_csv(config.paths.ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH, index=False)
        print(f"\nDetailed results saved to: {config.paths.ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH}")

    return results_df
