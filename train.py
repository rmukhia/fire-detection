import os
import sys
import logging
import argparse
import torch
import json
import json

import optuna

# Add 'aad' directory to sys.path for module imports
sys.path.insert(0, os.path.abspath("aad"))

from aad.common.config import Config
from aad.common.reload_all import reload_all
import aad.autoencoder.trainer_standard as trainer_standard
import aad.autoencoder.utils as utils
import aad.autoencoder.evaluator as evaluator
import aad.autoencoder.clustering as clustering
from aad.data.utils import create_full_dataloader, create_dataloaders
from aad.common.core_logging import ProcessLogger
from aad.autoencoder.trainer_standard import StandardTrainer
from aad.autoencoder.model_dense import Autoencoder
from aad.autoencoder.evaluator import ModelEvaluator

def main(dataset_dir, model_dir, output_dir):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set directories from arguments
    DATA_DIR = "data"
    # OUTPUT_DIR and MODEL_DIR will be set per trial inside objective
    OUTPUT_DIR = output_dir
    MODEL_DIR = model_dir
    DATASET_DIR = dataset_dir

    # No environment variables needed for Config

    def objective(trial):
        # Set trial-specific output/model directories
        trial_output_dir = os.path.join(output_dir, f"trial{trial.number}")
        trial_model_dir = os.path.join(model_dir, f"trial{trial.number}")

        # No environment variables needed for Config

        config = Config(
            data_dir=DATA_DIR,
            output_dir=trial_output_dir,
            model_dir=trial_model_dir,
            dataset_dir=DATASET_DIR
        )
        config.data_pipeline.LOCAL_OFFSET_MINUTES = 420
        config.data_pipeline.WINDOW_DURATION_MINUTES = 120

        print(f'Processing with window size {120} minutes and offset 7h')
        print(f"Using data directory: {config.paths.DATA_DIR}")
        print(f"Using output directory: {config.paths.OUTPUT_DIR}")
        print(f"Using model directory: {config.paths.MODEL_DIR}")

        # Ensure directories exist

        num_features = 9

        # Suggest hyperparameters
        config.tuning.LATENT_DIM = trial.suggest_categorical("latent_dim", [36])
        #config.training.LEARNING_RATE = trial.suggest_loguniform("learning_rate", 1e-5, 1e-4)
        config.training.LEARNING_RATE = 0.00004114867892071177
        config.training.BATCH_SIZE = trial.suggest_categorical("batch_size", [256])
        config.training.OPTIMIZER = trial.suggest_categorical("optimizer", ["SGD"])
        config.training.LOSS_FUNCTION = trial.suggest_categorical("loss_function", ["L1Loss"])
        config.training.SCHEDULER = trial.suggest_categorical("scheduler", ["CosineAnnealingLR"])

        model = Autoencoder(
            time_steps=60,
            num_features=num_features,
            latent_dim=config.tuning.LATENT_DIM,
            dropout=0.0,
            d_model=num_features,
            num_heads=3,
        )

        # Training configuration
        config.training.RANDOM_SEED = 42
        config.training.EPOCHS = 50
        config.training.PATIENCE = 25
        config.training.USE_BETA_SCHEDULE = True
        config.training.BETA_SCHEDULE_TYPE = "cosine"
        config.training.DEVICE = "cuda"
        config.data_pipeline.NUM_SAMPLES = 600_000

        logger = ProcessLogger(config, "trainer")
        trainer = StandardTrainer(
            logger=logger,
            random_seed=config.training.RANDOM_SEED,
            epochs=config.training.EPOCHS,
            patience=config.training.PATIENCE,
            learning_rate=config.training.LEARNING_RATE,
            loss_function_name=config.training.LOSS_FUNCTION,
            optimizer_name=config.training.OPTIMIZER,
            batch_size=config.training.BATCH_SIZE,
            latent_dim=config.tuning.LATENT_DIM,
            window_size=config.data_pipeline.WINDOW_SIZE,
            num_features=len(config.data_pipeline.INPUT_COLUMNS),
            device=config.training.DEVICE,
            stats_images_dir=config.paths.STATS_IMAGES_DIR,
            training_statistics_image_path=config.paths.TRAINING_STATISTICS_IMAGE_PATH,
            best_model_path=config.paths.BEST_MODEL_PATH,
            loss_history_path=config.paths.LOSS_HISTORY_PATH,
            callbacks=None,
        )

        config.training.TRAIN_SPLIT = 0.8
        config.training.VAL_SPLIT = 0.2

        train_loader, val_loader, test_loader = create_dataloaders(
            config.paths.DATASET_PATH,
            config.data_pipeline.NUM_SAMPLES,
            config.training.RANDOM_SEED,
            config.training.TRAIN_SPLIT,
            config.training.VAL_SPLIT,
            config.training.BATCH_SIZE,
            ProcessLogger(config, "dataloader"),
            remove_fire_labels=True,
            fire_threshold_distance_min=20000,
        )

        model, _ = trainer.train(model, train_loader, val_loader, True)

        # Evaluation
        test_loader = create_full_dataloader(
            config.paths.DATASET_PATH,
            None,
            config.training.RANDOM_SEED,
            config.training.BATCH_SIZE,
            ProcessLogger(config, "dataloader"),
        )

        logger = ProcessLogger(config, "evaluator")
        model.load_state_dict(torch.load(config.paths.BEST_MODEL_PATH, map_location=torch.device(config.training.DEVICE)))
        config.training.DISTANCE_FILTER_THRESHOLD_M = 7200
        evaluator_obj = ModelEvaluator(
            distance_filter_threshold_m=config.training.DISTANCE_FILTER_THRESHOLD_M,
            device=torch.device(config.training.DEVICE),
            logger=logger,
            model=model,
            test_loader=test_loader,
            stats_images_dir=config.paths.STATS_IMAGES_DIR,
            stats_csv_dir=config.paths.STATS_CSV_DIR,
            eval_stats_img_path=config.paths.EVALUATION_STATISTICS_IMAGE_PATH,
            eval_results_csv_path=config.paths.EVALUATION_RESULTS_CSV_PATH,
            eval_summary_json_path=config.paths.EVALUATION_SUMMARY_JSON_PATH,
        )

        anomaly_scores, all_window_ids, all_fire_ids, all_distances  = evaluator_obj.compute_anomaly_scores()

        result = evaluator_obj.evaluate_model(
            anomaly_scores,
            all_window_ids,
            all_fire_ids,
            all_distances,
            threshold_mode="optimal",
            metric="f1",
            save_stats=True
        )
        print("********")
        #print(result)
        trial_output_dir = os.path.join(output_dir, f"trial{trial.number}")
        os.makedirs(trial_output_dir, exist_ok=True)
        result_path = os.path.join(trial_output_dir, "result.csv")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {result_path}")
        f1_score = result.get("f1", 0.0)
        print(f"Trial {trial.number}: F1-score={f1_score:.4f} with params {trial.params}")
        # Save hyperparameters for this trial
        hyperparam_path = os.path.join(trial_output_dir, "hyperparameter.json")
        with open(hyperparam_path, "w") as f:
            json.dump(trial.params, f, indent=2)
        print(f"Hyperparameters saved to {hyperparam_path}")
        return f1_score

    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna_study2.db", study_name="forest_fire_detection-2", load_if_exists=True)
    study.optimize(objective, n_trials=1)

    print("Best trial:")
    trial = study.best_trial
    print(f"  F1-score: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best trial hyperparameters to <modeldir>/best_trial.json
    best_trial_json = os.path.join(model_dir, "best_trial.json")
    with open(best_trial_json, "w") as f:
        json.dump(trial.params, f, indent=2)
    print(f"Best trial hyperparameters saved to {best_trial_json}")

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Train and evaluate the Forest Fire Detection model.")
    #parser.add_argument("dataset_dir", help="Directory containing the dataset")
    #parser.add_argument("model_dir", help="Directory to save the trained model")
    #parser.add_argument("output_dir", help="Directory to save output files")

    #args = parser.parse_args()
    main('dataset_120min_7h_9x_slope', 'model_120min_7h_9x_slope', 'output_120min_7h_9x_slope')
    #main(args.dataset_dir, args.model_dir, args.output_dir)
