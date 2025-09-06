import os
import sys
import logging
import argparse
import torch

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
    OUTPUT_DIR = output_dir
    MODEL_DIR = model_dir
    DATASET_DIR = dataset_dir

    # Set environment variables for the Config class
    os.environ['OUTPUT_DIR'] = OUTPUT_DIR
    os.environ['MODEL_DIR'] = MODEL_DIR
    os.environ['DATASET_DIR'] = DATASET_DIR
    os.environ['DATA_DIR'] = DATA_DIR

    def objective(trial):
        config = Config()
        config.data_pipeline.LOCAL_OFFSET_MINUTES = 420
        config.data_pipeline.WINDOW_DURATION_MINUTES = 120

        print(f'Processing with window size {120} minutes and offset 7h')
        print(f"Using data directory: {config.paths.DATA_DIR}")
        print(f"Using output directory: {config.paths.OUTPUT_DIR}")
        print(f"Using model directory: {config.paths.MODEL_DIR}")

        # Ensure directories exist
        os.makedirs(config.paths.DATA_DIR, exist_ok=True)
        os.makedirs(config.paths.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.paths.MODEL_DIR, exist_ok=True)

        num_features = 6

        # Suggest hyperparameters
        config.tuning.LATENT_DIM = trial.suggest_categorical("latent_dim", [18, 36, 54, 64])
        config.training.LEARNING_RATE = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        config.training.BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
        config.training.ANOMALY_THRESHOLD_PERCENTILE = trial.suggest_float("anomaly_threshold_percentile", 99.9, 99.99)

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
        config.training.DEVICE = "cpu"
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
            anomaly_scores, all_window_ids, all_fire_ids, all_distances,
            config.training.ANOMALY_THRESHOLD_PERCENTILE, True
        )
        print("********")
        print(result)
        f1_score = result.get("f1", 0.0)
        print(f"Trial {trial.number}: F1-score={f1_score:.4f} with params {trial.params}")
        return f1_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  F1-score: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Train and evaluate the Forest Fire Detection model.")
    #parser.add_argument("dataset_dir", help="Directory containing the dataset")
    #parser.add_argument("model_dir", help="Directory to save the trained model")
    #parser.add_argument("output_dir", help="Directory to save output files")

    #args = parser.parse_args()
    main('dataset_120min_7h_9x', 'model_120min_7h_9x', 'output_120min_7h_9x')
    #main(args.dataset_dir, args.model_dir, args.output_dir)
