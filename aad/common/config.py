import os
from typing import Dict, List, Any
import yaml
import torch
import psutil
from .core_logging import ensure_directories


def load_yaml_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_optimal_device_config() -> str:
    """Auto-detect optimal device for training."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_optimal_workers() -> int:
    """Get optimal number of workers based on system resources."""
    return max(1, min(psutil.cpu_count(logical=False), psutil.virtual_memory().total // (2 * 1024**3)))


_YAML_CONFIG = load_yaml_config()


class PathsConfig:
    """
    Handles all path-related variables: data, logs, stats, output, etc.
    """

    def __init__(self):
        cfg = _YAML_CONFIG.get("paths", {})
        self.DATA_DIR = os.environ.get("DATA_DIR", cfg.get("DATA_DIR", "data"))
        self.OUTPUT_DIR = os.environ.get("OUTPUT_DIR", cfg.get("OUTPUT_DIR", "output"))
        self.DATASET_DIR = os.environ.get("DATASET_DIR", cfg.get("DATASET_DIR", "datasets"))
        self.ANNOTATED_DATA_DIR = os.path.join(self.OUTPUT_DIR, cfg.get("ANNOTATED_DATA_DIR", "annotated_data"))
        self.PROCESSED_DATA_DIR = os.path.join(self.OUTPUT_DIR, cfg.get("PROCESSED_DATA_DIR", "processed_data"))
        self.INTERMEDIATE_DIR = os.path.join(self.OUTPUT_DIR, cfg.get("INTERMEDIATE_DIR", "intermediate"))
        self.STATS_DIR = os.path.join(self.OUTPUT_DIR, cfg.get("STATS_DIR", "stats"))
        self.STATS_IMAGES_DIR = os.path.join(self.STATS_DIR, cfg.get("STATS_IMAGES_DIR", "images"))
        self.STATS_CSV_DIR = os.path.join(self.STATS_DIR, cfg.get("STATS_CSV_DIR", "csv"))
        self.STATS_HTML_DIR = os.path.join(self.STATS_DIR, cfg.get("STATS_HTML_DIR", "html"))
        self.LOGS_DIR = os.path.join(self.OUTPUT_DIR, cfg.get("LOGS_DIR", "logs"))
        self.MODEL_DIR = os.environ.get("MODEL_DIR", cfg.get("MODEL_DIR", "models"))

        # Raw and processed data files
        self.RAW_DATA_PATH = os.path.join(self.DATA_DIR, cfg.get("RAW_DATA_PATH", "sensor_data.csv"))
        self.LOCATION_DATA_PATH = os.path.join(self.DATA_DIR, cfg.get("LOCATION_DATA_PATH", "dim_location.csv"))
        self.PROCESSED_DATA_PATH = os.path.join(
            self.PROCESSED_DATA_DIR,
            cfg.get("PROCESSED_DATA_PATH", "consolidated_sensor_data.parquet"),
        )
        self.LABEL_DATA_PATH = os.path.join(self.DATASET_DIR, cfg.get("LABEL_DATA_PATH", "label.csv"))
        self.DISTANCE_ANNOTATED_PATH = os.path.join(
            self.ANNOTATED_DATA_DIR,
            cfg.get("DISTANCE_ANNOTATED_PATH", "sensor_data_window_fire_distance.parquet"),
        )
        self.ANNOTATED_DATA_PATH = os.path.join(
            self.ANNOTATED_DATA_DIR,
            cfg.get("ANNOTATED_DATA_PATH", "annotated_data.parquet"),
        )
        self.DATASET_PATH = os.path.join(self.DATASET_DIR, cfg.get("DATASET_PATH", "dataset.pt"))

        # Model/scaler paths
        self.SCALER_PATH = os.path.join(self.MODEL_DIR, cfg.get("SCALER_PATH", "scaler.joblib"))
        self.BEST_MODEL_PATH = os.path.join(self.MODEL_DIR, cfg.get("BEST_MODEL_PATH", "best_model.pth"))

        # Output/statistics/log paths (to be set by MainConfig)
        self.LOSS_HISTORY_PATH: str = ""
        self.PROCESSING_METRICS_CSV_PATH: str = ""
        self.PROCESSING_METADATA_JSON_PATH: str = ""
        self.PROCESSING_SUMMARY_JSON_PATH: str = ""
        self.EVALUATION_RESULTS_CSV_PATH: str = ""
        self.EVALUATION_SUMMARY_JSON_PATH: str = ""
        self.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH: str = ""
        self.ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH: str = ""
        self.PREPROCESSING_STATISTICS_IMAGE_PATH: str = ""
        self.SEQUENCE_CREATION_STATISTICS_IMAGE_PATH: str = ""
        self.EVALUATION_STATISTICS_IMAGE_PATH: str = ""
        self.TRAINING_STATISTICS_IMAGE_PATH: str = ""
        self.PROGRESS_BAR_LOG_FILE: str = ""

        # Create directories
        directories = [
            self.DATA_DIR,
            self.OUTPUT_DIR,
            self.DATASET_DIR,
            self.ANNOTATED_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.INTERMEDIATE_DIR,
            self.STATS_DIR,
            self.STATS_IMAGES_DIR,
            self.STATS_CSV_DIR,
            self.STATS_HTML_DIR,
            self.LOGS_DIR,
            self.MODEL_DIR,
        ]
        ensure_directories(directories)


class DataPipelineConfig:
    """
    Variables for data_preprocessing.py, data_annotation.py, data_sequences.py
    """

    def __init__(self):
        cfg = _YAML_CONFIG.get("data_pipeline", {})
        self.NUM_SAMPLES = int(cfg.get("NUM_SAMPLES", 10000000))
        self.RESAMPLE_INTERVAL = cfg.get("RESAMPLE_INTERVAL", "2min")
        self.WINDOW_DURATION_MINUTES = int(cfg.get("WINDOW_DURATION_MINUTES", 120))
        self.SAMPLING_INTERVAL_SECONDS = int(cfg.get("SAMPLING_INTERVAL_SECONDS", 120))
        self.EPSG_ZONE = cfg.get("EPSG_ZONE", "EPSG:32647")
        self.INPUT_COLUMNS = cfg.get("INPUT_COLUMNS", ["PM2.5", "Carbon dioxide (CO2)", "Relative humidity"])
        self.LOCAL_OFFSET_MINUTES = int(cfg.get("LOCAL_OFFSET_MINUTES", 420))
        self.NUM_WORKERS = int(cfg.get("NUM_WORKERS", get_optimal_workers()))
        self.RESAMPLE_TOLERANCE_FACTOR = float(cfg.get("RESAMPLE_TOLERANCE_FACTOR", 0.5))
        self.WINDOW_STEP_SIZE = int(cfg.get("WINDOW_STEP_SIZE", 1))
        self.DURATION_FILTER_TOLERANCE = cfg.get("DURATION_FILTER_TOLERANCE", "5s")
        self.SMA_MULTIPLIERS = cfg.get("SMA_MULTIPLIERS", [3, 6, 9, 12])

    @property
    def WINDOW_SIZE(self) -> int:
        return self.WINDOW_DURATION_MINUTES * 60 // self.SAMPLING_INTERVAL_SECONDS

    @property
    def INPUT_DIM(self) -> int:
        return len(self.INPUT_COLUMNS) * self.WINDOW_SIZE


class TrainingConfig:
    """
    Variables for model training, evaluation, metrics
    """

    def __init__(self):
        cfg = _YAML_CONFIG.get("training", {})
        self.EPOCHS = int(os.environ.get("EPOCHS", cfg.get("EPOCHS", 50)))
        self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", cfg.get("BATCH_SIZE", 32)))
        self.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", cfg.get("LEARNING_RATE", 1e-3)))
        self.RANDOM_SEED = int(os.environ.get("RANDOM_SEED", cfg.get("RANDOM_SEED", 42)))
        self.TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", cfg.get("TRAIN_SPLIT", 0.7)))
        self.VAL_SPLIT = float(os.environ.get("VAL_SPLIT", cfg.get("VAL_SPLIT", 0.15)))
        self.PATIENCE = int(os.environ.get("PATIENCE", cfg.get("PATIENCE", 25)))
        self.LOSS_FUNCTION = os.environ.get("LOSS_FUNCTION", cfg.get("LOSS_FUNCTION", "MSELoss"))
        self.OPTIMIZER = os.environ.get("OPTIMIZER", cfg.get("OPTIMIZER", "Adam"))
        self.USE_BETA_SCHEDULE = (
            os.environ.get("USE_BETA_SCHEDULE", str(cfg.get("USE_BETA_SCHEDULE", True))).lower() == "true"
        )
        self.BETA_SCHEDULE_TYPE = os.environ.get("BETA_SCHEDULE_TYPE", cfg.get("BETA_SCHEDULE_TYPE", "linear"))
        self.USE_DISTRIBUTED = (
            os.environ.get("USE_DISTRIBUTED", str(cfg.get("USE_DISTRIBUTED", False))).lower() == "true"
        )
        self.DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", cfg.get("DISTRIBUTED_BACKEND", "nccl"))
        self.USE_DATA_PARALLEL = (
            os.environ.get("USE_DATA_PARALLEL", str(cfg.get("USE_DATA_PARALLEL", False))).lower() == "true"
        )
        self.USE_MIXED_PRECISION = (
            os.environ.get("USE_MIXED_PRECISION", str(cfg.get("USE_MIXED_PRECISION", False))).lower() == "true"
        )
        self.DISTANCE_FILTER_THRESHOLD_M = int(cfg.get("DISTANCE_FILTER_THRESHOLD_M", 5000))
        self.ANOMALY_THRESHOLD_PERCENTILE = float(cfg.get("ANOMALY_THRESHOLD_PERCENTILE", 99.95))
        self.LOG_FILENAME_PATTERN = cfg.get("LOG_FILENAME_PATTERN", "%Y-%m-%d_%H-%M-%S.log")
        device_config = cfg.get("DEVICE", "auto")
        if device_config == "auto":
            self.DEVICE = get_optimal_device_config()
        else:
            self.DEVICE = device_config


class TuningConfig:
    """
    Model tuning/hyperparameter variables
    """

    def __init__(self):
        cfg = _YAML_CONFIG.get("tuning", {})
        self.AUTOENCODER_CLASS = cfg.get("AUTOENCODER_CLASS", "VariationalAutoencoder")
        self.LATENT_DIM = int(cfg.get("LATENT_DIM", 12))
        self.HIDDEN_DIM = int(cfg.get("HIDDEN_DIM", 360))
        self.CONV_NUM_LAYERS = int(cfg.get("CONV_NUM_LAYERS", 4))
        self.CONV_KERNEL_SIZE = int(cfg.get("CONV_KERNEL_SIZE", 5))
        self.CONV_DROPOUT_RATE = float(cfg.get("CONV_DROPOUT_RATE", 0.1))
        self.CONV_USE_ATTENTION = bool(cfg.get("CONV_USE_ATTENTION", True))
        self.CONV_USE_RESIDUAL = bool(cfg.get("CONV_USE_RESIDUAL", True))
        self.KMEANS_N_CLUSTERS = int(cfg.get("KMEANS_N_CLUSTERS", 32))
        self.DBSCAN_EPS = float(cfg.get("DBSCAN_EPS", 0.3))
        self.DBSCAN_MIN_SAMPLES = int(cfg.get("DBSCAN_MIN_SAMPLES", 5))
        self.DIM_REDUCTION_METHOD = cfg.get("DIM_REDUCTION_METHOD", "tsne")
        self.HYPERPARAMETER_GRID = cfg.get(
            "HYPERPARAMETER_GRID",
            {
                "latent_dims": [18, 36],
                "hidden_dims": [180, 360],
                "anomaly_threshold_percentiles": [99.9, 99.98],
            },
        )
        self.ADVANCED_HYPERPARAMETER_GRID = cfg.get(
            "ADVANCED_HYPERPARAMETER_GRID",
            {
                "latent_dims": [18, 36, 54],
                "hidden_dims": [180, 360, 540],
                "learning_rates": [0.001, 0.0005, 0.0001],
                "batch_sizes": [16, 32, 64],
                "anomaly_thresholds": [99.9, 99.95, 99.98],
            },
        )


class MainConfig:
    """
    Composes PathsConfig, DataPipelineConfig, TrainingConfig, TuningConfig.
    Sets up cross-config paths and parameters.
    """

    def __init__(self):
        self.paths = PathsConfig()
        self.data_pipeline = DataPipelineConfig()
        self.training = TrainingConfig()
        self.tuning = TuningConfig()

        # Set output/statistics/log paths in paths config
        self.paths.LOSS_HISTORY_PATH = os.path.join(self.paths.STATS_CSV_DIR, "loss_history.json")
        self.paths.PROCESSING_METRICS_CSV_PATH = os.path.join(self.paths.STATS_CSV_DIR, "processing_metrics.csv")
        self.paths.PROCESSING_METADATA_JSON_PATH = os.path.join(self.paths.STATS_CSV_DIR, "processing_metadata.json")
        self.paths.PROCESSING_SUMMARY_JSON_PATH = os.path.join(self.paths.STATS_CSV_DIR, "processing_summary.json")
        self.paths.EVALUATION_RESULTS_CSV_PATH = os.path.join(self.paths.STATS_CSV_DIR, "evaluation_results.csv")
        self.paths.EVALUATION_SUMMARY_JSON_PATH = os.path.join(self.paths.STATS_CSV_DIR, "evaluation_summary.json")
        self.paths.HYPERPARAMETER_TUNING_RESULTS_CSV_PATH = os.path.join(
            self.paths.OUTPUT_DIR, "hyperparameter_tuning_results.csv"
        )
        self.paths.ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH = os.path.join(
            self.paths.OUTPUT_DIR, "advanced_hyperparameter_results.csv"
        )
        self.paths.PREPROCESSING_STATISTICS_IMAGE_PATH = os.path.join(
            self.paths.STATS_IMAGES_DIR, "preprocessing_statistics.png"
        )
        self.paths.SEQUENCE_CREATION_STATISTICS_IMAGE_PATH = os.path.join(
            self.paths.STATS_IMAGES_DIR, "sequence_creation_statistics.png"
        )
        self.paths.EVALUATION_STATISTICS_IMAGE_PATH = os.path.join(
            self.paths.STATS_IMAGES_DIR, "evaluation_statistics.png"
        )
        self.paths.TRAINING_STATISTICS_IMAGE_PATH = os.path.join(self.paths.STATS_IMAGES_DIR, "training_statistics.png")
        self.paths.PROGRESS_BAR_LOG_FILE = os.path.join(self.paths.LOGS_DIR, "progress.log")


# For backward compatibility
Config = MainConfig
