import os
import torch
from core_logging import ensure_directories

class Config:
    """
    Central configuration class for forest fire detection system.
    
    Contains all paths, parameters and hyperparameters as class attributes.
    Automatically creates required directories on initialization.
    """
    # =========================================================================
    # Base Directories
    # =========================================================================
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

    # =========================================================================
    # Subdirectories
    # =========================================================================
    STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
    STATS_IMAGES_DIR = os.path.join(STATS_DIR, "images")
    STATS_CSV_DIR = os.path.join(STATS_DIR, "csv")
    STATS_HTML_DIR = os.path.join(STATS_DIR, "html")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    ANNOTATED_DATA_DIR = os.path.join(OUTPUT_DIR, "annotated_data")
    PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
    INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediate")

    # =========================================================================
    # Raw Data Paths
    # =========================================================================
    RAW_DATA_PATH = os.path.join(DATA_DIR, "sensor_data.csv")
    LOCATION_DATA_PATH = os.path.join(DATA_DIR, "dim_location.csv")

    # =========================================================================
    # Processed Data Paths
    # =========================================================================
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "consolidated_sensor_data.parquet")
    LABEL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "label.csv")
    DISTANCE_ANNOTATED_PATH = os.path.join(ANNOTATED_DATA_DIR, "sensor_data_window_fire_distance.parquet")
    ANNOTATED_DATA_PATH = os.path.join(ANNOTATED_DATA_DIR, "annotated_data.parquet")
    DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "dataset.pt")
    
    # =========================================================================
    # Ground Truth Config
    # =========================================================================
    LOCAL_OFFSET_MINUTES = 420  # UTC+7h = 420 minutes



    # =========================================================================
    # Model File Paths
    # =========================================================================
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

    # =========================================================================
    # Log and Output File Paths
    # =========================================================================
    LOG_FILENAME_PATTERN = "%Y-%m-%d_%H-%M-%S.log"  # Format for log filenames
    PROGRESS_BAR_LOG_FILE = os.path.join(LOGS_DIR, "progress.log")
    LOSS_HISTORY_PATH = os.path.join(STATS_CSV_DIR, "loss_history.json")
    PROCESSING_METRICS_CSV_PATH = os.path.join(STATS_CSV_DIR, "processing_metrics.csv")
    PROCESSING_METADATA_JSON_PATH = os.path.join(STATS_CSV_DIR, "processing_metadata.json")
    PROCESSING_SUMMARY_JSON_PATH = os.path.join(STATS_CSV_DIR, "processing_summary.json")
    EVALUATION_RESULTS_CSV_PATH = os.path.join(STATS_CSV_DIR, "evaluation_results.csv")
    EVALUATION_SUMMARY_JSON_PATH = os.path.join(STATS_CSV_DIR, "evaluation_summary.json")
    HYPERPARAMETER_TUNING_RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "hyperparameter_tuning_results.csv")
    ADVANCED_HYPERPARAMETER_TUNING_RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "advanced_hyperparameter_results.csv")
    
    # =========================================================================
    # Visualization Paths
    # =========================================================================
    PREPROCESSING_STATISTICS_IMAGE_PATH = os.path.join(STATS_IMAGES_DIR, "preprocessing_statistics.png")
    SEQUENCE_CREATION_STATISTICS_IMAGE_PATH = os.path.join(STATS_IMAGES_DIR, "sequence_creation_statistics.png")
    EVALUATION_STATISTICS_IMAGE_PATH = os.path.join(STATS_IMAGES_DIR, "evaluation_statistics.png")
    TRAINING_STATISTICS_IMAGE_PATH = os.path.join(STATS_IMAGES_DIR, "training_statistics.png")

    # =========================================================================
    # Data Loading Parameters
    # =========================================================================
    S_NROWS = 500000
    F_NROWS = None

    # =========================================================================
    # Preprocessing Parameters
    # =========================================================================
    RESAMPLE_INTERVAL = '2min'
    WINDOW_DURATION_MINUTES = 120  # 2 hours
    SAMPLING_INTERVAL_SECONDS = 120  # 2 minutes
    INPUT_COLUMNS = [
        'PM2.5',
        'Carbon dioxide (CO2)',
        'Relative humidity'
    ]

    # =========================================================================
    # Model Parameters
    # =========================================================================
    AUTOENCODER_CLASS = "DenseAutoencoder"  # Default autoencoder class
    LATENT_DIM = 32
    HIDDEN_DIM = 180

    # =========================================================================
    # Training Parameters
    # =========================================================================
    EPOCHS = int(os.environ.get("EPOCHS", 50))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
    RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
    TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", 0.7))
    VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.15))
    PATIENCE = int(os.environ.get("PATIENCE", 25))
    LOSS_FUNCTION = os.environ.get("LOSS_FUNCTION", "MSELoss")  # Loss function for model training
    OPTIMIZER = os.environ.get("OPTIMIZER", "Adam")             # Optimizer for model training

    # =========================================================================
    # Parallel Processing
    # =========================================================================
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 3))
    RESAMPLE_TOLERANCE_FACTOR = float(os.environ.get("RESAMPLE_TOLERANCE_FACTOR", 0.5))   # Factor of resample interval for tolerance
    WINDOW_STEP_SIZE = int(os.environ.get("WINDOW_STEP_SIZE", 1))              # Step size for moving window creation
    DURATION_FILTER_TOLERANCE = os.environ.get("DURATION_FILTER_TOLERANCE", "5s")  # Tolerance for window duration filtering

    # =========================================================================
    # Evaluation Parameters
    # =========================================================================
    DISTANCE_FILTER_THRESHOLD_M = 5000  # 5 km
    ANOMALY_THRESHOLD_PERCENTILE = 99.95
    EPSG_ZONE = "EPSG:32647"

    # =========================================================================
    # Clustering Parameters
    # =========================================================================
    KMEANS_N_CLUSTERS = 32
    DBSCAN_EPS = 0.3
    DBSCAN_MIN_SAMPLES = 5
    DIM_REDUCTION_METHOD = 'tsne'

    # =========================================================================
    # Hyperparameter Tuning
    # =========================================================================
    HYPERPARAMETER_GRID = {
        'latent_dims': [18, 36],
        'hidden_dims': [180, 360],
        'anomaly_threshold_percentiles': [99.9, 99.98]
    }
    ADVANCED_HYPERPARAMETER_GRID = {
        'latent_dims': [18, 36, 54],
        'hidden_dims': [180, 360, 540],
        'learning_rates': [1e-3, 5e-4, 1e-4],
        'batch_sizes': [16, 32, 64],
        'anomaly_thresholds': [99.9, 99.95, 99.98]
    }

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def WINDOW_SIZE(self) -> int:
        """
        Computes window size in samples based on duration and sampling interval.
        
        Returns:
            int: Number of samples per window
        """
        return (
            self.WINDOW_DURATION_MINUTES * 60 // self.SAMPLING_INTERVAL_SECONDS
        )

    @property
    def INPUT_DIM(self) -> int:
        """
        Computes total input dimension for the model.
        
        Returns:
            int: Total number of input features (columns * window_size)
        """
        return len(self.INPUT_COLUMNS) * self.WINDOW_SIZE

    @property
    def TEST_SPLIT(self) -> float:
        """
        Computes test split ratio from train and validation splits.
        
        Returns:
            float: Test data proportion (1 - train_split - val_split)
        """
        return 1 - self.TRAIN_SPLIT - self.VAL_SPLIT

    @property
    def DEVICE(self) -> str:
        """
        Determines available compute device (CUDA GPU or CPU).
        
        Returns:
            str: 'cuda' if GPU available, else 'cpu'
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Initialization
    # =========================================================================
    def __init__(self):
        """
        Initialize configuration and create required directories.
        
        Creates all directories specified in the directories list.
        Uses core_logging.ensure_directories() for creation.
        """
        directories = [
            self.DATA_DIR, self.MODEL_DIR, self.OUTPUT_DIR,
            self.STATS_DIR, self.STATS_IMAGES_DIR, self.STATS_CSV_DIR,
            self.STATS_HTML_DIR, self.LOGS_DIR, self.ANNOTATED_DATA_DIR,
            self.PROCESSED_DATA_DIR, self.INTERMEDIATE_DIR
        ]
        ensure_directories(directories)
