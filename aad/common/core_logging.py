"""Core logging infrastructure for the forest fire detection system."""

import logging
import os
import sys
from datetime import datetime
import pandas as pd
from typing import Optional, List, Dict, Any


def ensure_directories(paths: List[str]) -> None:
    """Create directories if they don't exist.

    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


# =========================================================================
# Plotting dependencies
# =========================================================================
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================================
# Logging Setup
# =========================================================================
def setup_logging(config: Any) -> logging.Logger:
    """Set up logging configuration.

    Args:
        config: Configuration object

    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    os.makedirs(config.paths.LOGS_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger("forest_fire_detection")
    # logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler
    log_filename = datetime.now().strftime(config.training.LOG_FILENAME_PATTERN)
    log_filepath = os.path.join(config.paths.LOGS_DIR, log_filename)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =========================================================================
# Process Logger Class
# =========================================================================
class ProcessLogger:
    """Enhanced process logger for tracking workflow steps."""

    def ensure_file_dir(self, file_path: str) -> None:
        """Ensure the parent directory for a file exists."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def __init__(self, config: Any, process_name: str) -> None:
        self.config: Any = config
        self.process_name: str = process_name
        self.logger: logging.Logger = logging.getLogger(f"forest_fire_detection.{process_name.lower()}")
        self.start_time: datetime = datetime.now()
        self.steps: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

        # Ensure logger is configured
        if not self.logger.handlers:
            setup_logging(config)

    def log_step(self, description: str, metrics: Optional[dict] = None) -> None:
        """Log a workflow step with optional metrics.

        Args:
            description: Description of the workflow step
            metrics: Optional dictionary of metrics
        """
        step_time = datetime.now()
        self.logger.info(f"[{self.process_name}] {description}")

        step_data = {
            "timestamp": step_time,
            "description": description,
            "metrics": metrics or {},
        }
        self.steps.append(step_data)

        if metrics:
            self.metrics.update(metrics)

    def log_info(self, message: str) -> None:
        """Log an informational message.

        Args:
            message: Message to log
        """
        self.logger.info(f"[{self.process_name}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Error message to log
        """
        self.logger.error(f"[{self.process_name}] {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Warning message to log
        """
        self.logger.warning(f"[{self.process_name}] {message}")

    def log_data_summary(self, df: pd.DataFrame, description: str = "Data") -> None:
        """Log a summary of dataframe contents.

        Args:
            df: DataFrame to summarize
            description: Contextual description of the data
        """
        self.logger.info(f"[{self.process_name}] {description}: {len(df)} rows, " f"{len(df.columns)} columns")
        if hasattr(df, "memory_usage"):
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            self.logger.info(f"[{self.process_name}] Memory usage: {memory_mb:.2f} MB")

    def save_process_timeline(self) -> None:
        """Save interactive timeline visualization of process steps.

        Note: Requires matplotlib to be installed
        """

        os.makedirs(self.config.paths.STATS_IMAGES_DIR, exist_ok=True)
        timeline_path = os.path.join(
            self.config.paths.STATS_IMAGES_DIR,
            f"{self.process_name.lower()}_timeline.png",
        )

        if not self.steps:
            return

        # Create timeline visualization with matplotlib
        timestamps = [step["timestamp"] for step in self.steps]
        step_numbers = list(range(len(timestamps)))
        descriptions = [step["description"] for step in self.steps]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, step_numbers, "o-", markersize=8)

        # Add annotations for each step
        for i, (timestamp, step_num, description) in enumerate(zip(timestamps, step_numbers, descriptions)):
            plt.annotate(
                f"Step {i+1}: {description}",
                (timestamp, step_num),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        plt.title(f"{self.process_name} Process Timeline")
        plt.xlabel("Time")
        plt.ylabel("Step Number")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(timeline_path, dpi=300, bbox_inches="tight")
        #plt.show()
        self.logger.info(f"Process timeline saved to {timeline_path}")

    def save_metrics_plot(self) -> None:
        """Save interactive bar plot of collected metrics.

        Note: Requires matplotlib to be installed
        """
        if not self.metrics:
            return

        # Filter only numeric metrics
        numeric_metrics = {k: v for k, v in self.metrics.items() if isinstance(v, (int, float))}
        if not numeric_metrics:
            self.logger.warning(f"No numeric metrics to plot in {self.process_name}. Skipping metrics plot.")
            return

        os.makedirs(self.config.paths.STATS_IMAGES_DIR, exist_ok=True)
        metrics_path = os.path.join(
            self.config.paths.STATS_IMAGES_DIR,
            f"{self.process_name.lower()}_metrics.png",
        )

        # Create metrics bar plot with seaborn/matplotlib
        plt.figure(figsize=(10, 6))
        keys = list(numeric_metrics.keys())
        values = list(numeric_metrics.values())

        # Debug: print metrics being plotted
        print(f"[DEBUG] Plotting metrics: {numeric_metrics}")

        # Use seaborn for enhanced styling
        sns.barplot(x=keys, y=values)
        sns.despine()

        plt.title(f"{self.process_name} Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
        #plt.show()
        self.logger.info(f"Metrics plot saved to {metrics_path}")
