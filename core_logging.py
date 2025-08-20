"""Core logging infrastructure for the forest fire detection system."""
import logging
import os
import sys
from datetime import datetime
import os

def ensure_directories(paths):
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

# =========================================================================
# Optional plotting dependencies
# =========================================================================
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

# =========================================================================
# Logging Setup
# =========================================================================
def setup_logging(config):
    """Set up logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('forest_fire_detection')
    # logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    log_filename = datetime.now().strftime(config.LOG_FILENAME_PATTERN)
    log_filepath = os.path.join(config.LOGS_DIR, log_filename)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    #console_handler.setLevel(log_level)
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
    
    def __init__(self, config, process_name):
        self.config = config
        self.process_name = process_name
        self.logger = logging.getLogger(f'forest_fire_detection.{process_name.lower()}')
        self.start_time = datetime.now()
        self.steps = []
        self.metrics = {}
        
        # Ensure logger is configured
        if not self.logger.handlers:
            setup_logging(config)
    
    def log_step(self, description: str, metrics: dict = None) -> None:
        """Log a workflow step with optional metrics.
        
        Args:
            description: Description of the workflow step
            metrics: Optional dictionary of metrics
        """
        step_time = datetime.now()
        self.logger.info(f"[{self.process_name}] {description}")
        
        step_data = {
            'timestamp': step_time,
            'description': description,
            'metrics': metrics or {}
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
    
    def log_data_summary(self, df: 'pd.DataFrame', description: str = "Data") -> None:
        """Log a summary of dataframe contents.
        
        Args:
            df: DataFrame to summarize
            description: Contextual description of the data
        """
        self.logger.info(
            f"[{self.process_name}] {description}: {len(df)} rows, "
            f"{len(df.columns)} columns"
        )
        if hasattr(df, 'memory_usage'):
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            self.logger.info(f"[{self.process_name}] Memory usage: {memory_mb:.2f} MB")
    
    def save_process_timeline(self) -> None:
        """Save interactive timeline visualization of process steps.
        
        Note: Requires plotly to be installed
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available - skipping process timeline visualization")
            return
            
        os.makedirs(self.config.STATS_HTML_DIR, exist_ok=True)
        timeline_path = os.path.join(
            self.config.STATS_HTML_DIR, f'{self.process_name.lower()}_timeline.html'
        )
        
        if not self.steps:
            return
        
        # Create timeline visualization
        fig = go.Figure()
        
        timestamps = [step['timestamp'] for step in self.steps]
        descriptions = [step['description'] for step in self.steps]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=list(range(len(timestamps))),
            mode='markers+lines',
            text=descriptions,
            name='Process Steps'
        ))
        
        fig.update_layout(
            title=f'{self.process_name} Process Timeline',
            xaxis_title='Time',
            yaxis_title='Step Number',
            template='plotly_white'
        )
        
        fig.write_html(timeline_path)
        self.logger.info(f"Process timeline saved to {timeline_path}")
    
    def save_metrics_plot(self) -> None:
        """Save interactive bar plot of collected metrics.
        
        Note: Requires plotly to be installed
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available - skipping metrics visualization")
            return
            
        if not self.metrics:
            return
            
        os.makedirs(self.config.STATS_HTML_DIR, exist_ok=True)
        metrics_path = os.path.join(
            self.config.STATS_HTML_DIR, f'{self.process_name.lower()}_metrics.html'
        )
        
        # Create metrics bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=list(self.metrics.keys()),
                y=list(self.metrics.values())
            )
        ])
        
        fig.update_layout(
            title=f'{self.process_name} Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white'
        )
        
        fig.write_html(metrics_path)
        self.logger.info(f"Metrics plot saved to {metrics_path}")
