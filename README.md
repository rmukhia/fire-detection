# Forest Fire Detection System

A machine learning system for detecting forest fires using sensor data and geospatial information. The system uses Variational Autoencoders (VAE) for anomaly detection to identify potential fire events based on environmental sensor readings.

## Features

- **Anomaly Detection**: Variational Autoencoder-based model for fire detection
- **Geospatial Processing**: Integration with fire hotspot data and sensor locations
- **Distributed Computing**: Dask-powered data processing pipeline
- **Hyperparameter Tuning**: Automated model optimization
- **Configuration-Driven**: Flexible YAML-based configuration system

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large datasets
- 10GB+ disk space for data and models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rmukhia/fire-detection.git
cd fire-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For development, install additional dependencies:
```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Using Jupyter Notebook (Recommended for exploration)
```bash
jupyter notebook ForestFireDetection.ipynb
```

### Using Python Scripts
```bash
# Run the preprocessing pipeline
python run_preprocessing_pipeline.py

# Run model training
python run_training.py

# Run hyperparameter tuning
python run_tuning.py
```

## Configuration

The system is configured via `config.yaml`. Key sections:

- `paths`: Data directories and file paths
- `data_pipeline`: Data processing parameters
- `training`: Model training configuration
- `tuning`: Hyperparameter optimization settings

### Environment Variables

Optional environment variables for customization:
- `DATA_DIR`: Override data directory path
- `OUTPUT_DIR`: Override output directory path
- `MODEL_DIR`: Override model directory path

## Data Format

### Required Data Files

1. **Sensor Data** (`data/sensor_data.csv`):
   - Columns: `Datetime`, `Sensor_Id`, `PM2.5`, `Carbon dioxide (CO2)`, `Relative humidity`
   - Time series data from environmental sensors

2. **Location Data** (`data/dim_location.csv`):
   - Columns: `Sensor_Id`, `GPS_Lat`, `GPS_Lon`
   - Geographic coordinates for each sensor

3. **Fire Labels** (optional, `datasets/label.csv`):
   - Fire event data for supervised training
   - Format: start_time, end_time, geometry (WKT format)

## Architecture

```
aad/
├── common/          # Core utilities and configuration
├── data/           # Data loading and preprocessing
├── autoencoder/    # VAE model implementation
└── model_*.py      # Training and evaluation scripts

tests/              # Unit tests
config.yaml         # System configuration
```

## Model Details

### Variational Autoencoder
- **Purpose**: Anomaly detection for fire events
- **Input**: Time-windowed sensor data (PM2.5, CO2, humidity)
- **Output**: Reconstruction error (anomaly score)
- **Architecture**: Convolutional encoder-decoder with attention

### Data Processing Pipeline
1. **Preprocessing**: Data cleaning, resampling, normalization
2. **Windowing**: Create time windows for sequence modeling
3. **Annotation**: Label data based on proximity to fire events
4. **Sequencing**: Generate training sequences for model input

## Performance Optimization

### Distributed Computing
- Uses Dask for parallel data processing
- Configurable worker count via `NUM_WORKERS`
- Optimized memory usage for large datasets

### GPU Acceleration
- PyTorch CUDA support for model training
- Mixed precision training available
- Device selection via configuration

### Memory Management
- Parquet format for efficient data storage
- Chunked processing for large datasets
- Configurable batch sizes

## Monitoring and Logging

- Comprehensive logging via `core_logging.py`
- Training progress visualization
- Model performance metrics tracking
- Error handling and recovery

## Contributing

1. Follow PEP 8 style guidelines
2. Add unit tests for new features
3. Update documentation for API changes
4. Use type hints for better code clarity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration
2. **Dask Worker Errors**: Adjust worker count for your system
3. **Data Loading Issues**: Check file paths in configuration

### Performance Tips

- Use SSD storage for data files
- Increase `NUM_WORKERS` for faster preprocessing
- Enable mixed precision training for GPU speedup
- Use appropriate `RESAMPLE_INTERVAL` for your data frequency

## References

- [FIRMS VIIRS Firehotspots](https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_VIIRS_Firehotspots.html)
- [Variational Autoencoders for Anomaly Detection](https://arxiv.org/abs/1312.6114)