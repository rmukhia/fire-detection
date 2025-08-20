"""Data preprocessing module for forest fire detection system."""

# =============================================================================
# IMPORT STATEMENTS - ORGANIZED BY CATEGORY
# =============================================================================
import gc
import json
import multiprocessing as mp
import os
import tempfile
import sys
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import shutil

# Third-Party Scientific Computing
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Distributed Computing
import dask
import dask.diagnostics
from dask.diagnostics import ProgressBar
from dask.distributed import Client

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Local Modules
import core_logging
import data_loader

import viz_style
from core_logging import ensure_directories

class DataPreprocessor:
    """
    Advanced data preprocessor with partitioned processing and comprehensive monitoring.
    
    This class implements a sophisticated preprocessing pipeline that:
    - Creates sensor-specific temporary partitions for efficient processing
    - Integrates filtering workflows within distributed computing tasks
    - Provides comprehensive data persistence and visualization capabilities
    - Maintains professional coding standards and documentation
    """
    
    def __init__(self, config, logger: Optional[core_logging.ProcessLogger] = None):
        self.config = config
        self.logger = logger or core_logging.ProcessLogger(config, "Preprocessing")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="forestfire_preprocessing_"))
        self.intermediate_data = {}
        self.processing_metrics = {}
        ensure_directories([
            self.config.OUTPUT_DIR,
            self.config.INTERMEDIATE_DIR,
            self.config.STATS_IMAGES_DIR,
            self.config.STATS_CSV_DIR,
            self.config.PROCESSED_DATA_DIR,
            self.temp_dir,
        ])
        self.logger.log_step(f"Created temporary directory: {self.temp_dir}")
        
        # Visualization Setup
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

    def create_window_views(self, data: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
        if data.shape[0] < window_size:
            self.logger.log_step(f"Warning: Data length {data.shape[0]} < window size {window_size}")
            return np.array([]).reshape(0, window_size, data.shape[1])
        n_windows = (data.shape[0] - window_size) // step_size + 1
        shape = (n_windows, window_size, data.shape[1])
        strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)

    def create_sensor_partitions(self, df: pd.DataFrame) -> Dict[str, Path]:
        self.logger.log_step("Creating sensor-specific temporary partitions")
        sensor_partitions = {}
        partition_metadata = {}
        for sensor_id, sensor_data in tqdm(df.groupby('Sensor_Id'), desc="Creating partitions"):
            partition_path = self.temp_dir / f"sensor_{sensor_id}.parquet"
            sensor_data = sensor_data.sort_values('Datetime').reset_index(drop=True)
            sensor_data = sensor_data[~sensor_data.duplicated(keep='last')]
            sensor_data.to_parquet(partition_path, index=False, compression='snappy')
            sensor_partitions[sensor_id] = partition_path
            partition_metadata[sensor_id] = {
                'rows': len(sensor_data),
                'date_range': {
                    'start': sensor_data['Datetime'].min().isoformat(),
                    'end': sensor_data['Datetime'].max().isoformat(),
                },
                'file_size_mb': partition_path.stat().st_size / (1024 * 1024),
            }
        metadata_path = self.temp_dir / "partition_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(partition_metadata, f, indent=2, default=str)
        self.logger.log_step(f"Created {len(sensor_partitions)} sensor partitions in {self.temp_dir}")
        self.intermediate_data['sensor_partitions'] = sensor_partitions
        self.intermediate_data['partition_metadata'] = partition_metadata
        return sensor_partitions

    def process_sensor_partition_with_filtering(self, partition_path: Path, sensor_id: str, window_id_start: int) -> Dict:
        sensor_df = pd.read_parquet(partition_path)
        numeric_cols = [col for col in self.config.INPUT_COLUMNS if col in sensor_df.columns]
        for col in numeric_cols:
            sensor_df[col] = pd.to_numeric(sensor_df[col], errors='coerce')
        
        metrics = {
            'sensor_id': sensor_id,
            'original_samples': len(sensor_df),
            'processing_start': pd.Timestamp.now().isoformat(),
            'visualization_metrics': {}
        }

        sensor_df.set_index('Datetime', inplace=True)
        sensor_df = sensor_df.sort_index()
        sensor_df = sensor_df[~sensor_df.index.duplicated(keep='last')]

        if len(sensor_df) < 2:
            metrics.update({'processing_status': 'insufficient_data', 'processing_end': pd.Timestamp.now().isoformat()})
            return metrics

        start_time = sensor_df.index.min()
        end_time = sensor_df.index.max()
        resample_td = pd.to_timedelta(self.config.RESAMPLE_INTERVAL)
        tolerance = resample_td * self.config.RESAMPLE_TOLERANCE_FACTOR
        regular_index = pd.date_range(start=start_time, end=end_time, freq=self.config.RESAMPLE_INTERVAL)
        resampled_sensor = sensor_df.reindex(regular_index, method='nearest', tolerance=tolerance).dropna()

        if resampled_sensor.empty:
            metrics.update({'processing_status': 'no_data_after_resampling', 'processing_end': pd.Timestamp.now().isoformat()})
            return metrics

        resampled_sensor = resampled_sensor.reset_index().rename(columns={'index': 'Datetime'})
        sensor_df = resampled_sensor

        time_samples_per_window = self.config.WINDOW_SIZE
        step_size = self.config.WINDOW_STEP_SIZE

        if len(sensor_df) < time_samples_per_window:
            metrics.update({'processing_status': 'insufficient_data_for_windowing', 'processing_end': pd.Timestamp.now().isoformat()})
            return metrics

        feature_data = sensor_df.to_numpy()
        timestamps = sensor_df['Datetime'].to_numpy()
        windowed_features = self.create_window_views(feature_data, time_samples_per_window, step_size)
        windowed_timestamps = self.create_window_views(timestamps[:, np.newaxis], time_samples_per_window, step_size)

        if windowed_features.shape[0] == 0:
            metrics.update({'processing_status': 'no_windows_created', 'processing_end': pd.Timestamp.now().isoformat()})
            return metrics

        n_windows = windowed_features.shape[0]
        metrics['windows_created'] = n_windows

        window_ids = np.arange(window_id_start, window_id_start + n_windows)
        windowed_df = pd.DataFrame({
            'window_id': np.repeat(window_ids, time_samples_per_window),
            'sensor_id': sensor_id,
            'Datetime': windowed_timestamps.flatten(),
            'sample_index': np.tile(np.arange(time_samples_per_window), n_windows),
            **{col: windowed_features[:, :, i].flatten() for i, col in enumerate(sensor_df.columns)}
        })

        filtered_df = self._apply_duration_filtering_to_windows(windowed_df, metrics)
        saved_file_path = None
        if not filtered_df.empty:
            sensor_file = os.path.join(self.config.PROCESSED_DATA_DIR, f"sensor_{sensor_id}.parquet")
            filtered_df.to_parquet(sensor_file, index=False, compression='snappy')
            saved_file_path = str(sensor_file)
            
            window_durations = filtered_df.groupby('window_id')['Datetime'].apply(lambda x: (x.max() - x.min()).total_seconds() / 3600)
            daily_counts = filtered_df.set_index('Datetime').resample('D').size()
            metrics['visualization_metrics'] = {
                'window_durations': window_durations.tolist(),
                'daily_counts': {k.isoformat(): v for k, v in daily_counts.to_dict().items()},
                'window_durations_before_filtering': metrics.pop('window_durations_before_filtering', [])
            }

        metrics.update({
            'windows_after_filtering': filtered_df['window_id'].nunique() if not filtered_df.empty else 0,
            'samples_after_filtering': len(filtered_df),
            'filtering_efficiency': len(filtered_df) / len(windowed_df) if len(windowed_df) > 0 else 0,
            'saved_file_path': saved_file_path,
            'processing_status': 'success',
            'processing_end': pd.Timestamp.now().isoformat(),
        })
        
        del sensor_df, windowed_df, filtered_df, feature_data, timestamps, windowed_features, windowed_timestamps
        gc.collect()
        
        return metrics

    def _apply_duration_filtering_to_windows(self, windowed_df: pd.DataFrame, metrics: Dict) -> pd.DataFrame:
        if windowed_df.empty:
            return windowed_df
        window_summary = windowed_df.groupby(['window_id', 'sensor_id']).agg(
            start_time=('Datetime', 'min'),
            end_time=('Datetime', 'max')
        ).reset_index()
        window_summary['actual_duration'] = window_summary['end_time'] - window_summary['start_time']
        metrics['window_durations_before_filtering'] = (window_summary['actual_duration'].dt.total_seconds() / 3600).tolist()
        
        resample_td = pd.to_timedelta(self.config.RESAMPLE_INTERVAL)
        expected_duration = pd.to_timedelta((self.config.WINDOW_SIZE - 1) * resample_td.total_seconds(), unit='s')
        tolerance = pd.to_timedelta(self.config.DURATION_FILTER_TOLERANCE)
        
        valid_windows = window_summary[np.abs(window_summary['actual_duration'] - expected_duration) <= tolerance]
        filtered_df = windowed_df[windowed_df['window_id'].isin(valid_windows['window_id'])].copy()
        
        metrics.update({
            'duration_filtering_applied': True,
            'windows_before_duration_filter': len(window_summary),
            'windows_after_duration_filter': len(valid_windows),
        })
        return filtered_df

    def coordinate_distributed_processing(self, sensor_partitions: Dict[str, Path]) -> List[Dict]:
        self.logger.log_step("Starting coordinated distributed processing")
        sensor_metadata = []
        window_id_counter = 0
        for sensor_id, partition_path in sensor_partitions.items():
            n_samples = pd.read_parquet(partition_path, columns=['Sensor_Id']).shape[0]
            if n_samples >= self.config.WINDOW_SIZE:
                n_windows = (n_samples - self.config.WINDOW_SIZE) // getattr(self.config, 'WINDOW_STEP_SIZE', 1) + 1
                sensor_metadata.append({
                    'sensor_id': sensor_id,
                    'partition_path': partition_path,
                    'window_id_start': window_id_counter,
                })
                window_id_counter += n_windows
        
        self.logger.log_step(f"Prepared {len(sensor_metadata)} sensors for processing, expecting ~{window_id_counter} total windows")
        
        n_workers = min(mp.cpu_count(), len(sensor_metadata), self.config.NUM_WORKERS)
        with Client(n_workers=n_workers, threads_per_worker=1) as client:
            self.logger.log_step(f"Started Dask cluster with {n_workers} workers: {client.dashboard_link}")
            tasks = [dask.delayed(self.process_sensor_partition_with_filtering)(m['partition_path'], m['sensor_id'], m['window_id_start']) for m in sensor_metadata]
            
            if self.config.PROGRESS_BAR_LOG_FILE:
                with open(self.config.PROGRESS_BAR_LOG_FILE, 'w') as f:
                    with ProgressBar(out=f):
                        all_metrics = list(dask.compute(*tasks))
            else:
                with ProgressBar():
                    all_metrics = list(dask.compute(*tasks))
        
        self.logger.log_step(f"Aggregated metrics from {len(all_metrics)} partitions.")
        self.processing_metrics['sensor_processing'] = all_metrics
        self._save_processing_metrics(all_metrics)
        return all_metrics

    def _save_processing_metrics(self, all_metrics: List[Dict]) -> None:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.config.PROCESSING_METRICS_CSV_PATH, index=False)
        
        summary_metrics = {
            'total_sensors_processed': len(all_metrics),
            'successful_sensors': len([m for m in all_metrics if m['processing_status'] == 'success']),
            'total_windows_created': sum(m.get('windows_created', 0) for m in all_metrics),
            'total_windows_after_filtering': sum(m.get('windows_after_filtering', 0) for m in all_metrics),
            'average_filtering_efficiency': np.mean([m.get('filtering_efficiency', 0) for m in all_metrics if m['processing_status'] == 'success']),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
        }
        with open(self.config.PROCESSING_SUMMARY_JSON_PATH, 'w') as f:
            json.dump(summary_metrics, f, indent=2, default=str)
        self.logger.log_step(f"Processing metrics saved to {self.config.PROCESSING_METRICS_CSV_PATH} and summary to {self.config.PROCESSING_SUMMARY_JSON_PATH}")

    def _log_preprocessing_statistic(self, processing_metrics: List[Dict]):
        self.logger.log_step("Generating preprocessing statistics from metrics")
        viz_style.set_publication_style()
        all_window_durations_before = [d for m in processing_metrics if m.get('visualization_metrics') for d in m['visualization_metrics'].get('window_durations_before_filtering', [])]
        all_window_durations_after = [d for m in processing_metrics if m.get('visualization_metrics') for d in m['visualization_metrics'].get('window_durations', [])]
        all_daily_counts = pd.concat([pd.Series(m['visualization_metrics'].get('daily_counts', {})) for m in processing_metrics if m.get('visualization_metrics')]).groupby(level=0).sum()
        all_daily_counts.index = pd.to_datetime(all_daily_counts.index)

        rows_before = sum(m.get('windows_created', 0) for m in processing_metrics) * self.config.WINDOW_SIZE
        rows_after = sum(m.get('samples_after_filtering', 0) for m in processing_metrics)
        rows_removed = rows_before - rows_after
        windows_before = sum(m.get('windows_created', 0) for m in processing_metrics)
        windows_after = sum(m.get('windows_after_filtering', 0) for m in processing_metrics)
        windows_removed = windows_before - windows_after
        sensors_before = len(processing_metrics)
        sensors_after = len([m for m in processing_metrics if m.get('saved_file_path')])
        sensors_removed = sensors_before - sensors_after

        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        fig.suptitle('Forest Fire Detection - Preprocessing & Filtering Statistics', fontsize=20, fontweight='bold')

        # Row 1: Preprocessing Stats
        ax1 = axes[0, 0]
        if all_window_durations_before:
            combined_durations = all_window_durations_before + all_window_durations_after
            min_duration = min(combined_durations)
            max_duration = max(combined_durations)
            bins = np.linspace(min_duration, max_duration, 31)

            ax1.hist(all_window_durations_before, bins=bins, color=viz_style.VALID_COLORS['red'], alpha=0.6, label='Before Filtering', log=True)
            if all_window_durations_after:
                ax1.hist(all_window_durations_after, bins=bins, color=viz_style.VALID_COLORS['blue'], label='After Filtering', log=True)
            ax1.set_title('Window Duration Histogram')
            ax1.set_xlabel('Duration (hours)')
            ax1.set_ylabel('Frequency (log scale)')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No duration data', ha='center')
        
        sensor_counts = pd.Series({m['sensor_id']: m['samples_after_filtering'] for m in processing_metrics if m.get('saved_file_path')}).sort_values(ascending=False)
        top_sensors = sensor_counts.head(10)
        if len(sensor_counts) > 10:
            top_sensors['Other'] = sensor_counts[10:].sum()
        axes[0, 1].pie(top_sensors, labels=top_sensors.index, autopct='%1.1f%%', startangle=140) if not top_sensors.empty else axes[0, 1].text(0.5, 0.5, 'No sensor data', ha='center')
        axes[0, 1].set_title('Sensor Data Distribution')

        axes[0, 2].plot(all_daily_counts.index, all_daily_counts.values, color=viz_style.VALID_COLORS['brown']) if not all_daily_counts.empty else axes[0, 2].text(0.5, 0.5, 'No timeline data', ha='center')
        axes[0, 2].set_title('Data Collection Timeline')

        # Row 2: Filtering Stats
        axes[1, 0].pie([rows_after, rows_removed], labels=[f'Kept {rows_after:,}', f'Removed {rows_removed:,}'], autopct='%1.1f%%')
        axes[1, 0].set_title('Row Filtering')
        axes[1, 1].pie([windows_after, windows_removed], labels=[f'Kept {windows_after:,}', f'Removed {windows_removed:,}'], autopct='%1.1f%%')
        axes[1, 1].set_title('Window Filtering')
        axes[1, 2].pie([sensors_after, sensors_removed], labels=[f'Kept {sensors_after:,}', f'Removed {sensors_removed:,}'], autopct='%1.1f%%')
        axes[1, 2].set_title('Sensor Filtering')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.config.PREPROCESSING_STATISTICS_IMAGE_PATH, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        self.logger.log_step(f"Statistics plot saved to: {self.config.PREPROCESSING_STATISTICS_IMAGE_PATH}")


    def preprocess_data(self) -> None:
        self.logger.log_step("Starting Streamlined Data Preprocessing Pipeline")
        self.logger.log_step("Stage 1: Loading raw data and creating sensor partitions")
        data_loader_instance = data_loader.DataLoader(self.config)
        df, _, _ = data_loader_instance.load_raw_data(False, False)
        numeric_cols = [col for col in self.config.INPUT_COLUMNS if col in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        sensor_partitions = self.create_sensor_partitions(df)
        del df; gc.collect()

        self.logger.log_step("Stage 2: Distributed processing per sensor")
        processing_metrics = self.coordinate_distributed_processing(sensor_partitions)

        self.logger.log_step("Stage 3: Creating comprehensive visualization dashboard")
        self._log_preprocessing_statistic(processing_metrics)

        self.logger.log_step("Stage 4: Saving final metadata")
        sensor_files = {m['sensor_id']: m['saved_file_path'] for m in processing_metrics if m.get('saved_file_path')}
        if sensor_files:
            metadata = {
                'processing_pipeline_version': '2.1',
                'n_sensors': len(sensor_files),
                'n_windows': sum(m.get('windows_after_filtering', 0) for m in processing_metrics),
                'n_samples': sum(m.get('samples_after_filtering', 0) for m in processing_metrics),
                'sensor_files': sensor_files
            }
            with open(self.config.PROCESSING_METADATA_JSON_PATH, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.log_step(f"Comprehensive metadata saved to {self.config.PROCESSING_METADATA_JSON_PATH}")
        else:
            self.logger.log_step("Warning: No data remaining after processing")

        self.cleanup_temporary_files()
        self.logger.save_process_timeline()
        self.logger.log_step("Advanced preprocessing pipeline completed")

    def cleanup_temporary_files(self) -> None:
        shutil.rmtree(self.temp_dir)
        self.logger.log_step(f"Cleaned up temporary directory: {self.temp_dir}")

# ==========================================================================
# Main Function
# ==========================================================================
def preprocess_data(config):
    """
    Main entry point for the data preprocessing pipeline.

    Initializes and runs the DataPreprocessor.
    """
    preprocessor = DataPreprocessor(config)
    preprocessor.preprocess_data()
