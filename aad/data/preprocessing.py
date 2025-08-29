import gc
import json
import os
import tempfile


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-Party Scientific Computing
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Distributed Computing
from dask.delayed import delayed
from .dask_pipeline_base import DaskPipelineBase

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Local Modules
from aad.common import core_logging

from .utils import create_window_views

from aad.common import viz_style


def rolling_slope(y):
    if len(y) < 2:
        return np.nan
    if np.all(np.isnan(y)):
        return np.nan
    
    x = np.arange(len(y))
    
    idx = np.isfinite(x) & np.isfinite(y)
    if (len(x[idx]) < 2):
        return np.nan

    try:
        fit = np.polyfit(x[idx], y[idx], 1)
        if np.any(np.isnan(x)):
            print('hasnan')
        if np.isnan(fit[0]):
            print('nan')
        return fit[0]  # slope
    except np.linalg.LinAlgError as e:
        print(f"[rolling_slope] LinAlgError: {e} | y={y}")
        raise

class DataPreprocessor(DaskPipelineBase):
    """
    Advanced data preprocessor with partitioned processing and comprehensive monitoring.

    This class implements a sophisticated preprocessing pipeline that:
    - Creates sensor-specific temporary partitions for efficient processing
    - Integrates filtering workflows within distributed computing tasks
    - Provides comprehensive data persistence and visualization capabilities
    - Maintains professional coding standards and documentation
    """

    def __init__(
        self,
        config: Any,
        multiplier: int,
        df_sensor: Optional[pd.DataFrame] = None,
        logger: Optional[core_logging.ProcessLogger] = None,
    ) -> None:
        self.config: Any = config
        self.df_sensor: Optional[pd.DataFrame] = df_sensor
        self.logger: core_logging.ProcessLogger = logger or core_logging.ProcessLogger(config, "Preprocessing")
        self.temp_dir: Path = Path(tempfile.mkdtemp(prefix="forestfire_preprocessing_"))
        self.intermediate_data: Dict[str, Any] = {}
        self.processing_metrics: Dict[str, Any] = {}
        self.logger.log_step(f"Created temporary directory: {self.temp_dir}")
        self.multiplier = multiplier
        # Visualization Setup
        plt.style.use("seaborn-v0_8-paper")
        sns.set_palette("husl")

    def create_sensor_partitions(self, df: pd.DataFrame) -> Dict[str, Path]:
        self.logger.log_step("Creating sensor-specific temporary partitions")
        sensor_partitions = {}
        partition_metadata = {}
        for sensor_id, sensor_data in tqdm(df.groupby("Sensor_Id"), desc="Creating partitions"):
            partition_path = self.temp_dir / f"sensor_{sensor_id}.parquet"
            sensor_data = sensor_data.sort_values("Datetime").reset_index(drop=True)
            sensor_data = sensor_data[~sensor_data.duplicated(keep="last")]
            sensor_data.to_parquet(partition_path, index=False, compression="snappy")
            sensor_partitions[sensor_id] = partition_path
            partition_metadata[sensor_id] = {
                "rows": len(sensor_data),
                "date_range": {
                    "start": sensor_data["Datetime"].min().isoformat(),
                    "end": sensor_data["Datetime"].max().isoformat(),
                },
                "file_size_mb": partition_path.stat().st_size / (1024 * 1024),
            }
        metadata_path = self.temp_dir / "partition_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(partition_metadata, f, indent=2, default=str)
        self.logger.log_step(f"Created {len(sensor_partitions)} sensor partitions in {self.temp_dir}")
        self.intermediate_data["sensor_partitions"] = sensor_partitions
        self.intermediate_data["partition_metadata"] = partition_metadata
        return sensor_partitions

    def process_sensor_partition_with_filtering(
        self, partition_path: Path, sensor_id: str, window_id_start: int
    ) -> Dict[str, Any]:
        sensor_df = pd.read_parquet(partition_path)
        numeric_cols = [col for col in self.config.data_pipeline.INPUT_COLUMNS if col in sensor_df.columns]
        for col in numeric_cols:
            sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce")

        metrics = {
            "sensor_id": sensor_id,
            "original_samples": len(sensor_df),
            "processing_start": pd.Timestamp.now().isoformat(),
            "visualization_metrics": {},
        }

        sensor_df.set_index("Datetime", inplace=True)
        sensor_df = sensor_df.sort_index()
        sensor_df = sensor_df[~sensor_df.index.duplicated(keep="last")]

        if len(sensor_df) < 2:
            metrics.update(
                {
                    "processing_status": "insufficient_data",
                    "processing_end": pd.Timestamp.now().isoformat(),
                }
            )
            return metrics

        start_time = sensor_df.index.min()
        end_time = sensor_df.index.max()
        resample_td = pd.to_timedelta(self.config.data_pipeline.RESAMPLE_INTERVAL)
        tolerance = resample_td * self.config.data_pipeline.RESAMPLE_TOLERANCE_FACTOR
        regular_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq=self.config.data_pipeline.RESAMPLE_INTERVAL,
        )

        resampled_sensor = sensor_df.reindex(regular_index, method="nearest", tolerance=tolerance)

         # --- Feature Engineering: SMA, Min, Max ---
        #self.logger.log_step(f"Sensor {sensor_id}: Adding SMA, Min, Max features.")

        # Use the same numeric columns identified at the start of the method
        feature_cols_to_process = numeric_cols[:]

        # add here the engineered features
        for col in feature_cols_to_process:
            window = self.config.data_pipeline.WINDOW_SIZE * self.multiplier

            # Rolling Mean (trend)
            sma_col = f"{col}_mean_{self.multiplier}x"
            resampled_sensor[sma_col] = (
                resampled_sensor[col].rolling(window=window, min_periods=1).mean()
            )

            # Rolling window regression slope
            slope_col = f"{col}_reg_slope_{self.multiplier}x"
            resampled_sensor[slope_col] = resampled_sensor[col].rolling(window=window, min_periods=1).apply(rolling_slope, raw=True)    

        resampled_sensor = resampled_sensor.dropna()
    
        if resampled_sensor.empty:
            metrics.update(
                {
                    "processing_status": "no_data_after_resampling",
                    "processing_end": pd.Timestamp.now().isoformat(),
                }
            )
            return metrics

        resampled_sensor = resampled_sensor.reset_index().rename(columns={"index": "Datetime"})
        sensor_df = resampled_sensor

        time_samples_per_window = self.config.data_pipeline.WINDOW_SIZE
        step_size = self.config.data_pipeline.WINDOW_STEP_SIZE

        if len(sensor_df) < time_samples_per_window:
            metrics.update(
                {
                    "processing_status": "insufficient_data_for_windowing",
                    "processing_end": pd.Timestamp.now().isoformat(),
                }
            )
            return metrics

        feature_data = sensor_df.to_numpy()
        timestamps = sensor_df["Datetime"].to_numpy()
        windowed_features = create_window_views(feature_data, time_samples_per_window, step_size)
        windowed_timestamps = create_window_views(timestamps[:, np.newaxis], time_samples_per_window, step_size)

        if windowed_features.shape[0] == 0:
            metrics.update(
                {
                    "processing_status": "no_windows_created",
                    "processing_end": pd.Timestamp.now().isoformat(),
                }
            )
            return metrics

        n_windows = windowed_features.shape[0]
        metrics["windows_created"] = n_windows

        window_ids = np.arange(window_id_start, window_id_start + n_windows)
        windowed_df = pd.DataFrame(
            {
                "window_id": np.repeat(window_ids, time_samples_per_window),
                "sensor_id": sensor_id,
                "Datetime": windowed_timestamps.flatten(),
                "sample_index": np.tile(np.arange(time_samples_per_window), n_windows),
                **{col: windowed_features[:, :, i].flatten() for i, col in enumerate(sensor_df.columns)},
            }
        )

        filtered_df = self._apply_duration_filtering_to_windows(windowed_df, metrics)
        saved_file_path = None
        if not filtered_df.empty:
            sensor_file = os.path.join(self.config.paths.PROCESSED_DATA_DIR, f"sensor_{sensor_id}.parquet")
            filtered_df.to_parquet(sensor_file, index=False, compression="snappy")
            saved_file_path = str(sensor_file)

            window_durations = filtered_df.groupby("window_id")["Datetime"].apply(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
            daily_counts = filtered_df.set_index("Datetime").resample("D").size()
            metrics["visualization_metrics"] = {
                "window_durations": window_durations.tolist(),
                "daily_counts": {k.isoformat(): v for k, v in daily_counts.to_dict().items()},
                "window_durations_before_filtering": metrics.pop("window_durations_before_filtering", []),
            }

        metrics.update(
            {
                "windows_after_filtering": (filtered_df["window_id"].nunique() if not filtered_df.empty else 0),
                "samples_after_filtering": len(filtered_df),
                "filtering_efficiency": (len(filtered_df) / len(windowed_df) if len(windowed_df) > 0 else 0),
                "saved_file_path": saved_file_path,
                "processing_status": "success",
                "processing_end": pd.Timestamp.now().isoformat(),
            }
        )

        del (
            sensor_df,
            windowed_df,
            filtered_df,
            feature_data,
            timestamps,
            windowed_features,
            windowed_timestamps,
        )
        gc.collect()

        return metrics

    def _apply_duration_filtering_to_windows(self, windowed_df: pd.DataFrame, metrics: Dict[str, Any]) -> pd.DataFrame:
        if windowed_df.empty:
            return windowed_df
        window_summary = (
            windowed_df.groupby(["window_id", "sensor_id"])
            .agg(start_time=("Datetime", "min"), end_time=("Datetime", "max"))
            .reset_index()
        )
        window_summary["actual_duration"] = window_summary["end_time"] - window_summary["start_time"]
        metrics["window_durations_before_filtering"] = (
            window_summary["actual_duration"].dt.total_seconds() / 3600
        ).tolist()

        resample_td = pd.to_timedelta(self.config.data_pipeline.RESAMPLE_INTERVAL)
        expected_duration = pd.to_timedelta(
            (self.config.data_pipeline.WINDOW_SIZE - 1) * resample_td.total_seconds(),
            unit="s",
        )
        tolerance = pd.to_timedelta(self.config.data_pipeline.DURATION_FILTER_TOLERANCE)

        valid_windows = window_summary[np.abs(window_summary["actual_duration"] - expected_duration) <= tolerance]
        filtered_df = windowed_df[windowed_df["window_id"].isin(valid_windows["window_id"])].copy()

        metrics.update(
            {
                "duration_filtering_applied": True,
                "windows_before_duration_filter": len(window_summary),
                "windows_after_duration_filter": len(valid_windows),
            }
        )
        return filtered_df

    def coordinate_distributed_processing(self, sensor_partitions: Dict[str, Path], client=None) -> List[Dict[str, Any]]:
        self.logger.log_step("Starting coordinated distributed processing")
        sensor_metadata = []
        window_id_counter = 0
        for sensor_id, partition_path in sensor_partitions.items():
            n_samples = pd.read_parquet(partition_path, columns=["Sensor_Id"]).shape[0]
            if n_samples >= self.config.data_pipeline.WINDOW_SIZE:
                n_windows = (n_samples - self.config.data_pipeline.WINDOW_SIZE) // getattr(
                    self.config.data_pipeline, "WINDOW_STEP_SIZE", 1
                ) + 1
                sensor_metadata.append(
                    {
                        "sensor_id": sensor_id,
                        "partition_path": partition_path,
                        "window_id_start": window_id_counter,
                    }
                )
                window_id_counter += n_windows

        self.logger.log_step(
            f"Prepared {len(sensor_metadata)} sensors for processing, expecting ~{window_id_counter} total windows"
        )

        def aggregate_fn(results):
            self.logger.log_step(f"Aggregated metrics from {len(results)} partitions.")
            self.processing_metrics["sensor_processing"] = results
            self._save_processing_metrics(results)
            return results

        arg_tuples = [
            (m["partition_path"], m["sensor_id"], m["window_id_start"])
            for m in sensor_metadata
        ]
        return self.run_distributed_map(
            self.process_sensor_partition_with_filtering,
            arg_tuples,
            client=client,
            logger=self.logger,
            aggregate_fn=aggregate_fn,
            pure=False,
        )

    def _save_processing_metrics(self, all_metrics: List[Dict[str, Any]]) -> None:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.config.paths.PROCESSING_METRICS_CSV_PATH, index=False)

        summary_metrics = {
            "total_sensors_processed": len(all_metrics),
            "successful_sensors": len([m for m in all_metrics if m["processing_status"] == "success"]),
            "total_windows_created": sum(m.get("windows_created", 0) for m in all_metrics),
            "total_windows_after_filtering": sum(m.get("windows_after_filtering", 0) for m in all_metrics),
            "average_filtering_efficiency": np.mean(
                [m.get("filtering_efficiency", 0) for m in all_metrics if m["processing_status"] == "success"]
            ),
            "processing_timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(self.config.paths.PROCESSING_SUMMARY_JSON_PATH, "w") as f:
            json.dump(summary_metrics, f, indent=2, default=str)
        self.logger.log_step(
            f"Processing metrics saved to {self.config.paths.PROCESSING_METRICS_CSV_PATH} and summary to {self.config.paths.PROCESSING_SUMMARY_JSON_PATH}"
        )

    def _log_preprocessing_statistic(self, processing_metrics: List[Dict[str, Any]]) -> None:
        self.logger.log_step("Generating preprocessing statistics from metrics")
        viz_style.set_publication_style()
        all_window_durations_before = [
            d
            for m in processing_metrics
            if m.get("visualization_metrics")
            for d in m["visualization_metrics"].get("window_durations_before_filtering", [])
        ]
        all_window_durations_after = [
            d
            for m in processing_metrics
            if m.get("visualization_metrics")
            for d in m["visualization_metrics"].get("window_durations", [])
        ]
        all_daily_counts = (
            pd.concat(
                [
                    pd.Series(m["visualization_metrics"].get("daily_counts", {}))
                    for m in processing_metrics
                    if m.get("visualization_metrics")
                ]
            )
            .groupby(level=0)
            .sum()
        )
        all_daily_counts.index = pd.to_datetime(all_daily_counts.index)

        rows_before = (
            sum(m.get("windows_created", 0) for m in processing_metrics) * self.config.data_pipeline.WINDOW_SIZE
        )
        rows_after = sum(m.get("samples_after_filtering", 0) for m in processing_metrics)
        rows_removed = rows_before - rows_after
        windows_before = sum(m.get("windows_created", 0) for m in processing_metrics)
        windows_after = sum(m.get("windows_after_filtering", 0) for m in processing_metrics)
        windows_removed = windows_before - windows_after
        sensors_before = len(processing_metrics)
        sensors_after = len([m for m in processing_metrics if m.get("saved_file_path")])
        sensors_removed = sensors_before - sensors_after

        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        fig.suptitle(
            "Forest Fire Detection - Preprocessing & Filtering Statistics",
            fontsize=20,
            fontweight="bold",
        )

        # Row 1: Preprocessing Stats
        ax1 = axes[0, 0]
        if all_window_durations_before:
            combined_durations = all_window_durations_before + all_window_durations_after
            min_duration = min(combined_durations)
            max_duration = max(combined_durations)
            bins = np.linspace(min_duration, max_duration, 31)

            ax1.hist(
                all_window_durations_before,
                bins=bins,
                color=viz_style.VALID_COLORS["red"],
                alpha=0.6,
                label="Before Filtering",
                log=True,
            )
            if all_window_durations_after:
                ax1.hist(
                    all_window_durations_after,
                    bins=bins,
                    color=viz_style.VALID_COLORS["blue"],
                    label="After Filtering",
                    log=True,
                )
            ax1.set_title("Window Duration Histogram")
            ax1.set_xlabel("Duration (hours)")
            ax1.set_ylabel("Frequency (log scale)")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No duration data", ha="center")

        sensor_counts = pd.Series(
            {m["sensor_id"]: m["samples_after_filtering"] for m in processing_metrics if m.get("saved_file_path")}
        ).sort_values(ascending=False)
        top_sensors = sensor_counts.head(10)
        if len(sensor_counts) > 10:
            top_sensors["Other"] = sensor_counts[10:].sum()
        (
            axes[0, 1].pie(top_sensors, labels=top_sensors.index, autopct="%1.1f%%", startangle=140)
            if not top_sensors.empty
            else axes[0, 1].text(0.5, 0.5, "No sensor data", ha="center")
        )
        axes[0, 1].set_title("Sensor Data Distribution")

        (
            axes[0, 2].plot(
                all_daily_counts.index,
                all_daily_counts.values,
                color=viz_style.VALID_COLORS["brown"],
            )
            if not all_daily_counts.empty
            else axes[0, 2].text(0.5, 0.5, "No timeline data", ha="center")
        )
        axes[0, 2].set_title("Data Collection Timeline")

        # Row 2: Filtering Stats
        axes[1, 0].pie(
            [rows_after, rows_removed],
            labels=[f"Kept {rows_after:,}", f"Removed {rows_removed:,}"],
            autopct="%1.1f%%",
        )
        axes[1, 0].set_title("Row Filtering")
        axes[1, 1].pie(
            [windows_after, windows_removed],
            labels=[f"Kept {windows_after:,}", f"Removed {windows_removed:,}"],
            autopct="%1.1f%%",
        )
        axes[1, 1].set_title("Window Filtering")
        axes[1, 2].pie(
            [sensors_after, sensors_removed],
            labels=[f"Kept {sensors_after:,}", f"Removed {sensors_removed:,}"],
            autopct="%1.1f%%",
        )
        axes[1, 2].set_title("Sensor Filtering")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(
            self.config.paths.PREPROCESSING_STATISTICS_IMAGE_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
        self.logger.log_step(f"Statistics plot saved to: {self.config.paths.PREPROCESSING_STATISTICS_IMAGE_PATH}")

    def preprocess_data(self, client=None) -> None:
        self.logger.log_step("Starting Streamlined Data Preprocessing Pipeline")
        self.logger.log_step("Stage 1: Using provided raw data and creating sensor partitions")
        if self.df_sensor is None:
            raise ValueError("df_sensor must be provided to DataPreprocessor.")
        df = self.df_sensor

        numeric_cols = [col for col in self.config.data_pipeline.INPUT_COLUMNS if col in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        sensor_partitions = self.create_sensor_partitions(df)
        del df
        gc.collect()

        self.logger.log_step("Stage 2: Distributed processing per sensor")
        processing_metrics = self.coordinate_distributed_processing(sensor_partitions, client=client)

        self.logger.log_step("Stage 3: Creating comprehensive visualization dashboard")
        self._log_preprocessing_statistic(processing_metrics)

        self.logger.log_step("Stage 4: Saving final metadata")
        sensor_files = {m["sensor_id"]: m["saved_file_path"] for m in processing_metrics if m.get("saved_file_path")}
        if sensor_files:
            metadata = {
                "processing_pipeline_version": "2.1",
                "n_sensors": len(sensor_files),
                "n_windows": sum(m.get("windows_after_filtering", 0) for m in processing_metrics),
                "n_samples": sum(m.get("samples_after_filtering", 0) for m in processing_metrics),
                "sensor_files": sensor_files,
            }
            with open(self.config.paths.PROCESSING_METADATA_JSON_PATH, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.log_step(f"Comprehensive metadata saved to {self.config.paths.PROCESSING_METADATA_JSON_PATH}")
        else:
            self.logger.log_step("Warning: No data remaining after processing")

        self.logger.save_process_timeline()
        self.logger.log_step("Advanced preprocessing pipeline completed")

    # batch_partition_generator removed (generator code not supported)
