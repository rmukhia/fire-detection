from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import seaborn as sns
from aad.common import core_logging

import os
from aad.common import viz_style

from dask.delayed import delayed
from dask.distributed import Future  # Added this import
from .dask_pipeline_base import DaskPipelineBase
import multiprocessing as mp
from pathlib import Path

import gc
import shapely
from aad.common.config import Config

# =========================================================================
# Visualization Setup
# =========================================================================
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")


# =========================================================================
# Data Annotator Class
# =========================================================================
class DataAnnotator(DaskPipelineBase):
    """Class for annotating sensor data with fire proximity information."""

    def __init__(
        self,
        config: Config,
        df_labels: Optional[gpd.GeoDataFrame] = None,
        df_locations: Optional[gpd.GeoDataFrame] = None,
        logger: Optional[core_logging.ProcessLogger] = None,
    ) -> None:
        self.config: Config = config
        self.df_labels: Optional[gpd.GeoDataFrame] = df_labels
        self.df_locations: Optional[gpd.GeoDataFrame] = df_locations
        self.logger: core_logging.ProcessLogger = logger or core_logging.ProcessLogger(config, "Annotation")
        self.annotation_metrics: Dict[str, Any] = {}

    def _log_annotation_statistics(self, metrics: Dict[str, Any], df_labels: pd.DataFrame, config: Config) -> None:
        """
        Generate and save annotation statistics visualizations from collected metrics.

        Args:
            metrics: Aggregated metrics dictionary
            df_labels: Fire labels DataFrame
            config: Configuration object
        """
        self.logger.log_step("Generating annotation statistics from collected metrics")
        viz_style.set_publication_style()

        total_windows: int = metrics["total_windows"]
        annotated_windows: int = metrics["annotated_windows"]
        unannotated_windows: int = metrics["unannotated_windows"]
        annotation_rate: float = (annotated_windows / total_windows) * 100 if total_windows > 0 else 0.0

        unique_fires: int = len(metrics["fire_assignments"])
        fire_counts: pd.Series = pd.Series(metrics["fire_assignments"]).sort_index()
        finite_distances: List[float] = [d for d in metrics["distances"] if d != float("inf")]

        fig: Figure
        axes: np.ndarray
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Forest Fire Detection - Annotation Statistics",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Annotation Success Rate
        ax1: Axes = axes[0, 0]
        sizes: List[int] = [annotated_windows, unannotated_windows]
        labels: List[str] = ["Annotated", "Unannotated"]
        colors: List[str] = [
            viz_style.VALID_COLORS["green"],
            viz_style.VALID_COLORS["red"],
        ]
        ax1.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=lambda p: f"{p:.1f}% ({int(p*total_windows/100):d})",
            startangle=90,
        )
        ax1.set_title("Annotation Success Rate", fontweight="bold")

        # 2. Fire ID Assignment Histogram
        ax2: Axes = axes[0, 1]
        if len(fire_counts) > 0:
            ax2.bar(
                range(len(fire_counts)),
                fire_counts.to_numpy(),
                color=viz_style.VALID_COLORS["blue"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.8,
            )
            ax2.set_xlabel("Fire ID (sorted)", fontweight="bold")
            ax2.set_ylabel("Number of Windows", fontweight="bold")
            ax2.set_title(
                f"Fire Assignment Distribution\n({unique_fires} unique fires)",
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3, linewidth=0.6)
        else:
            ax2.text(
                0.5,
                0.5,
                "No Fire Assignments",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="red",
            )
            ax2.set_title("Fire Assignment Distribution", fontweight="bold")

        # 3. Distance Distribution
        ax3: Axes = axes[1, 0]
        if len(finite_distances) > 0:
            distances_km: List[float] = [d / 1000 for d in finite_distances]
            ax3.hist(
                distances_km,
                bins=min(30, len(distances_km) // 10 + 1),
                color=viz_style.VALID_COLORS["orange"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.8,
            )
            ax3.set_xlabel("Distance to Fire (km)", fontweight="bold")
            ax3.set_ylabel("Number of Windows", fontweight="bold")
            ax3.set_title("Distance Distribution", fontweight="bold")
            ax3.grid(True, alpha=0.3, linewidth=0.6)
            stats_text: str = f"Mean: {np.mean(distances_km):.2f}km\nMedian: {np.median(distances_km):.2f}km"
            ax3.text(
                0.98,
                0.98,
                stats_text,
                transform=ax3.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        else:
            ax3.text(
                0.5,
                0.5,
                "No Distance Data",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="red",
            )
            ax3.set_title("Distance Distribution", fontweight="bold")

        plt.tight_layout()
        plot_path: str = os.path.join(
            self.config.paths.STATS_IMAGES_DIR,
            os.path.basename(self.config.paths.DISTANCE_ANNOTATED_PATH).replace(".parquet", "_statistics.png"),
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        if annotated_windows > 0:
            stats_data: Dict[str, List[Any]] = {
                "Metric": [
                    "Total Windows",
                    "Annotated Windows",
                    "Annotation Rate (%)",
                    "Unique Fires",
                    "Avg Distance (m)",
                    "Median Distance (m)",
                    "Avg Windows per Fire",
                ],
                "Value": [
                    total_windows,
                    annotated_windows,
                    f"{annotation_rate:.2f}",
                    unique_fires,
                    f"{np.mean(finite_distances):.1f}" if finite_distances else "N/A",
                    f"{np.median(finite_distances):.1f}" if finite_distances else "N/A",
                    f"{np.mean(list(metrics['fire_assignments'].values())):.2f}",
                ],
            }
            stats_df: pd.DataFrame = pd.DataFrame(stats_data)
            stats_csv_path: str = os.path.join(
                self.config.paths.STATS_CSV_DIR,
                os.path.basename(self.config.paths.DISTANCE_ANNOTATED_PATH).replace(".parquet", "_statistics.csv"),
            )
            stats_df.to_csv(stats_csv_path, index=False)

            fire_assignment_df: pd.DataFrame = pd.DataFrame(
                {
                    "fire_id": list(metrics["fire_assignments"].keys()),
                    "window_count": list(metrics["fire_assignments"].values()),
                }
            )
            fire_assignment_df["percentage"] = (fire_assignment_df["window_count"] / annotated_windows * 100).round(2)
            fire_assign_path: str = os.path.join(
                self.config.paths.STATS_CSV_DIR,
                os.path.basename(self.config.paths.DISTANCE_ANNOTATED_PATH).replace(
                    ".parquet", "_fire_assignments.csv"
                ),
            )
            fire_assignment_df.to_csv(fire_assign_path, index=False)

        self.logger.log_step(f"Statistics saved to: {plot_path}")

    def process_sensor_annotation(
        self,
        sensor_file_path: Path,
        df_labels_proj: gpd.GeoDataFrame,
        sensor_location_geometry: shapely.geometry.Point,
    ) -> Dict[str, Any]:
        """
        Annotate sensor data with nearest fire information and collect metrics.

        Args:
            sensor_file_path: Path to sensor data file
            df_labels_proj: GeoDataFrame of fire labels projected to the correct CRS
            sensor_location_geometry: Shapely Point geometry of the sensor location

        Returns:
            dict: Metrics dictionary with collected statistics
        """
        df_window: pd.DataFrame = pd.read_parquet(sensor_file_path)
        sensor_id: str = df_window["sensor_id"].iloc[0]

        # Support direct GeoDataFrames and scattered Futures for df_labels
        # If df_labels_proj is a Dask Future, compute its result
        if isinstance(df_labels_proj, Future):
            df_labels_proj = df_labels_proj.result()

        # Create window summary
        window_summary: pd.DataFrame = (
            df_window.groupby("window_id", sort=False)
            .agg(
                window_start_time=("Datetime", "min"),
                window_end_time=("Datetime", "max"),
                sensor_id=("sensor_id", "first"),
            )
            .reset_index()
        )

        projected_crs: str = self.config.data_pipeline.EPSG_ZONE

        # Assign sensor location geometry directly
        # Convert single shapely Point to GeoSeries for to_crs method
        sensor_geometry_series: gpd.GeoSeries = gpd.GeoSeries([sensor_location_geometry], crs=projected_crs)
        projected_sensor_geometry: shapely.geometry.Point = sensor_geometry_series.iloc[0]  # type: ignore

        # Create geometry list with explicit type handling
        geometry_list: List[shapely.geometry.Point] = [projected_sensor_geometry] * len(window_summary)
        window_geo: gpd.GeoDataFrame = gpd.GeoDataFrame(
            window_summary, geometry=geometry_list, crs=projected_crs  # type: ignore
        )

        # Filter relevant fires
        window_time_max: pd.Timestamp = df_window["Datetime"].max()
        window_time_min: pd.Timestamp = df_window["Datetime"].min()
        fire_mask: pd.Series = (df_labels_proj["start_time"] <= window_time_max) & (
            df_labels_proj["end_time"] >= window_time_min
        )
        df_labels_filtered: gpd.GeoDataFrame = df_labels_proj[fire_mask].copy()

        # Initialize metrics collection
        metrics: Dict[str, Any] = {
            "sensor_id": sensor_id,
            "total_windows": len(window_geo),
            "annotated_windows": 0,
            "unannotated_windows": 0,
            "fire_assignments": {},
            "distances": [],
            "window_timeline": [],
        }

        if df_labels_filtered.empty:
            df_result: pd.DataFrame = df_window.copy()
            df_result["fire_id"] = -1
            df_result["distance_to_fire_m"] = float("inf")
            df_result["fire_id"] = df_result["fire_id"].astype(int)
            sensor_file: str = os.path.join(self.config.paths.ANNOTATED_DATA_DIR, f"sensor_{sensor_id}.parquet")
            self.logger.ensure_file_dir(sensor_file)
            df_result.to_parquet(sensor_file, index=False, compression="snappy")

            # Update metrics
            metrics["unannotated_windows"] = len(window_geo)
            return metrics

        # EXHAUSTIVE SEARCH: Find all valid fires for each window
        results: List[Dict[str, Any]] = []
        for _, window in tqdm(
            window_geo.iterrows(),
            total=len(window_geo),
            desc=f"Annotating Sensor {sensor_id}",
        ):
            # Find all fires temporally overlapping with this window
            time_overlap_mask: pd.Series = (df_labels_filtered["start_time"] <= window.window_end_time) & (
                df_labels_filtered["end_time"] >= window.window_start_time
            )
            valid_fires: gpd.GeoDataFrame = df_labels_filtered[time_overlap_mask]

            if valid_fires.empty:
                results.append(
                    {
                        "window_id": window.window_id,
                        "fire_id": -1,
                        "distance_to_fire_m": float("inf"),
                    }
                )
                metrics["unannotated_windows"] += 1
                metrics["window_timeline"].append(
                    {
                        "window_id": window.window_id,
                        "start_time": window.window_start_time,
                        "end_time": window.window_end_time,
                        "fire_id": -1,
                        "distance": float("inf"),
                    }
                )
                continue

            # Calculate distances to all valid fires
            distances: pd.Series = valid_fires.geometry.distance(window.geometry)
            min_idx: Any = distances.idxmin()
            min_distance: float = distances[min_idx]  # type: ignore

            fire_id_val: Any = valid_fires.loc[min_idx, "fire_id"]
            fire_id: int = int(fire_id_val) if pd.notna(fire_id_val) else -1  # type: ignore
            results.append(
                {
                    "window_id": window.window_id,
                    "fire_id": fire_id,
                    "distance_to_fire_m": min_distance,
                }
            )

            # Update metrics
            metrics["annotated_windows"] += 1
            metrics["fire_assignments"][fire_id] = metrics["fire_assignments"].get(fire_id, 0) + 1
            if min_distance != float("inf"):
                metrics["distances"].append(min_distance)
            metrics["window_timeline"].append(
                {
                    "window_id": window.window_id,
                    "start_time": window.window_start_time,
                    "end_time": window.window_end_time,
                    "fire_id": fire_id,
                    "distance": min_distance,
                }
            )

        # Create annotations dataframe
        annotations: pd.DataFrame = pd.DataFrame(results)
        df_result: pd.DataFrame = df_window.merge(annotations, on="window_id", how="left")
        df_result.fillna({"fire_id": -1, "distance_to_fire_m": float("inf")}, inplace=True)
        df_result["fire_id"] = df_result["fire_id"].astype(int)

        # Save per-sensor results
        sensor_file: str = os.path.join(self.config.paths.ANNOTATED_DATA_DIR, f"sensor_{sensor_id}.parquet")
        self.logger.ensure_file_dir(sensor_file)
        df_result.to_parquet(sensor_file, index=False, compression="snappy")

        del df_window, window_summary, window_geo, df_labels_filtered, annotations
        gc.collect()
        return metrics

    def coordinate_distributed_annotation(
        self,
        sensor_files: List[Path],
        df_labels: gpd.GeoDataFrame,
        df_locations: gpd.GeoDataFrame,
        client=None,
    ) -> Dict[str, Any]:
        """
        Distribute annotation tasks across multiple workers and collect metrics.

        Args:
            sensor_files: List of sensor data file paths
            df_labels: GeoDataFrame of fire labels
            df_locations: GeoDataFrame of sensor locations

        Returns:
            dict: Aggregated metrics dictionary with combined statistics from all sensors
        """
        self.logger.log_step("Starting coordinated distributed annotation")

        df_labels = df_labels.to_crs(self.config.data_pipeline.EPSG_ZONE)  # Ensure labels are in the correct CRS
        df_locations = df_locations.to_crs(
            self.config.data_pipeline.EPSG_ZONE
        )  # Ensure locations are in the correct CRS

        # Create a mapping of sensor_id to its geometry
        sensor_locations_map: Dict[str, shapely.geometry.Point] = df_locations.set_index("sensor_id")[
            "geometry"
        ].to_dict()

        # Distributed annotation using base class method
        def aggregate_fn(results):
            metrics_list = results
            aggregated_metrics = {
                "total_windows": sum(m["total_windows"] for m in metrics_list),
                "annotated_windows": sum(m["annotated_windows"] for m in metrics_list),
                "unannotated_windows": sum(m["unannotated_windows"] for m in metrics_list),
                "fire_assignments": {},
                "distances": [],
                "window_timeline": [],
            }
            for m in metrics_list:
                for fire_id, count in m["fire_assignments"].items():
                    aggregated_metrics["fire_assignments"][fire_id] = (
                        aggregated_metrics["fire_assignments"].get(fire_id, 0) + count
                    )
                aggregated_metrics["distances"].extend(m["distances"])
                aggregated_metrics["window_timeline"].extend(m["window_timeline"])
            self.logger.log_step(f"Processed {len(metrics_list)} sensor files")
            return aggregated_metrics

        arg_tuples = []
        for sensor_file in sensor_files:
            sensor_id = os.path.basename(sensor_file).replace("sensor_", "").replace(".parquet", "")
            sensor_geometry = sensor_locations_map.get(sensor_id)
            if sensor_geometry is None:
                self.logger.log_error(f"Sensor ID {sensor_id} not found in df_locations. Skipping file {sensor_file}")
                continue
            arg_tuples.append((sensor_file, df_labels, sensor_geometry))

        return self.run_distributed_map(
            self.process_sensor_annotation,
            arg_tuples,
            client=client,
            logger=self.logger,
            aggregate_fn=aggregate_fn,
            pure=False,
        ) # type: ignore

    def annotate_data(self, client=None) -> None:
        """
        Execute complete annotation workflow using provided label and location dataframes.

        Returns:
            None: Results are saved to configured output paths
        """
        self.logger.log_step("Starting high-accuracy annotation workflow")
        if self.df_labels is None or self.df_locations is None:
            raise ValueError("df_labels and df_locations must be provided to DataAnnotator.")

        # Get all sensor files
        sensor_files: List[Path] = list(Path(self.config.paths.PROCESSED_DATA_DIR).glob("sensor_*.parquet"))
        if not sensor_files:
            raise ValueError("No processed sensor data found.")

        # Process in distributed manner and collect metrics
        aggregated_metrics: Dict[str, Any] = self.coordinate_distributed_annotation(
            sensor_files, self.df_labels, self.df_locations, client=client
        )

        self._log_annotation_statistics(aggregated_metrics, self.df_labels, self.config)

        self.logger.log_step("Annotation completed")
        self.logger.save_process_timeline()
        self.logger.save_metrics_plot()

        del self.df_labels, self.df_locations
        gc.collect()
        return
