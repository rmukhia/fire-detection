import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import core_logging
import data_loader
import os
import viz_style
from core_logging import ensure_directories
from dask.diagnostics import ProgressBar
import dask
import dask.diagnostics
from dask.distributed import Client
import multiprocessing as mp
from pathlib import Path
import json
import gc
import shapely

# =========================================================================
# Visualization Setup
# =========================================================================
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# =========================================================================
# Data Annotator Class
# =========================================================================
class DataAnnotator:
    """Class for annotating sensor data with fire proximity information."""
    def __init__(self, config, logger: core_logging.ProcessLogger = None):
        self.config = config
        self.logger = logger or core_logging.ProcessLogger(config, "Annotation")
        self.annotation_metrics = {}
        ensure_directories([
            self.config.ANNOTATED_DATA_DIR,
            self.config.STATS_IMAGES_DIR,
            self.config.STATS_CSV_DIR,
        ])

    def _log_annotation_statistics(self, metrics, df_labels, config):
        """
        Generate and save annotation statistics visualizations from collected metrics.
        
        Args:
            metrics: Aggregated metrics dictionary
            df_labels: Fire labels DataFrame
            config: Configuration object
            logger: Process logger instance
        """
        self.logger.log_step("Generating annotation statistics from collected metrics")
        viz_style.set_publication_style()

        total_windows = metrics['total_windows']
        annotated_windows = metrics['annotated_windows']
        unannotated_windows = metrics['unannotated_windows']
        annotation_rate = (annotated_windows / total_windows) * 100 if total_windows > 0 else 0

        unique_fires = len(metrics['fire_assignments'])
        fire_counts = pd.Series(metrics['fire_assignments']).sort_index()
        finite_distances = [d for d in metrics['distances'] if d != float('inf')]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Forest Fire Detection - Annotation Statistics', fontsize=16, fontweight='bold')

        # 1. Annotation Success Rate
        ax1 = axes[0, 0]
        sizes = [annotated_windows, unannotated_windows]
        labels = ['Annotated', 'Unannotated']
        colors = [viz_style.VALID_COLORS['green'], viz_style.VALID_COLORS['red']]
        ax1.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f'{p:.1f}% ({int(p*total_windows/100):d})', startangle=90)
        ax1.set_title('Annotation Success Rate', fontweight='bold')

        # 2. Fire ID Assignment Histogram
        ax2 = axes[0, 1]
        if len(fire_counts) > 0:
            ax2.bar(range(len(fire_counts)), fire_counts.values,
                    color=viz_style.VALID_COLORS['blue'], alpha=0.7, edgecolor='black', linewidth=0.8)
            ax2.set_xlabel('Fire ID (sorted)', fontweight='bold')
            ax2.set_ylabel('Number of Windows', fontweight='bold')
            ax2.set_title(f'Fire Assignment Distribution\n({unique_fires} unique fires)', fontweight='bold')
            ax2.grid(True, alpha=0.3, linewidth=0.6)
        else:
            ax2.text(0.5, 0.5, 'No Fire Assignments', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=14, color='red')
            ax2.set_title('Fire Assignment Distribution', fontweight='bold')

        # 3. Distance Distribution
        ax3 = axes[1, 0]
        if len(finite_distances) > 0:
            distances_km = [d / 1000 for d in finite_distances]
            ax3.hist(distances_km, bins=min(30, len(distances_km)//10 + 1),
                    color=viz_style.VALID_COLORS['orange'], alpha=0.7, edgecolor='black', linewidth=0.8)
            ax3.set_xlabel('Distance to Fire (km)', fontweight='bold')
            ax3.set_ylabel('Number of Windows', fontweight='bold')
            ax3.set_title('Distance Distribution', fontweight='bold')
            ax3.grid(True, alpha=0.3, linewidth=0.6)
            stats_text = f'Mean: {np.mean(distances_km):.2f}km\nMedian: {np.median(distances_km):.2f}km'
            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No Distance Data', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=14, color='red')
            ax3.set_title('Distance Distribution', fontweight='bold')

        # 4. Window Timeline with Fire Detection
        # ax4 = axes[1, 1]
        # if metrics['window_timeline']:
        #     window_timeline = pd.DataFrame(metrics['window_timeline'])
        #     window_timeline['start_time'] = pd.to_datetime(window_timeline['start_time'])
        #     window_timeline['end_time'] = pd.to_datetime(window_timeline['end_time'])
        #     window_timeline = window_timeline.sort_values('start_time')

        #     durations = (window_timeline['end_time'] - window_timeline['start_time']).dt.total_seconds() / 3600
        #     y_pos = np.arange(len(window_timeline))
        #     fire_detected = (window_timeline['fire_id'] != -1) & (window_timeline['distance'] != float('inf'))

        #     bar_colors = np.full((len(window_timeline), 4), [0.827, 0.827, 0.827, 0.4])
        #     if fire_detected.any():
        #         valid_distances = window_timeline.loc[fire_detected, 'distance']
        #         max_distance = valid_distances.max()
        #         normalized_distances = valid_distances / max_distance
        #         cmap = plt.cm.Reds
        #         colors = cmap(1.0 - normalized_distances * 0.7 + 0.3)
        #         bar_colors[fire_detected] = colors

        #     ax4.barh(y_pos, durations, left=mdates.date2num(window_timeline['start_time']),
        #              height=0.8, color=bar_colors)

        #     ax4.set_xlabel('Time', fontweight='bold')
        #     ax4.set_ylabel('Window ID', fontweight='bold')
        #     ax4.set_title('Window Timeline\n(Red intensity ∝ fire proximity, Gray = no fire)', fontweight='bold')
        #     ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        #     ax4.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(window_timeline)//20)))
        #     plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        #     ax4.grid(True, alpha=0.3, linewidth=0.6)

        #     if len(window_timeline) <= 30:
        #         ax4.set_yticks(y_pos)
        #         ax4.set_yticklabels([f'W{wid}' for wid in window_timeline['window_id']])
        # else:
        #     ax4.text(0.5, 0.5, 'No Window Data', transform=ax4.transAxes,
        #             ha='center', va='center', fontsize=14, color='red')
        #     ax4.set_title('Window Timeline', fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(self.config.STATS_IMAGES_DIR, os.path.basename(self.config.DISTANCE_ANNOTATED_PATH).replace('.parquet', '_statistics.png'))
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        if annotated_windows > 0:
            stats_data = {
                'Metric': ['Total Windows', 'Annotated Windows', 'Annotation Rate (%)', 'Unique Fires',
                            'Avg Distance (m)', 'Median Distance (m)', 'Avg Windows per Fire'],
                'Value': [total_windows, annotated_windows, f"{annotation_rate:.2f}", unique_fires,
                            f"{np.mean(finite_distances):.1f}" if finite_distances else "N/A",
                            f"{np.median(finite_distances):.1f}" if finite_distances else "N/A",
                            f"{np.mean(list(metrics['fire_assignments'].values())):.2f}"]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_csv_path = os.path.join(self.config.STATS_CSV_DIR, os.path.basename(self.config.DISTANCE_ANNOTATED_PATH).replace('.parquet', '_statistics.csv'))
            stats_df.to_csv(stats_csv_path, index=False)

            fire_assignment_df = pd.DataFrame({
                'fire_id': list(metrics['fire_assignments'].keys()),
                'window_count': list(metrics['fire_assignments'].values())
            })
            fire_assignment_df['percentage'] = (fire_assignment_df['window_count'] / annotated_windows * 100).round(2)
            fire_assign_path = os.path.join(self.config.STATS_CSV_DIR, os.path.basename(self.config.DISTANCE_ANNOTATED_PATH).replace('.parquet', '_fire_assignments.csv'))
            fire_assignment_df.to_csv(fire_assign_path, index=False)

        self.logger.log_step(f"Statistics saved to: {plot_path}")

    def process_sensor_annotation(self, sensor_file_path: Path, df_labels_proj: gpd.GeoDataFrame, sensor_location_geometry: shapely.geometry.Point):
        """
        Annotate sensor data with nearest fire information and collect metrics.
        
        Args:
            sensor_file_path: Path to sensor data file
            df_labels_proj: GeoDataFrame of fire labels projected to the correct CRS
            sensor_location_geometry: Shapely Point geometry of the sensor location
            
        Returns:
            tuple: (file_path, metrics) where metrics is a dict of collected statistics
        """
        df_window = pd.read_parquet(sensor_file_path)
        sensor_id = df_window['sensor_id'].iloc[0]

        # Support direct GeoDataFrames and scattered Futures for df_labels
        try:
            from dask.distributed import Future
            if isinstance(df_labels_proj, Future):
                df_labels_proj = df_labels_proj.result()
        except Exception:
            pass


        # Create window summary
        window_summary = df_window.groupby('window_id', sort=False).agg(
            window_start_time=('Datetime', 'min'),
            window_end_time=('Datetime', 'max'),
            sensor_id=('sensor_id', 'first')
        ).reset_index()

        projected_crs = self.config.EPSG_ZONE

        # Assign sensor location geometry directly
        # Convert single shapely Point to GeoSeries for to_crs method
        sensor_geometry_series = gpd.GeoSeries([sensor_location_geometry], crs=projected_crs) # Assuming original CRS is 4326
        projected_sensor_geometry = sensor_geometry_series.iloc[0]

        window_geo = gpd.GeoDataFrame(window_summary, geometry=[projected_sensor_geometry] * len(window_summary), crs=projected_crs)

        # Filter relevant fires
        window_time_max = df_window['Datetime'].max()
        window_time_min = df_window['Datetime'].min()
        fire_mask = (df_labels_proj['start_time'] <= window_time_max) & (df_labels_proj['end_time'] >= window_time_min)
        df_labels_filtered = df_labels_proj[fire_mask].copy()

        # Initialize metrics collection
        metrics = {
            'sensor_id': sensor_id,
            'total_windows': len(window_geo),
            'annotated_windows': 0,
            'unannotated_windows': 0,
            'fire_assignments': {},
            'distances': [],
            'window_timeline': []
        }

        if df_labels_filtered.empty:
            df_result = df_window.copy()
            df_result['fire_id'] = -1
            df_result['distance_to_fire_m'] = float('inf')
            df_result['fire_id'] = df_result['fire_id'].astype(int)
            sensor_file = os.path.join(self.config.ANNOTATED_DATA_DIR, f"sensor_{sensor_id}.parquet")
            df_result.to_parquet(sensor_file, index=False, compression='snappy')
            
            # Update metrics
            metrics['unannotated_windows'] = len(window_geo)
            return metrics

        # EXHAUSTIVE SEARCH: Find all valid fires for each window
        results = []
        for _, window in tqdm(window_geo.iterrows(), total=len(window_geo),
                             desc=f"Annotating Sensor {sensor_id}"):
            # Find all fires temporally overlapping with this window
            time_overlap_mask = (
                (df_labels_filtered['start_time'] <= window.window_end_time) &
                (df_labels_filtered['end_time'] >= window.window_start_time)
            )
            valid_fires = df_labels_filtered[time_overlap_mask]
            
            if valid_fires.empty:
                results.append({
                    'window_id': window.window_id,
                    'fire_id': -1,
                    'distance_to_fire_m': float('inf')
                })
                metrics['unannotated_windows'] += 1
                metrics['window_timeline'].append({
                    'window_id': window.window_id,
                    'start_time': window.window_start_time,
                    'end_time': window.window_end_time,
                    'fire_id': -1,
                    'distance': float('inf')
                })
                continue
                
            # Calculate distances to all valid fires
            distances = valid_fires.geometry.distance(window.geometry)
            min_idx = distances.idxmin()
            min_distance = distances[min_idx]
            
            fire_id = valid_fires.loc[min_idx, 'fire_id']
            results.append({
                'window_id': window.window_id,
                'fire_id': fire_id,
                'distance_to_fire_m': min_distance
            })
            
            # Update metrics
            metrics['annotated_windows'] += 1
            metrics['fire_assignments'][fire_id] = metrics['fire_assignments'].get(fire_id, 0) + 1
            if min_distance != float('inf'):
                metrics['distances'].append(min_distance)
            metrics['window_timeline'].append({
                'window_id': window.window_id,
                'start_time': window.window_start_time,
                'end_time': window.window_end_time,
                'fire_id': fire_id,
                'distance': min_distance
            })

        # Create annotations dataframe
        annotations = pd.DataFrame(results)
        df_result = df_window.merge(annotations, on='window_id', how='left')
        df_result.fillna({'fire_id': -1, 'distance_to_fire_m': float('inf')}, inplace=True)
        df_result['fire_id'] = df_result['fire_id'].astype(int)
        
        # Save per-sensor results
        sensor_file = os.path.join(self.config.ANNOTATED_DATA_DIR, f"sensor_{sensor_id}.parquet")
        df_result.to_parquet(sensor_file, index=False, compression='snappy')

        del df_window, window_summary, window_geo, df_labels_filtered, annotations
        gc.collect()
        return metrics

    def coordinate_distributed_annotation(self, sensor_files: list, df_labels: gpd.GeoDataFrame, df_locations: gpd.GeoDataFrame):
        """
        Distribute annotation tasks across multiple workers and collect metrics.
        
        Args:
            sensor_files: List of sensor data file paths
            df_labels: GeoDataFrame of fire labels
            df_locations: GeoDataFrame of sensor locations
            
        Returns:
            tuple: (annotated_files, aggregated_metrics) where:
                annotated_files: list of paths to annotated files
                aggregated_metrics: dict of combined statistics from all sensors
        """
        self.logger.log_step("Starting coordinated distributed annotation")
        n_workers = min(mp.cpu_count(), len(sensor_files), self.config.NUM_WORKERS)
        
        df_labels = df_labels.to_crs(self.config.EPSG_ZONE)  # Ensure labels are in the correct CRS
        df_locations = df_locations.to_crs(self.config.EPSG_ZONE)  # Ensure locations are in the correct CRS

        with Client(n_workers=n_workers, threads_per_worker=1) as client:
            self.logger.log_step(f"Started Dask cluster with {n_workers} workers: {client.dashboard_link}")

            # Broadcast data to workers
            df_labels_future = client.scatter(df_labels, broadcast=True)
            
            # Create a mapping of sensor_id to its geometry
            sensor_locations_map = df_locations.set_index('sensor_id')['geometry'].to_dict()

            # Create and process tasks
            tasks = []
            for sensor_file in sensor_files:
                # Extract sensor_id from the file path (assuming format sensor_ID.parquet)
                sensor_id = os.path.basename(sensor_file).replace('sensor_', '').replace('.parquet', '')
                sensor_geometry = sensor_locations_map.get(sensor_id)
                
                if sensor_geometry is None:
                    self.logger.log_error(f"Sensor ID {sensor_id} not found in df_locations. Skipping file {sensor_file}")
                    continue

                tasks.append(dask.delayed(self.process_sensor_annotation)(
                    sensor_file,
                    df_labels_future,
                    sensor_geometry
                ))

            with ProgressBar():
                results = list(dask.compute(*tasks))

        # Separate file paths and metrics
        metrics_list = results
        
        # Aggregate metrics
        aggregated_metrics = {
            'total_windows': sum(m['total_windows'] for m in metrics_list),
            'annotated_windows': sum(m['annotated_windows'] for m in metrics_list),
            'unannotated_windows': sum(m['unannotated_windows'] for m in metrics_list),
            'fire_assignments': {},
            'distances': [],
            'window_timeline': []
        }
        
        # Combine fire assignments
        for m in metrics_list:
            for fire_id, count in m['fire_assignments'].items():
                aggregated_metrics['fire_assignments'][fire_id] = aggregated_metrics['fire_assignments'].get(fire_id, 0) + count
            aggregated_metrics['distances'].extend(m['distances'])
            aggregated_metrics['window_timeline'].extend(m['window_timeline'])

        self.logger.log_step(f"Processed {len(results)} sensor files")
        return aggregated_metrics

    def annotate_data(self):
        """
        Execute complete annotation workflow.
        
        Returns:
            None: Results are saved to configured output paths
        """
        self.logger.log_step("Starting high-accuracy annotation workflow")
        data_loader_instance = data_loader.DataLoader(self.config)
        _, df_labels, df_locations = data_loader_instance.load_raw_data(True, True)
        if df_labels is None or df_locations is None:
            raise ValueError("Failed to load labels or locations data")

        # Get all sensor files
        sensor_files = list(Path(self.config.PROCESSED_DATA_DIR).glob("sensor_*.parquet"))
        if not sensor_files:
            raise ValueError("No processed sensor data found.")

        # Process in distributed manner and collect metrics
        aggregated_metrics = self.coordinate_distributed_annotation(sensor_files, df_labels, df_locations)

        self._log_annotation_statistics(aggregated_metrics, df_labels, self.config)

        self.logger.log_step("Annotation completed")
        self.logger.save_process_timeline()
        self.logger.save_metrics_plot()

        del df_labels, df_locations
        gc.collect()
        return

# =========================================================================
# Main Function
# =========================================================================
def annotate_data(config):
    """Main function to run data annotation process."""
    annotator = DataAnnotator(config)
    annotator.annotate_data()