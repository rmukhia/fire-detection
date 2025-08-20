"""
Data loading utilities for the forest fire detection system.
"""
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import core_logging

class DataLoader:
    """
    Handles loading of raw sensor data, fire labels, and sensor locations.
    """
    def __init__(self, config):
        self.config = config
        self.logger = core_logging.ProcessLogger(config, "DataLoader")

    def load_raw_data(self, label_load: bool, location_load: bool) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Loads the raw data, labels, and locations from files.
        This function will raise FileNotFoundError if any file is not found.

        Returns:
            tuple: Contains three elements:
                - df_sensor: Raw sensor data as pandas DataFrame
                - df_labels: Fire labels as GeoDataFrame with geometries
                - df_locations: Sensor locations as GeoDataFrame with point geometries
        """
        try:
            self.logger.log_step(f"Loading raw sensor data from {self.config.RAW_DATA_PATH}")
            df_sensor = pd.read_csv(self.config.RAW_DATA_PATH, parse_dates=['Datetime'], 
                                    nrows=self.config.S_NROWS)
            df_sensor['Datetime'] = df_sensor['Datetime'].dt.tz_localize(None)
            self.logger.log_data_summary(df_sensor, "Raw sensor data")

            if label_load:
                self.logger.log_step(f"Loading label data from {self.config.LABEL_DATA_PATH}")
                df_labels = pd.read_csv(
                    self.config.LABEL_DATA_PATH, parse_dates=['start_time', 'end_time'],
                    nrows=self.config.F_NROWS
                )
                for col in ['start_time', 'end_time']:
                    df_labels[col] = df_labels[col].dt.tz_localize(None)
                
                df_labels['geometry'] = df_labels['geometry'].apply(wkt.loads)
                df_labels = gpd.GeoDataFrame(
                    df_labels, geometry='geometry', crs='EPSG:4326'
                )
                df_labels.reset_index(inplace=True)
                df_labels.rename(columns={'index': 'fire_id'}, inplace=True)
                self.logger.log_data_summary(df_labels, "Fire labels")
            else:
                df_labels = None

            if location_load:
                self.logger.log_step(f"Loading location data from {self.config.LOCATION_DATA_PATH}")
                df_locations = pd.read_csv(self.config.LOCATION_DATA_PATH)
                df_locations.rename(columns={'Sensor_Id': 'sensor_id'}, inplace=True)
                df_locations['geometry'] = df_locations.apply(
                    lambda row: Point(row['GPS_Lon'], row['GPS_Lat']),
                    axis=1
                )
                df_locations = gpd.GeoDataFrame(
                    df_locations, geometry='geometry', crs='EPSG:4326'
                )
                self.logger.log_data_summary(df_locations, "Sensor locations")
            else:
                df_locations = None

            return df_sensor, df_labels, df_locations

        except FileNotFoundError as e:
            self.logger.log_error(f"Error loading data: {e}. Please check the file paths in your configuration.")
            raise
        except Exception as e:
            self.logger.log_error(f"An unexpected error occurred during data loading: {e}")
            raise
