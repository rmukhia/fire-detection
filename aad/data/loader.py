"""
Data loading utilities for the forest fire detection system.
"""

import pandas as pd
import geopandas as gpd

from typing import Optional, Tuple
from aad.common import core_logging
from aad.common.config import Config
from .utils import localize_datetime_columns, parse_geometry_column


class DataLoader:
    """
    Handles loading of raw sensor data, fire labels, and sensor locations.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = core_logging.ProcessLogger(config, "DataLoader")

    def load_raw_data(
        self, label_load: bool, location_load: bool
    ) -> Tuple[pd.DataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Loads the raw data, labels, and locations from files.

        Args:
            label_load (bool): Whether to load label data.
            location_load (bool): Whether to load location data.

        Returns:
            tuple: Contains three elements:
                - df_sensor: Raw sensor data as pandas DataFrame
                - df_labels: Fire labels as GeoDataFrame with geometries
                - df_locations: Sensor locations as GeoDataFrame with point geometries
        """
        self.logger.log_step(f"Loading raw sensor data from {self.config.paths.RAW_DATA_PATH}")
        df_sensor = pd.read_csv(self.config.paths.RAW_DATA_PATH, parse_dates=["Datetime"])
        df_sensor = localize_datetime_columns(df_sensor, ["Datetime"])
        self.logger.log_data_summary(df_sensor, "Raw sensor data")

        if label_load:
            self.logger.log_step(f"Loading label data from {self.config.paths.LABEL_DATA_PATH}")
            df_labels = pd.read_csv(
                self.config.paths.LABEL_DATA_PATH,
                parse_dates=["start_time", "end_time"],
            )
            df_labels = localize_datetime_columns(df_labels, ["start_time", "end_time"])
            df_labels = parse_geometry_column(df_labels, geometry_col="geometry", crs="EPSG:4326")
            df_labels.reset_index(inplace=True)
            df_labels.rename(columns={"index": "fire_id"}, inplace=True)
            self.logger.log_data_summary(df_labels, "Fire labels")
        else:
            df_labels = None

        if location_load:
            self.logger.log_step(f"Loading location data from {self.config.paths.LOCATION_DATA_PATH}")
            df_locations = pd.read_csv(self.config.paths.LOCATION_DATA_PATH)
            df_locations.rename(columns={"Sensor_Id": "sensor_id"}, inplace=True)
            df_locations["geometry"] = gpd.points_from_xy(df_locations["GPS_Lon"], df_locations["GPS_Lat"])
            df_locations = gpd.GeoDataFrame(df_locations, geometry="geometry", crs="EPSG:4326")
            self.logger.log_data_summary(df_locations, "Sensor locations")
        else:
            df_locations = None

        return df_sensor, df_labels, df_locations
