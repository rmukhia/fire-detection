import pandas as pd
import geopandas as gpd
from aad.common.config import Config


import os


class GroundTruthCollector:
    """
    Class to collect ground truth data for forest fire detection.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.firms_data_path = os.path.join(config.paths.DATA_DIR, "Hotspot_Jan-May-2023_100km.csv")
        self.report_data_path = os.path.join(config.paths.DATA_DIR, "GroundTruth_Combine_Output.csv")
        self.LOCAL_OFFSET_MINUTES = config.data_pipeline.LOCAL_OFFSET_MINUTES

    def _collect_firms(self, start_end_offset_min: int) -> pd.DataFrame:
        """
        Collect ground truth data for forest fire detection.
        """
        # Load the ground truth data
        df_firms = pd.read_csv(self.firms_data_path)
        # Process the data as needed
        # For example, filter or transform the data
        df_firms["utc_time"] = pd.to_datetime(
            df_firms["acq_date"].astype(str) + " " + df_firms["acq_time"].astype(str).str.zfill(4),
            format="%Y-%m-%d %H%M",
        )

        df_firms["local_time"] = df_firms["utc_time"] + pd.to_timedelta(self.LOCAL_OFFSET_MINUTES, unit="m")
        df_firms["local_time_in_minutes"] = df_firms["local_time"].dt.hour * 60 + df_firms["local_time"].dt.minute

        # Adjust start and end times based on the offset, in datetime format
        df_firms["start_time"] = df_firms["local_time"] - pd.to_timedelta(start_end_offset_min, unit="m")
        df_firms["end_time"] = df_firms["local_time"] + pd.to_timedelta(start_end_offset_min, unit="m")
        df_firms["report_time"] = df_firms["local_time"]  # Assuming report_time is the same as local_time
        geometry = gpd.points_from_xy(df_firms["longitude"], df_firms["latitude"])
        df_firms = gpd.GeoDataFrame(df_firms, geometry=geometry, crs="EPSG:4326")

        return df_firms

    def _collect_report(self) -> gpd.GeoDataFrame:
        """
        Collect report data for forest fire detection and return a GeoDataFrame
        with start_time, end_time, location, and geometry (from lat/lon).
        """

        df = pd.read_csv(self.report_data_path)

        # Expected columns (case-insensitive): date, start, end, location, latitude, longitude
        # Optional: report_time

        def combine_date_and_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
            # parse time flexibly; if parsing fails, will become NaT
            dt = pd.to_datetime(date_series + " " + time_series)
            return dt

        df["start_time"] = combine_date_and_time(df["Date"], df["start"])
        df["end_time"] = combine_date_and_time(df["Date"], df["end"])
        df["report_time"] = combine_date_and_time(df["Date"], df["report_time"])

        df["instrument"] = "ranger"

        df = df.dropna(subset=["latitude", "longitude", "start_time", "end_time"])

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )

        return gdf

    def collect_groundtruth(self, start_end_offset_min: int) -> gpd.GeoDataFrame:
        """
        Collect ground truth data for forest fire detection.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the ground truth data.
        """

        columns = ["report_time", "start_time", "end_time", "instrument", "geometry"]
        df_firms = self._collect_firms(start_end_offset_min)[columns]
        df_report = self._collect_report()[columns]

        # Combine the data as needed
        # For example, merge or concatenate the DataFrames
        groundtruth_data = pd.concat([df_firms, df_report], ignore_index=True)

        # Convert back to GeoDataFrame to maintain geometry
        groundtruth_data = gpd.GeoDataFrame(groundtruth_data, geometry=groundtruth_data["geometry"], crs="EPSG:4326")
        # Add unique fire_id for each fire event
        groundtruth_data = groundtruth_data.reset_index(drop=True)
        groundtruth_data["fire_id"] = groundtruth_data.index

        # save the combined data to a CSV file
        groundtruth_data.to_csv(self.config.paths.LABEL_DATA_PATH, index=False)

        return groundtruth_data
