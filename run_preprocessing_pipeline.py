import os
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
import dask.config
from aad.common.config import Config
from aad.data.loader import DataLoader
from aad.data.preprocessing import DataPreprocessor
from aad.data.annotation import DataAnnotator
from aad.data.sequences import DataSequencer
from aad.data.groundtruth import GroundTruthCollector
from aad.common.core_logging import ProcessLogger

from dask.distributed import Client
# %%
def main():
# %%
    config = Config()
# %%
    loader = DataLoader(config)
    # Load all data
    df_sensor, _, df_locations = loader.load_raw_data(label_load=False, location_load=True)
    n_workers: int = min(mp.cpu_count(), config.data_pipeline.NUM_WORKERS)
# %%
    #dask.config.set({'temporary_directory': r'D:\tmp'})
    # Start Dask cluster for the entire pipeline
    with Client(n_workers=1, threads_per_worker=1) as client:
        # Preprocessing
        preprocessor = DataPreprocessor(config, df_sensor=df_sensor)
        preprocessor.preprocess_data(client=client)
    # %%
    dask.config.set({'temporary_directory': r'D:\tmp'})
    with Client(n_workers=3, threads_per_worker=1) as client:
        # Ground Truth Processing
        groundtruth_collector = GroundTruthCollector(config)
        df_groundtruth = groundtruth_collector.collect_groundtruth(start_end_offset_min=180)
        # Annotation (using ground truth as labels)
        annotator = DataAnnotator(config, df_labels=df_groundtruth, df_locations=df_locations)
        annotator.annotate_data(client=client)
    # %%
    # Sequence creation
    sequencer = DataSequencer(config, ProcessLogger(config,'sequencer'))
    sequencer.create_dataset(fit_scaler=True)
    # %%

if __name__ == "__main__":
    main()
    
    