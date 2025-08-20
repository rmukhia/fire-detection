"""Generic utility functions for forest fire detection."""

import os


def set_matplotlib_backend_for_workers():
    """
    Set matplotlib backend to 'Agg' in Dask/distributed worker processes to avoid Tkinter errors.
    Call this before importing matplotlib.pyplot in worker code.
    """
    # Dask sets this env variable in workers; adjust as needed for your setup
    if os.environ.get("DASK_WORKER") == "1" or not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")


import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
