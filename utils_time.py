"""Utilities for time format conversion in forest fire detection."""
import pandas as pd


def timedelta_to_hms(td):
    """
    Convert pandas Timedelta to HH:MM:SS string format.
    
    Args:
        td: pandas.Timedelta to convert
        
    Returns:
        str: Time in HH:MM:SS format or None if input is NA
    """
    if pd.isna(td):
        return None
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'