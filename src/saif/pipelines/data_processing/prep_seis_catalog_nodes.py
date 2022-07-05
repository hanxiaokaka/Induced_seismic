import pandas as pd
import time
from saif.utils.seismic_catalog import SeismicCatalog

def create_seismic_catalog() -> pd.DataFrame:
    """Save a random csv

    Args:
        None
    Returns:
        A dummy csv file
    """

    epoch_range = [time.time() - 60 * 60 * 24 * 365, time.time()]
    latitude_range = [38.7, 38.95]
    longitude_range = [-123.0, -122.6]
    catalog = SeismicCatalog()
    catalog.load_catalog_comcat(epoch_range, latitude_range, longitude_range)

    return catalog