import pandas as pd
import time
from saif.utils.seismic_catalog import SeismicCatalog

def download_seismic_catalog() -> SeismicCatalog:
    """Create seismic catalog by downloading

    Args:
        None
    Returns:
        A catalog instance
    """

    epoch_range = [time.time() - 60 * 60 * 24 * 365, time.time()]
    latitude_range = [38.7, 38.95]
    longitude_range = [-123.0, -122.6]
    catalog = SeismicCatalog()
    catalog.load_catalog_comcat(epoch_range, latitude_range, longitude_range)

    return catalog


def load_seismic_catalog(catalog_data: pd.DataFrame) -> SeismicCatalog:
    """Create seismic catalog by loading from csv file

    Args:
        catalog_data: dataframe containing a seismic catalog
    Returns:
        A catalog instance
    """

    #data = {k: np.loadtxt(os.path.join(table_dir, '%s.csv' % (k))) for k in ['x', 'y', 'z', 't', 'pressure']}
    data = {columnName: columnData  for (columnName, columnData) in catalog_data.iteritems() }

    catalog = SeismicCatalog()
    catalog.load_catalog_array(**data)

    return catalog


