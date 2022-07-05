"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from saif.pipelines.data_processing import prep_seis_catalog_pipeline, prep_pressure_table_pipeline

#from saif.pipelines.crs import dummy_ds_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    prep_seis_catalog = prep_seis_catalog_pipeline.create_catalog_download_pipeline()
    prep_pressure_table = prep_pressure_table_pipeline.create_pressure_load_pipeline()

    return {
        "__default__": prep_seis_catalog + prep_pressure_table,
        "prep_seis_catalog": prep_seis_catalog,
        "prep_pressure_table": prep_pressure_table,
    }
