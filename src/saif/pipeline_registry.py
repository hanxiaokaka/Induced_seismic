"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from saif.pipelines.preprocessing import (
    prep_seis_catalog_pipeline,
    prep_pressure_table_pipeline,
)

from saif.pipelines.models import crs_orion_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    prep_seis_catalog = prep_seis_catalog_pipeline.create_catalog_load_pipeline()
    prep_pressure_table = prep_pressure_table_pipeline.create_pressure_load_pipeline()

    forecast_crs_orion = crs_orion_pipeline.create_crs_pipeline()

    return {
        "__default__": prep_seis_catalog + prep_pressure_table + forecast_crs_orion,
        "prep_seis_catalog": prep_seis_catalog,
        "prep_pressure_table": prep_pressure_table,
        "forecast_crs_orion": forecast_crs_orion,
    }
