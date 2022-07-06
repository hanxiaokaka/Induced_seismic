from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .prep_pressure_table_nodes import *

def create_crs_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=forecast_crs,
                inputs=["pressure_table", "seismic_catalog"],
                outputs="cumulative_seismic_events",
            ),
        ],
        namespace="crs",
    )