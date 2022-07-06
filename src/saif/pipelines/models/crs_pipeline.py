from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .crs_nodes import *


def create_crs_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=forecast_crs,
                inputs=["pressure_table", "seismic_catalog"],
                outputs="forecast_crs",
                tags=["crs"],
            ),
        ],
    )
