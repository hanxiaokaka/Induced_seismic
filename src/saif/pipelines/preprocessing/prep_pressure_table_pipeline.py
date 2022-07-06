from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .prep_pressure_table_nodes import *

def create_pressure_load_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_pressure_table,
                inputs="pressure_table_data",
                outputs="pressure_table",
                tags=["preprocessing"]
            ),
        ],
    )