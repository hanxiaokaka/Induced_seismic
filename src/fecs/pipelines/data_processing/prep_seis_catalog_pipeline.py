from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .prep_seis_catalog_nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_seismic_catalog,
                inputs=[],
                outputs="seismic_catalog",
                #name="create_seismic_catalog",
            ),
        ],
        namespace="data_processing",
    )