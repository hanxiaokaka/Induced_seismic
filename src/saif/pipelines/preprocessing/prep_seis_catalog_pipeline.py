from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .prep_seis_catalog_nodes import *

def create_catalog_download_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_seismic_catalog,
                inputs=[],
                outputs="seismic_catalog",
                #name="create_seismic_catalog",
            ),
        ],
        namespace="preprocessing",
    )

def create_catalog_load_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_seismic_catalog,
                inputs="seismic_catalog_data",
                outputs="seismic_catalog",
            ),
        ],
        namespace="preprocessing",
    )