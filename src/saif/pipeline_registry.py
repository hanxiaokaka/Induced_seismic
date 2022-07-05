"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from fecs.pipelines.data_processing import dummy_pipeline, prep_seis_catalog_pipeline

from fecs.pipelines.data_science import dummy_ds_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dummy_pipeline.create_pipeline()
    data_processing_pipeline_test = dummy_pipeline.create_pipeline_test()
    prep_seis_catalog = prep_seis_catalog_pipeline.create_pipeline()

    data_science_pipeline = dummy_ds_pipeline.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        "dp_test": data_processing_pipeline_test,
        "prep_seis_catalog": prep_seis_catalog,
    }
