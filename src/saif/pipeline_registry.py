"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from saif.pipelines.data_processing import dummy_pipeline, prep_seis_catalog_pipeline

from saif.pipelines.crs import dummy_ds_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    dummy_data_processing_pipeline = dummy_pipeline.create_pipeline()
    dummy_data_processing_pipeline_test = dummy_pipeline.create_pipeline_test()
    prep_seis_catalog = prep_seis_catalog_pipeline.create_catalog_creation_pipeline()

    dummy_data_science_pipeline = dummy_ds_pipeline.create_pipeline()

    return {
        "__default__": dummy_data_processing_pipeline + dummy_data_science_pipeline,
        "dp": dummy_data_processing_pipeline,
        "ds": dummy_data_science_pipeline,
        "dp_test": dummy_data_processing_pipeline_test,
        "prep_seis_catalog": prep_seis_catalog,
    }
