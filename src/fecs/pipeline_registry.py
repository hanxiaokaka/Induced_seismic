"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from fecs.pipelines import data_processing as dp
from fecs.pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    data_processing_pipeline_test = dp.create_pipeline_test()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        "dp_test": data_processing_pipeline_test,
    }
