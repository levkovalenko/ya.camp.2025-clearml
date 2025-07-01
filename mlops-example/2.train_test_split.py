from clearml import PipelineController
from mlops_example.preprocessing import (
    dataframe_preprocessing,
    lemmatize,
    text_preprocessing,
)
from mlops_example.visualisation import class_distribution

pipe = PipelineController(
    name="DataPrepare",
    project="Amazon reviews",
    version="0.0.1",
    packages=["./mlops-example"],
    docker="python:3.11.13-slim-bookworm",
    enable_local_imports=True,
    # working_dir="./mlops-example",
)
pipe.add_parameter(
    name="dataset_name",
    description="ClearML dataset name",
    default="Amazon reviews dataset",
)
pipe.add_parameter(
    name="dataset_project",
    description="ClearML project",
    default="Amazon reviews",
)
pipe.add_parameter(
    name="dataset_version",
    description="ClearML dataset version",
    default="1.2",
)
pipe.add_parameter(
    name="test_size", description="Test ratio size", default=0.2, param_type="float"
)
pipe.add_parameter(
    name="random_state", description="Random state", default=42, param_type="int"
)


def dataset_train_test_split(
    dataset_name,
    dataset_project,
    dataset_version,
    test_size,
    random_state,
    version_postfix,
):
    from pathlib import Path

    import pandas as pd
    import polars as pl
    import pyarrow
    from clearml import Dataset, Logger
    from sklearn.model_selection import train_test_split

    print(pyarrow.__version__)

    dataset = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=dataset_version,
    )
    datset_path = Path(dataset.get_local_copy())

    data: pd.DataFrame = pl.concat(
        [pl.read_csv(data_file) for data_file in datset_path.iterdir()]
    )
    train, test = train_test_split(
        data.to_pandas(), test_size=float(test_size), random_state=int(random_state)
    )
    train_distrib = class_distribution(train, "Polarity")
    test_distrib = class_distribution(test, "Polarity")
    result_path = Path("data/prepared/split")
    result_path.mkdir(exist_ok=True, parents=True)
    train.to_csv(result_path / "raw_train.csv", index=False)
    test.to_csv(result_path / "raw_test.csv", index=False)
    prepared_dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=f"{dataset_version}.{version_postfix}",
        parent_datasets=[dataset],
    )
    prepared_dataset.add_files(result_path)
    prepared_dataset.upload()
    prepared_dataset.get_logger().report_plotly(
        "Class distribution", "Train", train_distrib
    )
    prepared_dataset.get_logger().report_plotly(
        "Class distribution", "Test", test_distrib
    )
    prepared_dataset.finalize()

    pipe_logger = Logger.current_logger()
    pipe_logger.report_plotly("Class distribution", "Train", train_distrib)
    pipe_logger.report_plotly("Class distribution", "Test", test_distrib)
    return (
        pl.from_pandas(train, include_index=False),
        pl.from_pandas(test, include_index=False),
        prepared_dataset.id,
    )


def dataset_preprocessing(
    dataframe,
    parent_dataset,
    dataset_name,
    dataset_project,
    dataset_version,
    version_postfix,
    frame_name,
):
    from pathlib import Path

    import nltk
    import polars as pl
    import pyarrow
    from clearml import Dataset

    print(pyarrow.__version__)

    nltk.download("stopwords")
    nltk.download("wordnet")

    prepared_dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=f"{dataset_version}.{version_postfix}",
        parent_datasets=[parent_dataset],
    )
    dataframe: pl.DataFrame
    processed_dataframe = dataframe_preprocessing(dataframe, "Review")

    result_path = Path("data/prepared/processed")
    result_path.mkdir(exist_ok=True, parents=True)
    processed_dataframe.write_parquet(result_path / f"processed_{frame_name}.parquet")

    prepared_dataset.add_files(result_path)
    prepared_dataset.upload()
    prepared_dataset.finalize()
    return processed_dataframe, prepared_dataset.id


pipe.add_function_step(
    name="train_test_split",
    function=dataset_train_test_split,
    function_kwargs=dict(
        dataset_name="${pipeline.dataset_name}",
        dataset_project="${pipeline.dataset_project}",
        dataset_version="${pipeline.dataset_version}",
        test_size="${pipeline.test_size}",
        random_state="${pipeline.random_state}",
        version_postfix="1",
    ),
    function_return=["raw_train_dataframe", "raw_test_dataframe", "splited_dataset_id"],
    cache_executed_step=True,
    execution_queue="default",
    helper_functions=[class_distribution],
    packages=[
        "plotly>=6.2.0,<7",
        "plotly-express>=0.4.1,<0.5",
        "clearml-serving>=1.3.5,<2",
        "scikit-learn==1.2.2",
        "numpy==1.26.4",
        "pandas>=2.3.0,<3",
        "polars>=1.31.0,<2",
        "nltk>=3.9.1,<4",
        "pyarrow>=20.0.0,<21",
    ],
)

pipe.add_function_step(
    name="train_processing",
    function=dataset_preprocessing,
    function_kwargs=dict(
        dataframe="${train_test_split.raw_train_dataframe}",
        parent_dataset="${train_test_split.splited_dataset_id}",
        dataset_name="${pipeline.dataset_name}",
        dataset_project="${pipeline.dataset_project}",
        dataset_version="${pipeline.dataset_version}",
        version_postfix="2",
        frame_name="train",
    ),
    function_return=["processed_train_dataframe", "dataset_id"],
    cache_executed_step=True,
    execution_queue="default",
    helper_functions=[lemmatize, dataframe_preprocessing, text_preprocessing],
    parents=["train_test_split"],
    packages=[
        "plotly>=6.2.0,<7",
        "plotly-express>=0.4.1,<0.5",
        "clearml-serving>=1.3.5,<2",
        "scikit-learn==1.2.2",
        "numpy==1.26.4",
        "pandas>=2.3.0,<3",
        "polars>=1.31.0,<2",
        "nltk>=3.9.1,<4",
        "pyarrow>=20.0.0,<21",
    ],
)

pipe.add_function_step(
    name="test_processing",
    function=dataset_preprocessing,
    function_kwargs=dict(
        dataframe="${train_test_split.raw_test_dataframe}",
        parent_dataset="${train_test_split.splited_dataset_id}",
        dataset_name="${pipeline.dataset_name}",
        dataset_project="${pipeline.dataset_project}",
        dataset_version="${pipeline.dataset_version}",
        version_postfix="3",
        frame_name="test",
    ),
    function_return=["processed_test_dataframe", "dataset_id"],
    cache_executed_step=True,
    execution_queue="default",
    helper_functions=[lemmatize, dataframe_preprocessing, text_preprocessing],
    parents=["train_test_split"],
    packages=[
        "plotly>=6.2.0,<7",
        "plotly-express>=0.4.1,<0.5",
        "clearml-serving>=1.3.5,<2",
        "scikit-learn==1.2.2",
        "numpy==1.26.4",
        "pandas>=2.3.0,<3",
        "polars>=1.31.0,<2",
        "nltk>=3.9.1,<4",
        "pyarrow>=20.0.0,<21",
    ],
)


pipe.start("default")
