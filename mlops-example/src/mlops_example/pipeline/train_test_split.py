from clearml import PipelineController

pipe = PipelineController(
    name="Pipeline Controller", project="Amazon reviews", version="1.0.0"
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
    dataset_name, dataset_project, dataset_version, test_size, random_state
):
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from clearml import Dataset
    from sklearn.model_selection import train_test_split

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
    result_path = Path("data/prepared/")
    result_path.mkdir(exist_ok=True, parents=True)
    train.to_csv(result_path / "split" / "train.csv")
    test.to_csv(result_path / "split" / "test.csv")
    prepared_dataset = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=f"{dataset_version}.1",
    )
    prepared_dataset.add_files(result_path / "split")
    prepared_dataset.upload()
    prepared_dataset.finalize()
    return train, test


pipe.add_function_step(
    name="train_test_split",
    function=dataset_train_test_split,
    function_kwargs=dict(
        dataset_name="${pipeline.dataset_name}",
        dataset_project="${pipeline.dataset_project}",
        dataset_version="${pipeline.dataset_version}",
        test_size="${pipeline.test_size}",
        random_state="${pipeline.random_state}",
    ),
    function_return=["train_dataframe", "test_dataframe"],
    cache_executed_step=True,
    execution_queue="default",
    packages=[
        "clearml>=2.0.1,<3",
        "scikit-learn>=1.7.0,<2",
        "polars>=1.31.0,<2",
        "pandas>=2.3.0,<3",
        "pyarrow>=20.0.0,<21",
    ],
    docker="python:3.12.0-slim-bookworm",
)

pipe.start("default")
