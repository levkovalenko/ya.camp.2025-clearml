import click
import joblib
import polars as pl
from clearml import Dataset, Task

from .model import test, train
from .preprocessing import dataframe_preprocessing
from .vectorize import apply_vectorizer, train_vectorize


@click.command()
@click.argument("vec_max_feature", type=int, default=1000)
@click.argument("vec_analyzer", type=str, default="word")
@click.argument("random_state", type=int, default=42)
@click.argument("lr_multi_class", type=str, default="multinomial")
@click.argument("lr_solver", type=str, default="saga")
def cli_clearml(
    vec_max_feature: int,
    vec_analyzer: str,
    random_state: int,
    lr_multi_class: str,
    lr_solver: str,
):
    task = Task.init(
        project_name="Amazon reviews", task_name="baseline", output_uri=True
    )
    frame_path = Dataset.get(
        dataset_name="Raw data", dataset_project="Amazon reviews"
    ).get_local_copy()
    task.set_progress(0)
    data = pl.read_csv(
        frame_path + "/train.csv",
        has_header=False,
        new_columns=["Polarity", "Title", "Review"],
        n_rows=5000,
    )
    task.set_progress(10)
    processed_data = dataframe_preprocessing(data, "Review")
    task.set_progress(20)
    task.upload_artifact(name="processed_data", artifact_object=processed_data)

    vectorizer, train_data, test_data = train_vectorize(
        processed_data,
        {"max_features": vec_max_feature, "analyzer": vec_analyzer},
        random_state,
    )
    joblib.dump(vectorizer, "models/vectorizer.pkl", compress=True)

    train_result = apply_vectorizer(vectorizer, train_data)
    test_result = apply_vectorizer(vectorizer, test_data)
    task.set_progress(50)
    task.upload_artifact(
        name="train_features",
        artifact_object=(train_result, train_data["Polarity"].to_numpy()),
    )
    task.upload_artifact(
        name="test_features",
        artifact_object=(test_result, test_data["Polarity"].to_numpy()),
    )
    model = train(
        train_result,
        train_data["Polarity"].to_numpy(),
        {
            "random_state": random_state,
            "multi_class": lr_multi_class,
            "solver": lr_solver,
        },
    )
    joblib.dump(model, "models/model.pkl", compress=True)
    task.set_progress(80)
    result, confusion = test(model, test_result, test_data["Polarity"].to_numpy())
    task.set_progress(90)
    logger = task.get_logger()
    logger.report_single_value("accuracy", result.pop("accuracy"))
    for class_name, metrics in result.items():
        for metric, value in metrics.items():
            logger.report_single_value(f"{class_name}_{metric}", value)
    logger.report_confusion_matrix("conflusion matrix", "ignored", matrix=confusion)
