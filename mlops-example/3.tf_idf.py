from pathlib import Path

import joblib
import pandas as pd
import polars as pl
from clearml import Dataset, Logger, OutputModel, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

task = Task.init(
    project_name="Amazon reviews",
    task_name="TF-IDF Vectorize BernoulliNB",
    output_uri=True,
)
task.add_requirements("numpy", "1.26.4")
task.add_requirements("pyarrow", "20.0.0")
logger: Logger = task.get_logger()
output_model = OutputModel(task=task, framework="scikit-learn")
args = {
    "dataset_name": "Amazon reviews dataset",
    "dataset_project": "Amazon reviews",
    "train_dataset_version": "1.2.2",
    "test_dataset_version": "1.2.3",
    "dataset_version": "1.2.4",
    "random_state": 42,
    "max_features": 1000,
    "analyzer": "word",
}
task.connect(args)
print(args)
train_dataset = Dataset.get(
    dataset_name=args["dataset_name"],
    dataset_project=args["dataset_project"],
    dataset_version=args["train_dataset_version"],
)
test_dataset = Dataset.get(
    dataset_name=args["dataset_name"],
    dataset_project=args["dataset_project"],
    dataset_version=args["test_dataset_version"],
)
dataset = Dataset.create(
    dataset_name=args["dataset_name"],
    dataset_project=args["dataset_project"],
    dataset_version=args["dataset_version"],
    parent_datasets=[train_dataset, test_dataset],
)
dataset.finalize()


train_df = (
    pl.read_parquet(Path(dataset.get_local_copy()) / "processed_train.parquet")
    .with_columns(pl.col("corpus").list.join(" "))
    .drop("index")
)
test_df = (
    pl.read_parquet(Path(dataset.get_local_copy()) / "processed_test.parquet")
    .with_columns(pl.col("corpus").list.join(" "))
    .drop("index")
)


pipe = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=int(args["max_features"]),
                analyzer=args["analyzer"],
            ),
        ),
        ("bernoulli", BernoulliNB()),
    ]
)

train_x, train_y = (
    train_df.drop("Polarity").to_pandas(),
    train_df.to_pandas()["Polarity"],
)
test_x, test_y = (
    test_df.drop("Polarity").to_pandas(),
    test_df.to_pandas()["Polarity"],
)

pipe.fit(train_x["corpus"], train_y)
joblib.dump(pipe, "model.pkl", compress=True)
output_model.update_weights(weights_filename="model.pkl")


pred_y = pipe.predict(test_x["corpus"])
classification_report_table = pd.DataFrame(
    classification_report(test_y, pred_y, output_dict=True)
).T
logger.report_table(
    "Classifiacation Report", "Metrics", table_plot=classification_report_table
)
output_model.report_table(
    "Classifiacation Report", "Metrics", table_plot=classification_report_table
)
cm = confusion_matrix(test_y, pred_y, labels=[1, 2])
logger.report_confusion_matrix("Classifiacation Report", "ConflusionMatrix", cm)
output_model.report_confusion_matrix("Classifiacation Report", "ConflusionMatrix", cm)
