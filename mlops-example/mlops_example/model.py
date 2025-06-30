import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def train(
    data: np.ndarray, target: np.ndarray, model_params: dict
) -> LogisticRegression:
    model_lr = LogisticRegression(**model_params)
    model_lr.fit(data, target)
    return model_lr


def test(
    model: LogisticRegression, data: np.ndarray, target: np.ndarray
) -> tuple[dict, np.ndarray]:
    predicts = model.predict(data)
    return classification_report(target, predicts, output_dict=True), confusion_matrix(
        target, predicts
    )
