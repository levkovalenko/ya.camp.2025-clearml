import re
from typing import Any

import nltk
import numpy as np
import polars as pl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")


def text_preprocessing(input_text: str) -> str:
    text = input_text.lower()  # приведение к нижнему регистру
    text = re.sub(
        r"https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*", "", text
    )  # убираем ссылки
    text = re.sub("[0-9 \-_]+", " ", text)  # убираем спец символы
    text = re.sub("[^a-z A-Z]+", " ", text)  # оставляем только буквы
    text = " ".join(  # убираем стоп слова
        [word for word in text.split() if word not in stopwords.words("english")]
    )
    return text.strip()


def lemmatize(input_frame: pl.DataFrame) -> pl.DataFrame:
    lemmatizer = WordNetLemmatizer()

    return input_frame.with_columns(
        pl.col("corpus").map_elements(
            lambda input_list: [lemmatizer.lemmatize(token) for token in input_list]
        )
    )


def dataframe_preprocessing(data: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return lemmatize(
        data.with_columns(
            pl.col(col_name)
            .map_elements(text_preprocessing)
            .str.split(" ")
            .alias("corpus")
        )
    )


class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(
        self, body: dict, state: dict, collect_custom_statistics_fn=None
    ) -> Any:
        # we expect to get two valid on the dict x0, and x1
        text = body.get("text", None)
        processed_words = text_preprocessing(text).split(" ")
        lemmatizer = WordNetLemmatizer()
        processed_text = " ".join(
            [lemmatizer.lemmatize(token) for token in processed_words]
        )
        return [processed_text]

    def postprocess(
        self, data: Any, state: dict, collect_custom_statistics_fn=None
    ) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=data.tolist() if isinstance(data, np.ndarray) else data)
