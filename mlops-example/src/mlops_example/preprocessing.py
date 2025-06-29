import re

import nltk
import polars as pl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")


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
