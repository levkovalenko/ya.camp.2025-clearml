import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def train_vectorize(
    data: pl.DataFrame, vectorizer_params: dict, random_state: int
) -> tuple[TfidfVectorizer, pl.DataFrame, pl.DataFrame]:
    tfidf_vectorizer = TfidfVectorizer(**vectorizer_params)

    train, val = train_test_split(
        data,
        test_size=0.3,
        shuffle=True,
        random_state=random_state,
    )
    tfidf_vectorizer.fit(train["corpus"].list.join(" ").to_numpy())
    return tfidf_vectorizer, train, val


def apply_vectorizer(vectorizer: TfidfVectorizer, data: pl.DataFrame) -> pl.DataFrame:
    return vectorizer.transform(data["corpus"].list.join(" ").to_numpy())
