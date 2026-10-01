import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .features import add_text_length

NON_ALPHA = re.compile(r"[^a-z\s]")


def ensure_nltk_data(*names):
    for name in names:
        try:
            nltk.data.find(f"corpora/{name}")
        except LookupError:
            nltk.download(name, quiet=True)


def get_stopwords() -> set:
    ensure_nltk_data("stopwords")
    return set(stopwords.words("english"))


def get_lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data("wordnet")
    return WordNetLemmatizer()


def preprocess_text(text, stop_words=None, lemmatizer=None) -> str:
    if stop_words is None:
        stop_words = get_stopwords()
    if lemmatizer is None:
        lemmatizer = get_lemmatizer()

    text = NON_ALPHA.sub("", str(text).lower())
    words = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    junk = [col for col in df.columns if col.startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)

    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].astype(str)

    df = add_text_length(df, "text")
    df["clean_text"] = df["text"].map(preprocess_text)
    return df.reset_index(drop=True)
