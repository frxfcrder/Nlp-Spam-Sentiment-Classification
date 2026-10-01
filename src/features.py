from collections import Counter

from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

VECTORIZERS = {
    "CountVectorizer": CountVectorizer,
    "TfidfVectorizer": TfidfVectorizer,
}


def add_text_length(df, text_column="text", column_name="text_length"):
    df = df.copy()
    df[column_name] = df[text_column].astype(str).str.len()
    return df


def tokenize(text, stopwords=()):
    return [
        word
        for word in (token.lower() for token in str(text).split())
        if word.isalpha() and word not in stopwords
    ]


def count_words(texts, top_n=10, stopwords=()):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text, stopwords))
    return counter.most_common(top_n)


def build_pipeline(model, vectorizer="CountVectorizer"):
    if isinstance(vectorizer, str):
        vectorizer = VECTORIZERS[vectorizer]
    if isinstance(vectorizer, type):
        vectorizer = vectorizer()

    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", clone(model)),
    ])
