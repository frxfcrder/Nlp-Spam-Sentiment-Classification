import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import add_text_length, build_pipeline, count_words, tokenize


def test_tokenize_lowercases_and_filters_non_alpha():
    tokens = tokenize("Win FREE cash 123 now!", stopwords={"now"})
    assert tokens == ["win", "free", "cash"]


def test_tokenize_default_stopswords_is_empty():
    assert tokenize("the cat") == ["the", "cat"]


def test_count_words_returns_most_common_first():
    texts = ["spam spam ham", "spam ham ham", "ham"]
    top = count_words(texts, top_n=2)
    assert top[0] == ("ham", 4)
    assert top[1] == ("spam", 3)


def test_add_text_length_returns_copy():
    df = pd.DataFrame({"text": ["hello", "hi"]})
    out = add_text_length(df, "text")

    assert "text_length" not in df.columns
    assert out["text_length"].tolist() == [5, 2]


def test_build_pipeline_accepts_name_and_clones_estimator():
    model = LogisticRegression(max_iter=200)
    pipeline = build_pipeline(model, "TfidfVectorizer")

    assert list(pipeline.named_steps) == ["vectorizer", "classifier"]
    assert pipeline.named_steps["classifier"] is not model
    assert pipeline.named_steps["vectorizer"].__class__.__name__ == "TfidfVectorizer"


def test_build_pipeline_leaves_original_unfitted():
    model = LogisticRegression(max_iter=200)
    pipeline = build_pipeline(model, "CountVectorizer")
    pipeline.fit(["spam text", "ham text"], ["spam", "ham"])

    assert not hasattr(model, "coef_")
    assert hasattr(pipeline.named_steps["classifier"], "coef_")
