import pandas as pd

from src.preprocessing import clean_dataframe, preprocess_text


class IdentityLemmatizer:
    def lemmatize(self, word):
        return word


STOP = {"the", "a", "for", "to"}


def test_preprocess_text_lowercases_and_strips_punctuation():
    text = preprocess_text("Hello, WORLD! Free-Entry 123", stop_words=set(),
                           lemmatizer=IdentityLemmatizer())
    assert text == "hello world freeentry"


def test_preprocess_text_removes_stopwords():
    text = preprocess_text("the cat sat for the dog", stop_words=STOP,
                           lemmatizer=IdentityLemmatizer())
    assert text == "cat sat dog"
    assert "the" not in text.split()


def test_clean_dataframe_drops_and_renames(raw_df):
    df = clean_dataframe(raw_df)

    assert not any(col.startswith("Unnamed") for col in df.columns)
    assert {"label", "text", "clean_text", "text_length"} <= set(df.columns)
    assert len(df) == 4


def test_clean_dataframe_adds_text_length(raw_df):
    df = clean_dataframe(raw_df)

    assert (df["text_length"] == df["text"].str.len()).all()
    assert df["text_length"].between(1, 500).all()


def test_clean_dataframe_drops_missing_labels():
    raw = pd.DataFrame({"v1": ["ham", None], "v2": ["hi there", "win free"]})
    df = clean_dataframe(raw)

    assert len(df) == 1
    assert df["label"].tolist() == ["ham"]


def test_clean_dataframe_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean_dataframe(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)
