from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "spam_classifier.joblib"


def save_model(pipeline, path=None) -> Path:
    target = Path(path) if path is not None else DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    return target


def load_model(path=None):
    source = Path(path) if path is not None else DEFAULT_MODEL_PATH
    return joblib.load(source)


def predict_labels(model, texts) -> pd.Series:
    labels = pd.Series(model.predict(texts), name="predicted_label")
    if isinstance(texts, pd.Series):
        labels.index = texts.index
    return labels


def predict_spam_probability(model, texts) -> pd.Series:
    classes = list(model.classes_)
    spam_idx = classes.index("spam") if "spam" in classes else 1
    probs = pd.Series(model.predict_proba(texts)[:, spam_idx], name="spam_probability")
    if isinstance(texts, pd.Series):
        probs.index = texts.index
    return probs
