
from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "uciml/sms-spam-collection-dataset"
RAW_FILE = "spam.csv"


def download_dataset() -> Path:
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def load_dataset(path=None) -> pd.DataFrame:
    directory = Path(path) if path is not None else download_dataset()
    return pd.read_csv(directory / RAW_FILE, encoding="latin1")


def split_features_target(df, text_column=None, target_column="label"):
    if text_column is None:
        text_column = "clean_text" if "clean_text" in df.columns else "text"
    return df[text_column], df[target_column]
