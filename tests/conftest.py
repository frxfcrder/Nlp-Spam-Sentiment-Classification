import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "v1": ["ham", "spam", "ham", "spam"],
            "v2": [
                "Hey, are we still on for lunch?",
                "WINNER!! Call 0800 prize now to claim",
                "See you at 7 then",
                "Free entry, text WIN to 80085 to claim",
            ],
            "Unnamed: 2": [None, None, None, None],
            "Unnamed: 3": [None, None, None, None],
            "Unnamed: 4": [None, None, None, None],
        }
    )


@pytest.fixture
def corpus():
    spam = [
        "win a free prize call now to claim your reward",
        "congratulations you won the lottery send bank details",
        "free entry to win cash txt winner to 80085 now",
        "buy cheap meds online no prescription needed",
        "urgent claim your free mobile upgrade today",
        "you have been selected for a cash prize call now",
        "win a brand new car text win to 12345 now",
        "get rich quick with this free investment offer",
        "claim your free voucher code expires today call",
        "cheap loans approved instantly apply now online",
        "win free tickets to the concert txt now to claim",
        "limited offer buy one get one free shop now",
    ]
    ham = [
        "are we still meeting for lunch tomorrow",
        "see you at seven then thanks",
        "can you pick up milk on the way home",
        "the meeting has been moved to friday",
        "thanks for the birthday wishes appreciated",
        "i will call you back after the movie",
        "did you finish the report for monday",
        "lets grab coffee sometime next week",
        "the train arrives at half past six",
        "happy birthday hope you have a great day",
        "mom asked me to call you when i get home",
        "no i already ate thanks for offering",
    ]
    texts = spam + ham
    labels = ["spam"] * len(spam) + ["ham"] * len(ham)
    return pd.Series(texts, name="text"), pd.Series(labels, name="label")
