from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .features import VECTORIZERS, build_pipeline

DEFAULT_MODELS = {
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_one(model, X_train, y_train, vectorizer="CountVectorizer"):
    pipeline = build_pipeline(model, vectorizer)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_all(X_train, y_train, models=None, vectorizers=None):
    models = DEFAULT_MODELS if models is None else models
    vectorizers = VECTORIZERS if vectorizers is None else vectorizers

    trained = {}
    for name, model in models.items():
        for vec_name in vectorizers:
            trained[f"{name} + {vec_name}"] = train_one(model, X_train, y_train, vec_name)
    return trained
