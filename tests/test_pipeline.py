import numpy as np

from src.evaluate import evaluate_models, format_metrics_table
from src.predict import load_model, predict_labels, predict_spam_probability, save_model
from src.train import DEFAULT_MODELS, split_data, train_all

METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]


def test_split_data_is_stratified(corpus):
    X, y = corpus
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)

    assert len(X_train) + len(X_test) == len(X)
    assert set(y_train) == {"ham", "spam"}
    assert set(y_test) == {"ham", "spam"}


def test_train_all_fits_every_combination(corpus):
    X, y = corpus
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_all(X_train, y_train)

    assert len(models) == len(DEFAULT_MODELS) * 2
    assert "Logistic Regression + CountVectorizer" in models
    for pipeline in models.values():
        assert hasattr(pipeline, "predict")
        assert pipeline.predict(X_test[:1]) is not None


def test_train_all_does_not_share_state_between_pipelines(corpus):
    X, y = corpus
    X_train, _, y_train, _ = split_data(X, y)

    models = train_all(X_train, y_train)
    classifiers = [p.named_steps["classifier"] for p in models.values()]

    assert len({id(c) for c in classifiers}) == len(classifiers)


def test_evaluate_models_shape_and_ranges(corpus):
    X, y = corpus
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_all(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    assert len(results) == len(models)
    assert ["Model"] + METRICS == list(results.columns)
    assert results[METRICS].apply(lambda col: col.between(0, 1).all()).all()


def test_format_metrics_table_formats_only_numeric_metrics(corpus):
    X, y = corpus
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_all(X_train, y_train)

    table = format_metrics_table(evaluate_models(models, X_test, y_test))

    assert table["Accuracy"].str.endswith("%").all()
    assert table["F1 Score"].str.endswith("%").all()
    assert not table["Model"].str.contains("%").any()


def test_save_load_roundtrip(tmp_path, corpus):
    X, y = corpus
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_all(X_train, y_train)
    pipeline = models["Logistic Regression + CountVectorizer"]

    path = save_model(pipeline, tmp_path / "model.joblib")
    restored = load_model(path)

    np.testing.assert_array_equal(pipeline.predict(X_test), restored.predict(X_test))


def test_predict_labels_and_probability_index(corpus):
    X, y = corpus
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_all(X_train, y_train)["Logistic Regression + CountVectorizer"]

    labels = predict_labels(pipeline, X_test)
    probs = predict_spam_probability(pipeline, X_test)

    assert set(labels) <= {"ham", "spam"}
    assert labels.index.equals(X_test.index)
    assert probs.between(0, 1).all()
    assert probs.name == "spam_probability"
