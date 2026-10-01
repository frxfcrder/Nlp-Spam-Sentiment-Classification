import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1 Score"]


def evaluate_model(pipeline, name, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    }


def evaluate_models(models, X_test, y_test) -> pd.DataFrame:
    return pd.DataFrame(
        [evaluate_model(pipeline, name, X_test, y_test) for name, pipeline in models.items()]
    )


def format_metrics_table(results, percent=True) -> pd.DataFrame:
    table = results.copy()
    if percent:
        for col in METRIC_COLS:
            if col in table.columns:
                table[col] = (table[col] * 100).map("{:.2f}%".format)
    return table


def classification_summary(pipeline, X_test, y_test, digits=4) -> str:
    return classification_report(y_test, pipeline.predict(X_test), digits=digits)
