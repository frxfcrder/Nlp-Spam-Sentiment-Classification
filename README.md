# NLP Spam Sentiment Classification

## Dataset

- Source: Kaggle — SMS Spam Collection (auto-downloaded with kagglehub)
- Size: 5,572 messages x 2 columns (label + text), no missing values
- Target: label — binary (ham / spam), imbalanced (~13.4% spam)
- Features: text — raw SMS, lowercased, punctuation stripped, stopwords removed, lemmatized

## Project Structure

    nlp-spam-sentiment-classification/
    ├── data/
    ├── models/
    ├── notebooks/
    │   └── 01_eda.ipynb
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py
    │   ├── preprocessing.py
    │   ├── features.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    ├── tests/
    │   ├── conftest.py
    │   ├── test_preprocessing.py
    │   ├── test_features.py
    │   └── test_pipeline.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

data/ holds the downloaded CSV and models/ the saved artifacts; both are git-ignored (.gitkeep keeps the folders in the repo).

## How to Run

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

On macOS/Linux activate with `source .venv/bin/activate`.

Notebook (EDA + results):

    jupyter notebook notebooks/01_eda.ipynb

As a library:

```python
from src.data_loader import load_dataset, split_features_target
from src.preprocessing import clean_dataframe
from src.train import split_data, train_all
from src.evaluate import evaluate_models, format_metrics_table
from src.predict import save_model, load_model, predict_labels

df = clean_dataframe(load_dataset())
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)

models = train_all(X_train, y_train)
print(format_metrics_table(evaluate_models(models, X_test, y_test)))

save_model(models["Logistic Regression + CountVectorizer"])
print(predict_labels(load_model(), X_test.head()))
```

Tests:

    pytest

## Results

Held-out test set (80/20 stratified split, random_state=42):

| Model | Vectorizer | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Dummy Classifier | Count | 86.64% | 43.32% | 50.00% | 46.42% |
| Logistic Regression | Count | 98.39% | 99.09% | 93.96% | 96.32% |
| Multinomial Naive Bayes | Count | 98.03% | 97.31% | 94.04% | 95.59% |
| Random Forest | TF-IDF | 97.67% | 98.69% | 91.28% | 94.56% |

(best vectorizer per model; metrics are macro averages)

- Winner: **Logistic Regression + CountVectorizer — 98.39% accuracy, 96.32% F1**, and the best spam-catching profile (99.1% precision, 94.0% recall)
- CountVectorizer beat TF-IDF in 2 of 3 models — see [Why CountVectorizer beats TF-IDF here](#why-countvectorizer-beats-tfidf-here)
- Dummy baseline: 86.6% — it predicts "ham" every time, confirming the class imbalance; every real model beats it by ~12 points
- On the spam class specifically: 100% precision, 87.9% recall — 0 legitimate messages flagged, 18 of 149 spam messages missed (18 of 1,115 test messages total)

### Why CountVectorizer beats TF-IDF here

- SMS spam is short and keyword-driven: *free*, *win*, *claim* repeat heavily in spam and rarely in ham, so raw term counts already separate the classes.
- TF-IDF down-weights exactly those repeated keywords, so it loses recall (86.5-86.9% vs 94.0% for Count) without buying precision back.
- Count wins in 2 of 3 models (Logistic Regression and Naive Bayes); only Random Forest prefers TF-IDF.
- On longer, more varied text TF-IDF usually wins — here it is a real finding, not an error.
