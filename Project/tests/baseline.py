import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
)

DATA_ROOT = "../data/dev_phase/subtask1"
TRAIN_FILE = os.path.join(DATA_ROOT, "train", "eng.csv")

TEXT_COL = "text"
LABEL_COL = "polarization"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_split_data():
    df = pd.read_csv(TRAIN_FILE)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    print("Confusion matrix:")
    print(cm_df)


def evaluate_model(y_true, y_pred, model_name: str):
    print("\n" + "=" * 60)
    print(f"RESULTS FOR: {model_name}")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Per-class precision / recall / F1:")
    for label, p, r, f, s in zip([0, 1], precision, recall, f1, support):
        print(
            f"  Label {label}: "
            f"precision={p:.4f}, recall={r:.4f}, f1={f:.4f}, support={s}"
        )

    print()
    print_confusion_matrix(y_true, y_pred)

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, digits=4))


def run_naive_bayes(X_train, X_test, y_train, y_test):
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)

    y_pred = nb.predict(X_test_vec)
    evaluate_model(y_test, y_pred, "Multinomial Naive Bayes (CountVectorizer)")


def run_logistic_regression(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logreg = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    logreg.fit(X_train_vec, y_train)

    y_pred = logreg.predict(X_test_vec)
    evaluate_model(y_test, y_pred, "Logistic Regression (TF-IDF)")


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    run_naive_bayes(X_train, X_test, y_train, y_test)
    run_logistic_regression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
