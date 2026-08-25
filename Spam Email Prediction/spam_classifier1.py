"""
Email Spam Classifier
======================
End-to-end pipeline: load data -> clean/preprocess text -> extract TF-IDF
features -> train & compare several models -> evaluate -> save the best
model.

Usage:
    python spam_classifier.py
    python spam_classifier.py --data "C:\\path\\to\\emails.csv"

By default it looks for the dataset at the path given below (edit DATA_PATH
or pass --data to point at your own copy of emails.csv). The CSV is expected
to have two columns: "text" (the raw email) and "spam" (1 = spam, 0 = ham).
"""

import argparse
import json
import re
import string
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
DEFAULT_DATA_PATH = r"C:\Users\DELL\My Everything\Internship\First 4 Weeks\Project 01\emails.csv\merged_emails.csv"
OUTPUT_DIR = "."


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "spam"]).drop_duplicates(subset=["text"])
    df["spam"] = df["spam"].astype(int)
    print(f"Loaded {len(df):,} emails "
          f"({df['spam'].sum():,} spam / {(df['spam'] == 0).sum():,} ham) "
          f"after dropping empty rows and duplicates.")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------------------------------
SUBJECT_RE = re.compile(r"^subject\s*:\s*", flags=re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
STOP_WORDS = ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, strip the leading 'Subject:' tag, remove punctuation/
    numbers, collapse whitespace, and drop stopwords / single-letter
    tokens (these emails were tokenised with spaces around punctuation,
    e.g. 'suqgestions .', so simple regex cleaning works well here)."""
    text = text.lower()
    text = SUBJECT_RE.sub("", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 3. Feature extraction + models
# --------------------------------------------------------------------------
def build_models():
    """Candidate models. LinearSVC has no predict_proba, so we use its
    decision_function for ROC-AUC instead."""
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def get_scores(model, X):
    """Return spam scores usable for ROC-AUC regardless of model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# --------------------------------------------------------------------------
# 4. Train, evaluate, compare
# --------------------------------------------------------------------------
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)

        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score),
            "cv_f1_mean": cv_f1,
            "fit_time_s": fit_time,
        })
        fitted[name] = model
        print(f"  trained {name} in {fit_time:.2f}s")

    # Rank by cross-validated F1 (computed on the training folds only) rather
    # than the single test-set F1 — this avoids picking a model that just
    # happened to get lucky on one particular train/test split.
    return pd.DataFrame(results).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True), fitted


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------
def plot_model_comparison(results_df, out_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_df = results_df.set_index("model")[metrics]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_and_roc(model, model_name, X_test, y_test, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title(f"Confusion Matrix — {model_name}")

    y_score = get_scores(model, X_test)
    RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[1], name=model_name)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axes[1].set_title(f"ROC Curve — {model_name}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_spam_features(vectorizer, model, out_path, top_n=20):
    """Only meaningful for linear models (coef_ available)."""
    if not hasattr(model, "coef_"):
        return
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_.ravel()
    top_idx = np.argsort(coefs)[-top_n:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(feature_names[top_idx], coefs[top_idx], color="crimson")
    ax.set_title(f"Top {top_n} words most associated with SPAM")
    ax.set_xlabel("Model weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train an email spam classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to emails.csv")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="Where to save plots/model")
    args = parser.parse_args()

    # 1. Load
    df = load_data(args.data)

    # 2. Preprocess
    df = preprocess(df)

    # 3. Split (stratified so both sets keep the ~24% spam ratio)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["spam"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["spam"]
    )

    # 4. TF-IDF feature extraction (uni+bigrams capture phrases like
    #    "click here" / "free offer" that single words miss)
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF matrix: {X_train.shape[0]} train rows x {X_train.shape[1]} features")

    # 5. Train & compare candidate models
    print("\nTraining candidate models...")
    models = build_models()
    results_df, fitted_models = evaluate_models(models, X_train, X_test, y_train, y_test)

    print("\n=== Model comparison (sorted by F1) ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 6. Pick the best model by F1 (balances catching spam vs. not
    #    flagging real mail — a good default for an imbalanced problem)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")
    print("\nClassification report on test set:")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["Ham", "Spam"]))

    # 7. Plots
    plot_model_comparison(results_df, f"{args.outdir}/model_comparison.png")
    plot_confusion_and_roc(best_model, best_name, X_test, y_test, f"{args.outdir}/best_model_eval.png")
    plot_top_spam_features(vectorizer, best_model, f"{args.outdir}/top_spam_words.png")

    # 8. Save artifacts
    joblib.dump(best_model, f"{args.outdir}/spam_classifier_model.joblib")
    joblib.dump(vectorizer, f"{args.outdir}/tfidf_vectorizer.joblib")
    results_df.to_csv(f"{args.outdir}/model_comparison_results.csv", index=False)
    with open(f"{args.outdir}/summary.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "test_set_size": len(y_test),
            "metrics": results_df.iloc[0].to_dict(),
        }, f, indent=2, default=float)

    print(f"\nSaved: spam_classifier_model.joblib, tfidf_vectorizer.joblib, "
          f"model_comparison.png, best_model_eval.png, top_spam_words.png")


def predict_email(text: str, model_path="spam_classifier_model.joblib",
                   vectorizer_path="tfidf_vectorizer.joblib") -> str:
    """Helper to classify a brand-new email once the model is trained/saved."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return "SPAM" if pred == 1 else "NOT SPAM"


if __name__ == "__main__":
    main()
