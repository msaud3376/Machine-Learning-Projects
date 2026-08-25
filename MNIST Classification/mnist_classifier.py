"""
MNIST Handwritten Digit Classifier
====================================
End-to-end pipeline: load the MNIST digit images -> preprocess/normalise
pixels -> reduce dimensionality with PCA -> train & compare several models
-> evaluate -> save the best model.

Expects two CSV files in the classic "MNIST in CSV" layout (each row =
1 label + 784 pixel values, no header):
    mnist_train.csv   (60,000 rows)
    mnist_test.csv    (10,000 rows)

This is the same underlying NIST data Kaggle's "Digit Recognizer"
competition uses. If you downloaded Kaggle's train.csv/test.csv instead,
note that Kaggle's test.csv has NO labels (it's for competition
submission only) — swap in mnist_test.csv (or split off part of
train.csv yourself) so this script has ground truth to evaluate against.

Usage:
    python mnist_classifier.py
    python mnist_classifier.py --train mnist_train.csv --test mnist_test.csv
"""

import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42
DEFAULT_TRAIN_PATH = r"C:\Users\DELL\My Everything\Internship\First 4 Weeks\Project 02\Dataset\mnist_train.csv"
DEFAULT_TEST_PATH = r"C:\Users\DELL\My Everything\Internship\First 4 Weeks\Project 02\Dataset\mnist_test.csv"
OUTPUT_DIR = "."
N_PCA_COMPONENTS = 100


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data(train_path: str, test_path: str):
    """Each CSV row is: label, pixel0, pixel1, ..., pixel783 (no header)."""
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)

    y_train, X_train = train.iloc[:, 0].values, train.iloc[:, 1:].values
    y_test, X_test = test.iloc[:, 0].values, test.iloc[:, 1:].values

    print(f"Train: {X_train.shape[0]:,} images, Test: {X_test.shape[0]:,} images, "
          f"{X_train.shape[1]} pixels each (28x28)")
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------
# 2. Preprocess: normalise pixel values, reduce dimensionality
# --------------------------------------------------------------------------
def preprocess(X_train, X_test, n_components=N_PCA_COMPONENTS):
    # Scale pixel intensities from [0, 255] to [0, 1] — puts every feature
    # on the same scale, which every model here (especially the distance-
    # based KNN and the gradient-based MLP) benefits from.
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 784 raw pixels are highly redundant (neighbouring pixels are
    # correlated, and the corners are almost always blank). PCA compresses
    # them to the directions of greatest variance, which speeds up every
    # model by 5-10x with only a small accuracy cost.
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: {X_train.shape[1]} pixels -> {n_components} components "
          f"({pca.explained_variance_ratio_.sum():.1%} of variance retained)")

    return X_train_pca, X_test_pca, pca


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=100, early_stopping=True,
            n_iter_no_change=10, random_state=RANDOM_STATE
        ),
    }


# --------------------------------------------------------------------------
# 4. Train & evaluate
# --------------------------------------------------------------------------
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    fitted = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "fit_time_s": fit_time,
            "predict_time_s": pred_time,
        })
        fitted[name] = model
        print(f"  {name}: fit={fit_time:.1f}s predict={pred_time:.1f}s "
              f"accuracy={results[-1]['accuracy']:.4f}")

    # 10,000 held-out test images is already a large, stable sample, so we
    # rank on test accuracy directly rather than adding expensive k-fold CV
    # on top (a single Random Forest fit alone takes ~2-3 minutes here).
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return results_df, fitted


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------
def plot_sample_digits(X_raw, y_raw, out_path, n=25):
    """Show a grid of raw (un-normalised) sample digits with their labels."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_raw), n, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X_raw[i].reshape(28, 28), cmap="gray")
        ax.set_title(str(y_raw[i]), fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample training digits")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(results_df, out_path):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    plot_df = results_df.set_index("model")[metrics]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set (10,000 images)")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(model, model_name, X_test, y_test, out_path):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format="d"
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_misclassified(model, model_name, X_test_pca, X_test_raw, y_test, out_path, n=15):
    y_pred = model.predict(X_test_pca)
    wrong_idx = np.where(y_pred != y_test)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    show_idx = rng.choice(wrong_idx, min(n, len(wrong_idx)), replace=False)

    cols = 5
    rows = int(np.ceil(len(show_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.2 * rows))
    for ax, i in zip(np.array(axes).ravel(), show_idx):
        ax.imshow(X_test_raw[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"true={y_test[i]} pred={y_pred[i]}", fontsize=9, color="crimson")
        ax.axis("off")
    for ax in np.array(axes).ravel()[len(show_idx):]:
        ax.axis("off")
    plt.suptitle(f"Misclassified examples — {model_name} "
                 f"({len(wrong_idx)}/{len(y_test)} wrong = {len(wrong_idx)/len(y_test):.1%} error rate)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train an MNIST digit classifier.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_PATH, help="Path to mnist_train.csv")
    parser.add_argument("--test", default=DEFAULT_TEST_PATH, help="Path to mnist_test.csv")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="Where to save plots/model")
    args = parser.parse_args()

    # 1. Load
    X_train_raw, y_train, X_test_raw, y_test = load_data(args.train, args.test)
    plot_sample_digits(X_train_raw, y_train, f"{args.outdir}/sample_digits.png")

    # 2. Preprocess (normalise + PCA)
    X_train, X_test, pca = preprocess(X_train_raw, X_test_raw)

    # 3. Train & compare candidate models
    print("\nTraining candidate models...")
    models = build_models()
    results_df, fitted_models = evaluate_models(models, X_train, X_test, y_train, y_test)

    print("\n=== Model comparison (sorted by accuracy) ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4. Pick the best model
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")
    print("\nPer-digit classification report on test set:")
    print(classification_report(y_test, best_model.predict(X_test), digits=4))

    # 5. Plots
    plot_model_comparison(results_df, f"{args.outdir}/model_comparison.png")
    plot_confusion_matrix(best_model, best_name, X_test, y_test, f"{args.outdir}/confusion_matrix.png")
    plot_misclassified(best_model, best_name, X_test, X_test_raw, y_test, f"{args.outdir}/misclassified.png")

    # 6. Save artifacts
    joblib.dump(best_model, f"{args.outdir}/mnist_classifier_model.joblib")
    joblib.dump(pca, f"{args.outdir}/pca_transformer.joblib")
    results_df.to_csv(f"{args.outdir}/model_comparison_results.csv", index=False)
    with open(f"{args.outdir}/summary.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "test_set_size": len(y_test),
            "n_pca_components": N_PCA_COMPONENTS,
            "metrics": results_df.iloc[0].to_dict(),
        }, f, indent=2, default=float)

    print(f"\nSaved: mnist_classifier_model.joblib, pca_transformer.joblib, "
          f"sample_digits.png, model_comparison.png, confusion_matrix.png, misclassified.png")


def predict_digit(pixel_array_784, model_path="mnist_classifier_model.joblib",
                   pca_path="pca_transformer.joblib") -> int:
    """Classify a single new 28x28 digit image.
    pixel_array_784: flat array/list of 784 pixel values (0-255)."""
    model = joblib.load(model_path)
    pca = joblib.load(pca_path)
    x = np.asarray(pixel_array_784, dtype=np.float32).reshape(1, -1) / 255.0
    x_pca = pca.transform(x)
    return int(model.predict(x_pca)[0])


if __name__ == "__main__":
    main()
