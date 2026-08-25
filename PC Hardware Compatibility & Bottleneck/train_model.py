"""
===============================================================
AI-Powered PC Hardware Compatibility & Bottleneck Analyzer
===============================================================
train_model.py - Multi-Task Deep Learning Model Training Script

Architecture:
 - Embedding layers for all categorical inputs
 - Shared Dense backbone
 - Three output branches (Binary Classification, Regression, Multi-Class)

Author: Senior Deep Learning Engineer
Course: ANN & Deep Learning Final Project
===============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Dense, Concatenate,
    BatchNormalization, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, Precision, Recall

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------
DATASET_PATH   = "dataset.csv"
MODEL_PATH     = "hardware_model.keras"
ENCODERS_PKL   = "encoders.pkl"
LABEL_ENC_PKL  = "label_encoders.pkl"
TARGET_ENC_PKL = "target_encoders.pkl"
META_JSON      = "model_meta.json"
PLOTS_DIR      = "plots"
EMBED_DIM      = 8       # Embedding dimension for each categorical feature
BATCH_SIZE     = 32
EPOCHS         = 100
PATIENCE       = 10
SEED           = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print(" AI-Powered PC Hardware Compatibility & Bottleneck Analyzer")
print(" Model Training Script")
print("=" * 60)

# ---------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------
print("\n[1/9] Loading dataset...")
df = pd.read_csv("PC Hardware Compatibility & Bottleneck Dataset - PC Hardware Compatibility & Bottleneck Dataset.csv")
print(f"    Dataset shape: {df.shape}")
print(f"    Columns: {df.columns.tolist()}")

# ---------------------------------------------------------------
# 2. REMOVE BUILD_ID
# ---------------------------------------------------------------
print("\n[2/9] Removing Build_ID column...")
df.drop(columns=["Build_ID"], inplace=True, errors="ignore")

# ---------------------------------------------------------------
# 3. HANDLE MESSY / MISSING VALUES
# ---------------------------------------------------------------
print("\n[3/9] Handling missing values and data quality issues...")

# --- Is_Compatible ---
# Values: 'Yes', 'No', 'No (Socket Mismatch)', '0%'
# Normalize to binary: 'Yes' → 1, everything else → 0
def normalize_compatible(val):
    if isinstance(val, str) and val.strip().lower() == "yes":
        return "Yes"
    return "No"

df["Is_Compatible"] = df["Is_Compatible"].apply(normalize_compatible)
print(f"    Is_Compatible distribution:\n{df['Is_Compatible'].value_counts().to_string()}")

# --- Bottleneck_Percentage ---
# Values: '5%', '22%', 'Incompatible Motherboard' etc.
# Strategy: extract numeric; non-numeric → 0.0
def parse_bottleneck(val):
    if isinstance(val, str):
        val = val.strip().replace("%", "")
        try:
            return float(val)
        except ValueError:
            return 0.0
    return float(val) if pd.notnull(val) else 0.0

df["Bottleneck_Percentage"] = df["Bottleneck_Percentage"].apply(parse_bottleneck)
print(f"    Bottleneck_Percentage stats: min={df['Bottleneck_Percentage'].min():.1f}, "
      f"max={df['Bottleneck_Percentage'].max():.1f}, mean={df['Bottleneck_Percentage'].mean():.1f}")

# --- Primary_Bottleneck_Component ---
# 8 rows are NaN; fill them with 'General / Low'
df["Primary_Bottleneck_Component"].fillna("General / Low", inplace=True)
print(f"    Primary_Bottleneck_Component distribution:\n"
      f"{df['Primary_Bottleneck_Component'].value_counts().to_string()}")

# General null sweep for any remaining NaN in object columns
for col in df.select_dtypes(include="object").columns:
    df[col].fillna("Unknown", inplace=True)

# ---------------------------------------------------------------
# 4. LABEL ENCODE CATEGORICAL INPUT FEATURES
# ---------------------------------------------------------------
print("\n[4/9] Label encoding categorical inputs...")

INPUT_CAT_COLS = [
    "CPU_Model", "GPU_Model", "Motherboard_Model",
    "RAM_Capacity", "Target_Resolution"
]

# Store individual encoders so the Streamlit app can rebuild dropdowns
input_encoders = {}  # col → LabelEncoder
for col in INPUT_CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    input_encoders[col] = le
    print(f"    {col}: {len(le.classes_)} classes → indices 0..{len(le.classes_)-1}")

# ---------------------------------------------------------------
# 5. ENCODE TARGET FEATURES
# ---------------------------------------------------------------
print("\n[5/9] Encoding target features...")

# Binary target: Is_Compatible
le_compat = LabelEncoder()
df["Is_Compatible_enc"] = le_compat.fit_transform(df["Is_Compatible"])
print(f"    Is_Compatible classes: {le_compat.classes_}  →  indices {le_compat.transform(le_compat.classes_)}")

# Multi-class target: Primary_Bottleneck_Component
le_component = LabelEncoder()
df["Component_enc"] = le_component.fit_transform(df["Primary_Bottleneck_Component"].astype(str))
n_component_classes = len(le_component.classes_)
print(f"    Primary_Bottleneck_Component classes ({n_component_classes}): {le_component.classes_}")

# ---------------------------------------------------------------
# 6. SAVE ENCODERS
# ---------------------------------------------------------------
print("\n[6/9] Saving encoders...")

# encoders.pkl → input feature encoders (for Streamlit dropdowns)
joblib.dump(input_encoders, ENCODERS_PKL)
print(f"    Saved input encoders → {ENCODERS_PKL}")

# label_encoders.pkl → same as above, also store for completeness
joblib.dump(input_encoders, LABEL_ENC_PKL)
print(f"    Saved label encoders → {LABEL_ENC_PKL}")

# target_encoders.pkl → compatibility + component encoders
target_encoders = {
    "Is_Compatible": le_compat,
    "Primary_Bottleneck_Component": le_component
}
joblib.dump(target_encoders, TARGET_ENC_PKL)
print(f"    Saved target encoders → {TARGET_ENC_PKL}")

# Persist metadata (class counts) for inference-time reconstruction
meta = {
    "n_component_classes": n_component_classes,
    "input_vocab_sizes": {col: int(len(enc.classes_)) for col, enc in input_encoders.items()}
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)
print(f"    Saved model metadata → {META_JSON}")

# ---------------------------------------------------------------
# 7. PREPARE FEATURES AND TARGETS; SPLIT DATA
# ---------------------------------------------------------------
print("\n[7/9] Preparing features & splitting data (80 / 20)...")

# Inputs: five encoded integer columns
X = {
    "cpu_input":    df["CPU_Model"].values,
    "gpu_input":    df["GPU_Model"].values,
    "mb_input":     df["Motherboard_Model"].values,
    "ram_input":    df["RAM_Capacity"].values,
    "res_input":    df["Target_Resolution"].values,
}

# Targets
y_compat    = df["Is_Compatible_enc"].values.reshape(-1, 1)
y_bottleneck = df["Bottleneck_Percentage"].values.reshape(-1, 1)
y_component_cat = to_categorical(df["Component_enc"].values, num_classes=n_component_classes)

# Split indices
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=df["Is_Compatible_enc"])

def split_dict(d, idx):
    return {k: v[idx] for k, v in d.items()}

X_train = split_dict(X, train_idx)
X_test  = split_dict(X, test_idx)

y_compat_train,     y_compat_test     = y_compat[train_idx],     y_compat[test_idx]
y_bottleneck_train, y_bottleneck_test = y_bottleneck[train_idx], y_bottleneck[test_idx]
y_comp_train,       y_comp_test       = y_component_cat[train_idx], y_component_cat[test_idx]

print(f"    Training samples : {len(train_idx)}")
print(f"    Testing  samples : {len(test_idx)}")

# ---------------------------------------------------------------
# 8. BUILD MULTI-TASK NEURAL NETWORK (Keras Functional API)
# ---------------------------------------------------------------
print("\n[8/9] Building Multi-Task ANN architecture...")

vocab = meta["input_vocab_sizes"]

# --- Five categorical inputs, each goes through an Embedding layer ---
def build_embedding_branch(name, vocab_size, embed_dim=EMBED_DIM):
    """Create Input → Embedding → Flatten sub-graph."""
    inp = Input(shape=(1,), name=name)
    emb = Embedding(input_dim=vocab_size + 1,  # +1 for safety
                    output_dim=embed_dim,
                    name=f"{name}_emb")(inp)
    flat = Flatten(name=f"{name}_flat")(emb)
    return inp, flat

cpu_inp,    cpu_flat    = build_embedding_branch("cpu_input",    vocab["CPU_Model"])
gpu_inp,    gpu_flat    = build_embedding_branch("gpu_input",    vocab["GPU_Model"])
mb_inp,     mb_flat     = build_embedding_branch("mb_input",     vocab["Motherboard_Model"])
ram_inp,    ram_flat    = build_embedding_branch("ram_input",    vocab["RAM_Capacity"])
res_inp,    res_flat    = build_embedding_branch("res_input",    vocab["Target_Resolution"])

# --- Concatenate all embedding outputs ---
merged = Concatenate(name="merged")([cpu_flat, gpu_flat, mb_flat, ram_flat, res_flat])

# --- Shared Dense Backbone ---
x = Dense(256, activation="relu", name="shared_dense1")(merged)
x = BatchNormalization(name="bn1")(x)
x = Dropout(0.3, name="drop1")(x)

x = Dense(128, activation="relu", name="shared_dense2")(x)
x = BatchNormalization(name="bn2")(x)
x = Dropout(0.3, name="drop2")(x)

x = Dense(64, activation="relu", name="shared_dense3")(x)
x = Dense(32, activation="relu", name="shared_dense4")(x)

# --- Output Branch 1: Compatibility (Binary Classification) ---
out_compat = Dense(1, activation="sigmoid", name="compatibility_output")(x)

# --- Output Branch 2: Bottleneck Percentage (Regression) ---
out_bottleneck = Dense(1, activation="linear", name="bottleneck_output")(x)

# --- Output Branch 3: Primary Bottleneck Component (Multi-Class) ---
out_component = Dense(n_component_classes, activation="softmax", name="component_output")(x)

# --- Assemble Model ---
model = Model(
    inputs=[cpu_inp, gpu_inp, mb_inp, ram_inp, res_inp],
    outputs=[out_compat, out_bottleneck, out_component],
    name="PC_HW_MultiTask_ANN"
)

model.summary()

# ---------------------------------------------------------------
# COMPILE
# ---------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "compatibility_output": "binary_crossentropy",
        "bottleneck_output":    "mean_squared_error",
        "component_output":     "categorical_crossentropy",
    },
    loss_weights={
        "compatibility_output": 1.0,
        "bottleneck_output":    0.5,   # Scale regression loss relative to CE losses
        "component_output":     1.0,
    },
    metrics={
        "compatibility_output": [
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
        "bottleneck_output": [
            "mae",
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
        "component_output": [
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    }
)

# ---------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
]

# ---------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------
print("\n[9/9] Training model (up to {} epochs, patience={})...".format(EPOCHS, PATIENCE))

history = model.fit(
    X_train,
    {
        "compatibility_output": y_compat_train,
        "bottleneck_output":    y_bottleneck_train,
        "component_output":     y_comp_train,
    },
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

print("\n✓ Training complete. Best model saved to:", MODEL_PATH)

# ---------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print(" EVALUATION")
print("=" * 60)

# Raw predictions
preds = model.predict(X_test, verbose=0)
pred_compat_prob  = preds[0].flatten()           # sigmoid probabilities
pred_bottleneck   = preds[1].flatten()           # regression
pred_component_p  = preds[2]                     # softmax probabilities

# Convert to hard labels
pred_compat_bin   = (pred_compat_prob >= 0.5).astype(int)
pred_component_cls = np.argmax(pred_component_p, axis=1)

# Ground truth
true_compat    = y_compat_test.flatten().astype(int)
true_bottleneck = y_bottleneck_test.flatten()
true_component  = np.argmax(y_comp_test, axis=1)

# --- Task 1: Compatibility ---
print("\n--- Task 1: Hardware Compatibility (Binary Classification) ---")
print(f"Accuracy  : {accuracy_score(true_compat, pred_compat_bin):.4f}")
print(f"Precision : {precision_score(true_compat, pred_compat_bin, zero_division=0):.4f}")
print(f"Recall    : {recall_score(true_compat, pred_compat_bin, zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(true_compat, pred_compat_bin, zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(true_compat, pred_compat_bin,
                             target_names=le_compat.classes_, zero_division=0))

# --- Task 2: Bottleneck Percentage ---
print("\n--- Task 2: Bottleneck Percentage (Regression) ---")
mae  = mean_absolute_error(true_bottleneck, pred_bottleneck)
rmse = np.sqrt(mean_squared_error(true_bottleneck, pred_bottleneck))
r2   = r2_score(true_bottleneck, pred_bottleneck)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# --- Task 3: Bottleneck Component ---
print("\n--- Task 3: Primary Bottleneck Component (Multi-Class) ---")
print(f"Accuracy  : {accuracy_score(true_component, pred_component_cls):.4f}")
print(f"Precision : {precision_score(true_component, pred_component_cls, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(true_component, pred_component_cls, average='weighted', zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(true_component, pred_component_cls, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(true_component, pred_component_cls,
                             target_names=le_component.classes_, zero_division=0))

# ---------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------
print("\nGenerating and saving plots...")

# Helper: clean history keys (they may have prefix from multi-output)
def get_hist(history, keys):
    """Find first matching key in history.history dict."""
    h = history.history
    for k in keys:
        if k in h:
            return np.array(h[k])
    return None

# 1. Compatibility Accuracy Graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 1 – Compatibility: Accuracy & Loss", fontsize=14, fontweight="bold")

acc_train = get_hist(history, ["compatibility_output_accuracy", "accuracy"])
acc_val   = get_hist(history, ["val_compatibility_output_accuracy", "val_accuracy"])
if acc_train is not None:
    axes[0].plot(acc_train, label="Train Accuracy", color="royalblue")
    axes[0].plot(acc_val,   label="Val Accuracy",   color="orange")
    axes[0].set_title("Accuracy Curve")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

loss_train = get_hist(history, ["compatibility_output_loss", "loss"])
loss_val   = get_hist(history, ["val_compatibility_output_loss", "val_loss"])
if loss_train is not None:
    axes[1].plot(loss_train, label="Train Loss", color="royalblue")
    axes[1].plot(loss_val,   label="Val Loss",   color="orange")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR}/accuracy.png")

# 2. Combined Training Loss Graph
fig, ax = plt.subplots(figsize=(10, 5))
total_loss     = get_hist(history, ["loss"])
total_loss_val = get_hist(history, ["val_loss"])
if total_loss is not None:
    ax.plot(total_loss,     label="Total Train Loss", color="royalblue", linewidth=2)
    ax.plot(total_loss_val, label="Total Val Loss",   color="orange",    linewidth=2)
ax.set_title("Combined Multi-Task Training Loss", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "loss.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR}/loss.png")

# 3. Confusion Matrix – Compatibility
fig, ax = plt.subplots(figsize=(7, 5))
cm_compat = confusion_matrix(true_compat, pred_compat_bin)
sns.heatmap(cm_compat, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_compat.classes_,
            yticklabels=le_compat.classes_, ax=ax)
ax.set_title("Confusion Matrix – Hardware Compatibility", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR}/confusion_matrix.png")

# 4. Confusion Matrix – Bottleneck Component
fig, ax = plt.subplots(figsize=(8, 6))
cm_comp = confusion_matrix(true_component, pred_component_cls)
sns.heatmap(cm_comp, annot=True, fmt="d", cmap="Greens",
            xticklabels=le_component.classes_,
            yticklabels=le_component.classes_, ax=ax)
ax.set_title("Confusion Matrix – Primary Bottleneck Component", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_component.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR}/confusion_matrix_component.png")

# 5. Prediction Distribution – Compatibility
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(pred_compat_prob[true_compat == 1], bins=30, alpha=0.7, label="Compatible (True=1)", color="green")
ax.hist(pred_compat_prob[true_compat == 0], bins=30, alpha=0.7, label="Incompatible (True=0)", color="red")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Boundary (0.5)")
ax.set_title("Prediction Distribution – Compatibility Sigmoid Output", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted Probability of Compatible"); ax.set_ylabel("Count")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "prediction_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR}/prediction_distribution.png")

# 6. Bottleneck Percentage Error Plot (Actual vs Predicted)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(true_bottleneck, pred_bottleneck, alpha=0.4, color="steelblue", s=15)
lims = [min(true_bottleneck.min(), pred_bottleneck.min()) - 2,
        max(true_bottleneck.max(), pred_bottleneck.max()) + 2]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_title("Actual vs Predicted – Bottleneck %", fontweight="bold")
axes[0].set_xlabel("Actual Bottleneck %"); axes[0].set_ylabel("Predicted Bottleneck %")
axes[0].legend(); axes[0].grid(alpha=0.3)

residuals = pred_bottleneck - true_bottleneck
axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Residuals Distribution – Bottleneck %", fontweight="bold")
axes[1].set_xlabel("Prediction Error (Predicted − Actual)"); axes[1].set_ylabel("Count")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "bottleneck_error_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR}/bottleneck_error_plot.png")

print("\n" + "=" * 60)
print(" ✓ All training steps complete!")
print(f"   Model  : {MODEL_PATH}")
print(f"   Plots  : {PLOTS_DIR}/")
print("=" * 60)