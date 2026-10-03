import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/hair_v3/test")
MODEL_PATH = Path("models/hair_classifier_v3_best.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HAIR CLASSIFIER V3 — DETAILED TEST EVALUATION")
print("=" * 70)

model = keras.models.load_model(MODEL_PATH)

print("Model:", MODEL_PATH)

# ============================================================
# LOAD TEST DATA
# ============================================================

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = test_ds.class_names

print("Classes:", class_names)

# ============================================================
# PREDICTIONS
# ============================================================

y_true = []
y_prob = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten().astype(int)
    )

    y_prob.extend(
        predictions.flatten()
    )

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Sigmoid output corresponds to class 1 = short
y_pred = (y_prob >= 0.5).astype(int)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Accuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Macro Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Macro F1       : {f1:.4f} ({f1 * 100:.2f}%)")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    f"{'':12} "
    f"{'long':>10} "
    f"{'short':>10}"
)

for i, name in enumerate(class_names):

    print(
        f"{name:12} "
        f"{cm[i, 0]:10d} "
        f"{cm[i, 1]:10d}"
    )

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for i, name in enumerate(class_names):

    actual_count = np.sum(y_true == i)
    predicted_count = np.sum(y_pred == i)

    print(
        f"{name:>5} | "
        f"actual: {actual_count:3d} | "
        f"predicted: {predicted_count:3d}"
    )

# ============================================================
# CONFIDENCE
# ============================================================

confidence = np.maximum(
    y_prob,
    1 - y_prob
)

print()
print("=" * 70)
print("CONFIDENCE")
print("=" * 70)

print(
    f"Mean confidence: "
    f"{np.mean(confidence) * 100:.2f}%"
)

print(
    f"Minimum confidence: "
    f"{np.min(confidence) * 100:.2f}%"
)

print(
    f"Maximum confidence: "
    f"{np.max(confidence) * 100:.2f}%"
)

# ============================================================
# MISCLASSIFICATIONS
# ============================================================

wrong = y_true != y_pred

print()
print("=" * 70)
print("MISCLASSIFICATIONS")
print("=" * 70)

print(
    "Incorrect predictions:",
    np.sum(wrong),
    "out of",
    len(y_true)
)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "class_names": class_names,
    "test_samples": int(len(y_true)),
    "incorrect_predictions": int(np.sum(wrong)),
    "mean_confidence": float(np.mean(confidence)),
}

output_path = Path(
    "models/hair_v3_test_results.json"
)

with open(output_path, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print()
print("Results saved to:")
print(output_path)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)