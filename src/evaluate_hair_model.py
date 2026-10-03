from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

TEST_DIR = BASE / "data" / "hair_split" / "test"
MODEL_PATH = BASE / "models" / "hair_classifier_best.keras"
CLASS_NAMES_PATH = BASE / "models" / "hair_class_names.json"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("=" * 60)
print("HAIR CLASSIFIER TEST EVALUATION")
print("=" * 60)

print("Classes:", class_names)


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("Loading untouched test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", sum(1 for _ in TEST_DIR.glob("*/*.jpg")))


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded:")
print(MODEL_PATH)


# ============================================================
# MODEL EVALUATION
# ============================================================

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)

print()
print("Raw test results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")


# ============================================================
# COLLECT TRUE LABELS
# ============================================================

y_true = []

for _, labels in test_ds:
    y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true).astype(int)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("Generating predictions...")

probabilities = model.predict(
    test_ds,
    verbose=1
).flatten()

y_pred = (probabilities >= 0.5).astype(int)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print()
print("=" * 60)
print("TEST ACCURACY")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print()
print("Matrix interpretation:")

print(f"True {class_names[0]}, Predicted {class_names[0]} : {cm[0, 0]}")
print(f"True {class_names[0]}, Predicted {class_names[1]} : {cm[0, 1]}")
print(f"True {class_names[1]}, Predicted {class_names[0]} : {cm[1, 0]}")
print(f"True {class_names[1]}, Predicted {class_names[1]} : {cm[1, 1]}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("TEST CLASS DISTRIBUTION")
print("=" * 60)

for i, class_name in enumerate(class_names):
    count = np.sum(y_true == i)
    predicted_count = np.sum(y_pred == i)

    print(
        f"{class_name}: "
        f"actual={count}, "
        f"predicted={predicted_count}"
    )


# ============================================================
# CONFIDENCE INFORMATION
# ============================================================

confidence = np.maximum(
    probabilities,
    1 - probabilities
)

print()
print("=" * 60)
print("PREDICTION CONFIDENCE")
print("=" * 60)

print(f"Mean confidence : {np.mean(confidence):.4f}")
print(f"Minimum confidence : {np.min(confidence):.4f}")
print(f"Maximum confidence : {np.max(confidence):.4f}")


print()
print("=" * 60)
print("TEST EVALUATION COMPLETE")
print("=" * 60)