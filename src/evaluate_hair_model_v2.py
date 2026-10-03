import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(PROJECT_ROOT, "data", "hair_split", "test")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hair_classifier_v2_best.keras")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "hair_class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("V2 HAIR CLASSIFIER - TEST EVALUATION")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found:\n{TEST_DIR}")

print("\nModel:")
print(MODEL_PATH)

print("\nTest directory:")
print(TEST_DIR)

# ============================================================
# LOAD CLASS NAMES
# ============================================================

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
else:
    class_names = ["long", "short"]

print("\nClass names:")
print(class_names)

# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading untouched test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

print("\nDataset classes:")
print(test_ds.class_names)

# ============================================================
# OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")
print("This may take several minutes on CPU.\n")

y_true = []
y_prob = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())

    # Binary sigmoid output
    if predictions.shape[-1] == 1:
        y_prob.extend(predictions.reshape(-1))
    else:
        # If model outputs 2 probabilities
        y_prob.extend(predictions[:, 1])

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Threshold = 0.5
y_pred = (y_prob >= 0.5).astype(int)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

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

cm = confusion_matrix(y_true, y_pred)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V2 TEST RESULTS")
print("=" * 70)

print(f"\nTest images      : {len(y_true):,}")
print(f"Accuracy         : {accuracy * 100:.2f}%")
print(f"Macro Precision  : {precision * 100:.2f}%")
print(f"Macro Recall     : {recall * 100:.2f}%")
print(f"Macro F1         : {f1 * 100:.2f}%")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(f"{'':15} {'Pred Long':>12} {'Pred Short':>12}")

print(
    f"{'Actual Long':15} "
    f"{cm[0,0]:>12} "
    f"{cm[0,1]:>12}"
)

print(
    f"{'Actual Short':15} "
    f"{cm[1,0]:>12} "
    f"{cm[1,1]:>12}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("=" * 70)
print("ACTUAL / PREDICTED DISTRIBUTION")
print("=" * 70)

for i, class_name in enumerate(class_names):

    actual_count = np.sum(y_true == i)
    predicted_count = np.sum(y_pred == i)

    print(
        f"\n{class_name.upper():8}"
        f" Actual: {actual_count:5,}"
        f" | Predicted: {predicted_count:5,}"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "hair_classifier_v2_best.keras",
    "test_images": int(len(y_true)),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "confusion_matrix": cm.tolist()
}

results_path = os.path.join(
    PROJECT_ROOT,
    "models",
    "hair_v2_test_results.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print(results_path)

print("\n" + "=" * 70)
print("V2 TEST EVALUATION COMPLETE")
print("=" * 70)