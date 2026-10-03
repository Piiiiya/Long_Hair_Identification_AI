import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "hair_classifier_v2_best.keras"
)

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hair_split",
    "val"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HAIR CLASSIFIER V2 - THRESHOLD ANALYSIS")
print("=" * 70)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

print("\nClasses:")
print(val_ds.class_names)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")

# ============================================================
# GET PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")
print("Please wait...\n")

y_true = []
y_prob = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    # V2 uses sigmoid binary output
    probabilities = predictions.reshape(-1)

    y_true.extend(
        labels.numpy()
    )

    y_prob.extend(
        probabilities
    )

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# ============================================================
# IMPORTANT
# ============================================================
#
# Model output:
# 0 = LONG
# 1 = SHORT
#
# Therefore:
# probability >= threshold = SHORT
#
# We calculate LONG probability as:
#
# long_probability = 1 - short_probability
# ============================================================

long_probability = 1.0 - y_prob

# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    f"\n{'Threshold':>10} "
    f"{'Accuracy':>12} "
    f"{'Long Recall':>14} "
    f"{'Short Recall':>15} "
    f"{'Macro F1':>12}"
)

print("-" * 70)

best_threshold = None
best_f1 = -1

for threshold in thresholds:

    # Long if long probability >= threshold
    y_pred = (
        long_probability >= threshold
    ).astype(int)

    # Convert:
    # 1 = LONG
    # 0 = SHORT
    #
    # Our actual labels are:
    # 0 = LONG
    # 1 = SHORT
    #
    # So convert prediction accordingly.

    y_pred_actual = 1 - y_pred

    accuracy = accuracy_score(
        y_true,
        y_pred_actual
    )

    macro_f1 = f1_score(
        y_true,
        y_pred_actual,
        average="macro",
        zero_division=0
    )

    # Recall for LONG = class 0
    recall_values = recall_score(
        y_true,
        y_pred_actual,
        average=None,
        labels=[0, 1],
        zero_division=0
    )

    long_recall = recall_values[0]
    short_recall = recall_values[1]

    print(
        f"{threshold:>10.2f} "
        f"{accuracy * 100:>11.2f}% "
        f"{long_recall * 100:>13.2f}% "
        f"{short_recall * 100:>14.2f}% "
        f"{macro_f1 * 100:>11.2f}%"
    )

    if macro_f1 > best_f1:

        best_f1 = macro_f1
        best_threshold = threshold

# ============================================================
# BEST THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION THRESHOLD")
print("=" * 70)

print(
    f"\nBest threshold: {best_threshold:.2f}"
)

print(
    f"Best validation Macro F1: "
    f"{best_f1 * 100:.2f}%"
)

print("\nInterpretation:")

print(
    "\nThe threshold controls how easily the model "
    "accepts LONG HAIR."
)

print(
    "A lower threshold generally makes LONG predictions "
    "easier, increasing long-hair recall but potentially "
    "creating more false LONG predictions."
)

print(
    "\nThe threshold must be selected using validation data."
)

print(
    "We should NOT choose it based on the test set."
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)