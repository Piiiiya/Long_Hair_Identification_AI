from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "hairmony_4class_split"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "hair_classifier_4class_best.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

CLASS_NAMES = [
    "bald",
    "short",
    "medium",
    "long"
]

SEED = 42


# ============================================================
# START
# ============================================================

print("=" * 70)
print("HAIRMONY 4-CLASS MODEL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


print("\nModel:")
print(MODEL_PATH)


# ============================================================
# TEST DATASET
# ============================================================

TEST_DIR = DATA_DIR / "test"

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )


print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="int",

    class_names=CLASS_NAMES,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False,

    seed=SEED
)


print("\nClasses:")
print(test_ds.class_names)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best model...")

best_model = keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "preprocess_input": preprocess_input
    },

    compile=False
)


print("Model loaded successfully.")


# ============================================================
# COMPILE FOR EVALUATION
# ============================================================

best_model.compile(

    optimizer=keras.optimizers.Adam(),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)


print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []
y_prob = []


for images, labels in test_ds:

    probabilities = best_model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predictions
    )

    y_prob.extend(
        probabilities
    )


y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_prob = np.array(y_prob)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report_text = classification_report(

    y_true,

    y_pred,

    target_names=[
        name.upper()
        for name in CLASS_NAMES
    ],

    digits=4
)

print(report_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    "             "
    + " ".join(
        f"{name.upper():>8}"
        for name in CLASS_NAMES
    )
)

for index, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[index].upper():>8} "
        + " ".join(
            f"{value:8d}"
            for value in row
        )
    )


# ============================================================
# MEAN CONFIDENCE
# ============================================================

confidence = np.max(
    y_prob,
    axis=1
)

print(
    f"\nMean prediction confidence: "
    f"{confidence.mean() * 100:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "classes": CLASS_NAMES,

    "test_accuracy":
        float(test_accuracy),

    "test_loss":
        float(test_loss),

    "mean_confidence":
        float(confidence.mean()),

    "confusion_matrix":
        cm.tolist()

}


results_path = (
    MODEL_DIR
    / "hair_4class_test_results.json"
)


with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nResults:")
print(results_path)

print("\nDONE")
print("=" * 70)