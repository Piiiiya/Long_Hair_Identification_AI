from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "hairmony_4class_split"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_CLASSES = 4

SEED = 42

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10

FINE_TUNE_FROM = 100

CLASS_NAMES = [
    "bald",
    "short",
    "medium",
    "long"
]


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("HAIRMONY 4-CLASS HAIR LENGTH CLASSIFIER")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("\nGPU detected:")

    for gpu in gpus:
        print(" ", gpu)

else:

    print("\nGPU detected: NO")
    print("Training will use CPU.")


print("\nTensorFlow version:", tf.__version__)


# ============================================================
# CHECK DATASET
# ============================================================

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n{directory}"
        )


print("\nDataset:")
print(DATA_DIR)

print("\nClasses:")

for class_name in CLASS_NAMES:

    print(
        f"  {class_name.upper():8s} -> "
        f"{TRAIN_DIR / class_name}"
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nDetected class names:")
print(train_ds.class_names)


# ============================================================
# DATA PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# COUNT TRAINING SAMPLES
# ============================================================

train_counts = {}

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = TRAIN_DIR / class_name

    count = len(
        list(
            class_dir.glob("*.png")
        )
    )

    train_counts[class_index] = count


print("\nTraining class counts:")

for index, class_name in enumerate(CLASS_NAMES):

    print(
        f"  {class_name.upper():8s}: "
        f"{train_counts[index]}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

y_train = []

for class_index, count in train_counts.items():

    y_train.extend(
        [class_index] * count
    )

y_train = np.array(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for index, class_name in enumerate(CLASS_NAMES):

    print(
        f"  {class_name.upper():8s}: "
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.15
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.15
        ),

    ],
    name="hair_augmentation"
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

base_model.trainable = False


inputs = keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


x = data_augmentation(inputs)

x = layers.Lambda(
    preprocess_input,
    name="mobilenet_preprocessing"
)(x)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.BatchNormalization()(x)


x = layers.Dropout(
    0.35
)(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.25
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="hair_class"
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="Hairmony_4Class_MobileNetV2"
)


# ============================================================
# PHASE 1
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 - FROZEN MOBILENETV2")
print("=" * 70)

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=3e-4
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)


best_model_path = (
    MODEL_DIR
    / "hair_classifier_4class_best.keras"
)


callbacks_phase1 = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

]


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=PHASE1_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase1

)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 - MOBILENETV2 FINE-TUNING")
print("=" * 70)


base_model.trainable = True


# Freeze earlier layers
for layer in base_model.layers[:FINE_TUNE_FROM]:

    layer.trainable = False


# Keep BatchNorm frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


trainable_count = sum(
    1
    for layer in model.layers
    if layer.trainable
)


print(
    "\nTrainable layers:",
    trainable_count
)


model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)


callbacks_phase2 = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

]


history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=PHASE2_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase2

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / "hair_classifier_4class_final.keras"
)


model.save(
    final_model_path
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = (
    MODEL_DIR
    / "hair_4class_names.json"
)


with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


best_model = keras.models.load_model(
    best_model_path
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

print("\nGenerating test predictions...")

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

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        name.upper()
        for name in CLASS_NAMES
    ],
    digits=4
)

print(report)


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
# SAVE TEST RESULTS
# ============================================================

results_path = (
    MODEL_DIR
    / "hair_4class_test_results.json"
)


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
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nBest model:"
)

print(best_model_path)

print(
    "\nFinal model:"
)

print(final_model_path)

print(
    "\nClass names:"
)

print(class_names_path)

print(
    "\nTest results:"
)

print(results_path)

print("\nClasses:")

for class_name in CLASS_NAMES:

    print(
        " -",
        class_name.upper()
    )

print("\nDONE")
print("=" * 70)