from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE / "data" / "hair_split" / "train"
VAL_DIR = BASE / "data" / "hair_split" / "val"

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "hair_classifier_v2_best.keras"
FINAL_MODEL = MODEL_DIR / "hair_classifier_v2_final.keras"

CLASS_NAMES_FILE = MODEL_DIR / "hair_class_names.json"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 15


# ============================================================
# GPU
# ============================================================

print("=" * 60)
print("HAIR CLASSIFIER V2")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

if tf.config.list_physical_devices("GPU"):
    print("GPU available")
else:
    print("GPU not available - using CPU")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("Loading datasets...")
print("=" * 60)

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("Classes:", class_names)

with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# DATASET PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ],
    name="augmentation"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

short_count = len(list((TRAIN_DIR / "short").glob("*.jpg")))
long_count = len(list((TRAIN_DIR / "long").glob("*.jpg")))

total = short_count + long_count

# class_names is alphabetically:
# 0 = long
# 1 = short

# Give the minority LONG class additional importance.
long_weight = total / (2.0 * long_count)
short_weight = total / (2.0 * short_count)

# Additional emphasis on long hair.
long_weight *= 1.35

class_weights = {
    0: long_weight,
    1: short_weight
}

print()
print("Training distribution:")
print("Long :", long_count)
print("Short:", short_count)

print()
print("Class weights:")
print(class_weights)


# ============================================================
# BUILD MOBILENETV2
# ============================================================

print()
print("=" * 60)
print("Building MobileNetV2...")
print("=" * 60)

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


inputs = keras.Input(
    shape=IMG_SIZE + (3,),
    name="image"
)

x = augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.40)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="hair_probability"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="HairLengthMobileNetV2_V2"
)


# ============================================================
# CUSTOM F1 METRIC
# ============================================================

f1_metric = keras.metrics.F1Score(
    average="macro",
    threshold=0.5,
    name="f1"
)


# ============================================================
# INITIAL COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=3e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        f1_metric
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks_initial = [

    keras.callbacks.ModelCheckpoint(
        str(BEST_MODEL),
        monitor="val_f1",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_f1",
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


# ============================================================
# STAGE 1
# ============================================================

print()
print("=" * 60)
print("STAGE 1 - TRANSFER LEARNING")
print("=" * 60)

history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_initial
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print()
print("=" * 60)
print("STAGE 2 - FINE TUNING")
print("=" * 60)

base_model.trainable = True

# Freeze the early feature-extraction layers.
# Fine-tune only the upper part of MobileNetV2.

fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_from:]:
    layer.trainable = True

# Keep BatchNormalization frozen for more stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(
            average="macro",
            threshold=0.5,
            name="f1"
        )
    ]
)


callbacks_fine = [

    keras.callbacks.ModelCheckpoint(
        str(BEST_MODEL),
        monitor="val_f1",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=4,
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


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_fine
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 60)
print("Saving V2 model...")
print("=" * 60)

model.save(FINAL_MODEL)

print("Best V2 model :", BEST_MODEL)
print("Final V2 model:", FINAL_MODEL)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("V2 VALIDATION")
print("=" * 60)

results = model.evaluate(
    val_ds,
    verbose=1
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")


print()
print("=" * 60)
print("V2 TRAINING COMPLETE")
print("=" * 60)