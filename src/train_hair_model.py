from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE / "data" / "hair_split" / "train"
VAL_DIR = BASE / "data" / "hair_split" / "val"

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "hair_classifier_best.keras"
FINAL_MODEL = MODEL_DIR / "hair_classifier_final.keras"
CLASS_NAMES_FILE = MODEL_DIR / "hair_class_names.json"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("TensorFlow / GPU information")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

if tf.config.list_physical_devices("GPU"):
    print("GPU available")
else:
    print("GPU not available - training will use CPU")


# ============================================================
# LOAD DATASETS
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

print()
print("Class names:", class_names)

# Save class names
with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print("Saved class names:", CLASS_NAMES_FILE)


# ============================================================
# COUNT TRAINING SAMPLES
# ============================================================

short_count = len(list((TRAIN_DIR / "short").glob("*.jpg")))
long_count = len(list((TRAIN_DIR / "long").glob("*.jpg")))

print()
print("Training images:")
print("Short:", short_count)
print("Long :", long_count)


# ============================================================
# CLASS WEIGHTS
# ============================================================

labels = np.array(
    [0] * short_count +
    [1] * long_count
)

classes = np.unique(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = {
    int(classes[i]): float(weights[i])
    for i in range(len(classes))
}

print()
print("Class weights:")
print(class_weights)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# BASE MODEL
# ============================================================

print()
print("=" * 60)
print("Building MobileNetV2 model...")
print("=" * 60)

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = keras.Input(
    shape=IMG_SIZE + (3,),
    name="image"
)

x = data_augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="hair_probability"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="HairLengthMobileNetV2"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
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
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 60)
print("Saving final model...")
print("=" * 60)

model.save(FINAL_MODEL)

print("Final model:", FINAL_MODEL)
print("Best model :", BEST_MODEL)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

results = model.evaluate(
    val_ds,
    verbose=1
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)