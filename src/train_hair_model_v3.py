import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/hair_v3")
MODEL_DIR = Path("models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 12

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("HAIR CLASSIFIER V3")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print()


# ============================================================
# LOAD DATASETS
# ============================================================

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = train_ds.class_names

print()
print("Class names:", class_names)

with open(
    MODEL_DIR / "hair_v3_class_names.json",
    "w"
) as f:
    json.dump(class_names, f, indent=2)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
    ],
    name="augmentation",
)


# ============================================================
# MOBILENETV2
# ============================================================

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = keras.Input(
    shape=IMG_SIZE + (3,)
)

x = augmentation(inputs)

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
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs,
    outputs
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = (
    MODEL_DIR /
    "hair_classifier_v3_best.keras"
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ============================================================
# PHASE 1 — TRANSFER LEARNING
# ============================================================

print()
print("=" * 70)
print("PHASE 1: TRANSFER LEARNING")
print("=" * 70)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=3e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# PHASE 2 — FINE TUNING
# ============================================================

print()
print("=" * 70)
print("PHASE 2: FINE-TUNING")
print("=" * 70)

base_model.trainable = True

# Freeze the earlier MobileNetV2 layers.
for layer in base_model.layers[:100]:
    layer.trainable = False

# Keep BatchNorm layers frozen.
for layer in base_model.layers:
    if isinstance(
        layer,
        layers.BatchNormalization
    ):
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
    ],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best V3 model...")

best_model = keras.models.load_model(
    checkpoint_path
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("V3 TEST EVALUATION")
print("=" * 70)

results = best_model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(
    best_model.metrics_names,
    results
):
    print(
        f"{name:>12}: {value:.4f}"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_path = (
    MODEL_DIR /
    "hair_classifier_v3_final.keras"
)

best_model.save(final_path)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Best model :", checkpoint_path)
print("Final model:", final_path)
print("Classes    :", class_names)