import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

MARKET_TRAIN = Path("data/hair_split/train")
MARKET_VAL = Path("data/hair_split/val")
MARKET_TEST = Path("data/hair_split/test")

SYNTHETIC_TRAIN = Path("data/hair_v3/train")

V3_MODEL_PATH = Path("models/hair_classifier_v3_best.keras")

V4_BEST_PATH = Path("models/hair_classifier_v4_best.keras")
V4_FINAL_PATH = Path("models/hair_classifier_v4_final.keras")
V4_CLASS_NAMES = Path("models/hair_v4_class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

SEED = 42

# Repeat synthetic data to improve portrait diversity.
SYNTHETIC_REPEAT = 2


# ============================================================
# SETUP
# ============================================================

print("=" * 75)
print("HAIR CLASSIFIER V4 — CORRECTED REAL + SYNTHETIC TRAINING")
print("=" * 75)

print()
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 75)
print("LOADING DATASETS")
print("=" * 75)

market_train = keras.utils.image_dataset_from_directory(
    MARKET_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED,
)

market_val = keras.utils.image_dataset_from_directory(
    MARKET_VAL,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

market_test = keras.utils.image_dataset_from_directory(
    MARKET_TEST,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

synthetic_train = keras.utils.image_dataset_from_directory(
    SYNTHETIC_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED,
)

print()
print("Market classes   :", market_train.class_names)
print("Synthetic classes:", synthetic_train.class_names)


# ============================================================
# PREPROCESSING
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def preprocess(images, labels):
    images = tf.cast(images, tf.float32)

    images = keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    labels = tf.cast(labels, tf.float32)

    return images, labels


market_train = market_train.map(
    preprocess,
    num_parallel_calls=AUTOTUNE,
)

market_val = market_val.map(
    preprocess,
    num_parallel_calls=AUTOTUNE,
)

market_test = market_test.map(
    preprocess,
    num_parallel_calls=AUTOTUNE,
)

synthetic_train = synthetic_train.map(
    preprocess,
    num_parallel_calls=AUTOTUNE,
)


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

print()
print("=" * 75)
print("COMBINING DATA")
print("=" * 75)

synthetic_train = synthetic_train.repeat(
    SYNTHETIC_REPEAT
)

combined_train = market_train.concatenate(
    synthetic_train
)

# Smaller shuffle buffer to reduce RAM pressure.
combined_train = combined_train.shuffle(
    3000,
    seed=SEED,
    reshuffle_each_iteration=True,
)

combined_train = combined_train.prefetch(1)

market_val = market_val.prefetch(1)
market_test = market_test.prefetch(1)

print("Synthetic repeat factor:", SYNTHETIC_REPEAT)
print("Training data:")
print("  Market-1501 + synthetic portraits")


# ============================================================
# LOAD V3 MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING V3 MODEL")
print("=" * 75)

print("Loading:", V3_MODEL_PATH)

v3_model = keras.models.load_model(
    V3_MODEL_PATH
)

print("V3 model loaded successfully.")

print()
print("V3 architecture:")
v3_model.summary()


# ============================================================
# FIND MOBILE-NET BACKBONE
# ============================================================

print()
print("=" * 75)
print("LOCATING MOBILE-NET V2 BACKBONE")
print("=" * 75)

backbone = None

for layer in v3_model.layers:

    if isinstance(layer, keras.Model):

        try:
            if "mobilenetv2" in layer.name.lower():
                backbone = layer
                break
        except Exception:
            pass


if backbone is None:

    # Search recursively through nested layers.
    for layer in v3_model.layers:

        if hasattr(layer, "layers"):

            for sublayer in layer.layers:

                if isinstance(sublayer, keras.Model):

                    if "mobilenetv2" in sublayer.name.lower():
                        backbone = sublayer
                        break

        if backbone is not None:
            break


if backbone is None:

    raise RuntimeError(
        "Could not find MobileNetV2 backbone inside V3 model."
    )


print("Found backbone:", backbone.name)


# ============================================================
# PHASE 1
# TRAIN CLASSIFICATION HEAD
# ============================================================

print()
print("=" * 75)
print("PHASE 1 — TRAIN CLASSIFICATION HEAD")
print("=" * 75)


# Freeze everything first.
for layer in v3_model.layers:
    layer.trainable = False


# Explicitly freeze backbone.
backbone.trainable = False


# Find Dense layers.
dense_layers = []

for layer in v3_model.layers:

    if isinstance(layer, layers.Dense):
        dense_layers.append(layer)


print()
print("Dense layers found:", len(dense_layers))

for layer in dense_layers:
    print(
        "  ",
        layer.name,
        "units=",
        layer.units,
    )


# The final two Dense layers in the V3 classifier are the head.
# Make Dense layers trainable.
for layer in dense_layers:
    layer.trainable = True


trainable_count = sum(
    np.prod(variable.shape)
    for variable in v3_model.trainable_weights
)


print()
print(
    "Trainable parameters in Phase 1:",
    f"{trainable_count:,}",
)


if trainable_count == 0:

    raise RuntimeError(
        "ERROR: Phase 1 still has zero trainable parameters."
    )


v3_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=2e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        ),
    ],
)


callbacks_phase1 = [

    keras.callbacks.ModelCheckpoint(
        V4_BEST_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
]


history1 = v3_model.fit(
    combined_train,
    validation_data=market_val,
    epochs=6,
    callbacks=callbacks_phase1,
)


# ============================================================
# PHASE 2
# FINE-TUNE UPPER MOBILENETV2
# ============================================================

print()
print("=" * 75)
print("PHASE 2 — FINE-TUNING MOBILENETV2")
print("=" * 75)


# Unfreeze backbone.
backbone.trainable = True


# Freeze early layers.
FINE_TUNE_FROM = 100

for i, layer in enumerate(backbone.layers):

    if i < FINE_TUNE_FROM:
        layer.trainable = False

    else:
        layer.trainable = True


# Keep BatchNormalization frozen.
for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


# Keep classifier head trainable.
for layer in dense_layers:
    layer.trainable = True


trainable_count = sum(
    np.prod(variable.shape)
    for variable in v3_model.trainable_weights
)


print()
print(
    "Trainable parameters in Phase 2:",
    f"{trainable_count:,}",
)


v3_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        ),
    ],
)


callbacks_phase2 = [

    keras.callbacks.ModelCheckpoint(
        V4_BEST_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    ),
]


history2 = v3_model.fit(
    combined_train,
    validation_data=market_val,
    epochs=10,
    callbacks=callbacks_phase2,
)


# ============================================================
# SAVE V4
# ============================================================

print()
print("=" * 75)
print("SAVING V4 MODEL")
print("=" * 75)

v3_model.save(V4_FINAL_PATH)

with open(V4_CLASS_NAMES, "w") as f:
    json.dump(
        ["long", "short"],
        f,
        indent=4,
    )


print("Best model :", V4_BEST_PATH)
print("Final model:", V4_FINAL_PATH)
print("Classes    :", V4_CLASS_NAMES)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 75)
print("FINAL VALIDATION RESULTS")
print("=" * 75)

val_results = v3_model.evaluate(
    market_val,
    verbose=1,
)

for name, value in zip(
    v3_model.metrics_names,
    val_results,
):

    print(
        f"{name:12}: {value:.4f}"
    )


# ============================================================
# MARKET-1501 TEST
# ============================================================

print()
print("=" * 75)
print("MARKET-1501 TEST RESULTS")
print("=" * 75)

test_results = v3_model.evaluate(
    market_test,
    verbose=1,
)

for name, value in zip(
    v3_model.metrics_names,
    test_results,
):

    print(
        f"{name:12}: {value:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 75)
print("V4 TRAINING COMPLETE")
print("=" * 75)

print()
print("Created:")

print(
    " ",
    V4_BEST_PATH
)

print(
    " ",
    V4_FINAL_PATH
)

print(
    " ",
    V4_CLASS_NAMES
)

print()
print("Next step:")
print("Run the detailed V4 test evaluation.")
print()
print("=" * 75)