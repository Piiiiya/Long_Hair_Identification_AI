import os
import sys
import json
import numpy as np
import tensorflow as tf

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "hair_classifier_v2_best.keras"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "hair_class_names.json"
)

IMG_SIZE = (224, 224)

# ============================================================
# CHECK ARGUMENT
# ============================================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python src\\predict_hair.py \"path_to_image.jpg\"\n")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f"\nImage not found:\n{IMAGE_PATH}")
    sys.exit(1)

# ============================================================
# LOAD CLASS NAMES
# ============================================================

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
else:
    class_names = ["long", "short"]

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")

# ============================================================
# LOAD IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

image_array = tf.keras.utils.img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# MobileNetV2 preprocessing
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    image_array,
    verbose=0
)

if prediction.shape[-1] == 1:

    short_probability = float(prediction[0][0])

    if short_probability >= 0.5:
        predicted_index = 1
        confidence = short_probability
    else:
        predicted_index = 0
        confidence = 1.0 - short_probability

else:

    probabilities = prediction[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

# ============================================================
# RESULT
# ============================================================

predicted_class = class_names[predicted_index]

print("\n" + "=" * 60)
print("HAIR LENGTH PREDICTION")
print("=" * 60)

print(f"\nImage      : {IMAGE_PATH}")
print(f"Prediction : {predicted_class.upper()}")
print(f"Confidence : {confidence * 100:.2f}%")

print("\nClass probabilities:")

if prediction.shape[-1] == 1:

    print(
        f"LONG  : {(1 - short_probability) * 100:.2f}%"
    )

    print(
        f"SHORT : {short_probability * 100:.2f}%"
    )

else:

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name.upper():6} : "
            f"{probabilities[i] * 100:.2f}%"
        )

print("\n" + "=" * 60)