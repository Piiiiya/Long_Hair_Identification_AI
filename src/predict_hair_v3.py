import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

MODEL_PATH = Path("models/hair_classifier_v3_best.keras")
IMG_SIZE = (224, 224)

CLASS_NAMES = ["long", "short"]


def predict(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    model = keras.models.load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Same preprocessing used during V3 training
    image_array = keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    probability = float(model.predict(image_array, verbose=0)[0][0])

    # V3 uses:
    # 0 = long
    # 1 = short
    if probability >= 0.5:
        predicted_class = "short"
        confidence = probability
    else:
        predicted_class = "long"
        confidence = 1.0 - probability

    long_probability = 1.0 - probability
    short_probability = probability

    print()
    print("=" * 60)
    print("HAIR CLASSIFIER V3 — REAL IMAGE TEST")
    print("=" * 60)
    print(f"Image       : {image_path}")
    print(f"Prediction  : {predicted_class.upper()}")
    print(f"Confidence  : {confidence * 100:.2f}%")
    print()
    print(f"Long        : {long_probability * 100:.2f}%")
    print(f"Short       : {short_probability * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print()
        print("Usage:")
        print(
            r"python src\predict_hair_v3.py "
            r'"path\to\image.jpg"'
        )
        print()
        sys.exit(1)

    predict(sys.argv[1])