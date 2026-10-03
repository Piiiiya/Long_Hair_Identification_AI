from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hair_classifier_4class_best.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "BALD",
    "SHORT",
    "MEDIUM",
    "LONG"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hair Length Identification",
    page_icon="💇",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("💇 Hair Length Identification System")

st.write(
    "Upload a portrait image to classify visible hair "
    "into four categories."
)

st.caption(
    "Classes: BALD • SHORT • MEDIUM • LONG"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the hair classification model.")

    st.code(str(e))

    st.stop()


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Input Image",
            width="stretch"
        )


        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        resized_image = image.resize(
            IMG_SIZE,
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        image_array = preprocess_input(
            image_array
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = (
            CLASS_NAMES[predicted_index]
        )

        confidence = (
            float(probabilities[predicted_index])
            * 100
        )


        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("Prediction")

        st.success(
            f"### {predicted_class} HAIR"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # ALL CLASS PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                float(probabilities[index])
                * 100
            )

            st.write(
                f"**{class_name}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability / 100,
                        0.0
                    ),
                    1.0
                )
            )


        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        st.divider()

        st.info(
            "This system classifies the visible hair-length "
            "category of the uploaded image. It does not "
            "infer gender or other sensitive personal attributes."
        )


    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)