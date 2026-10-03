# Long Hair Identification AI

A deep learning project that classifies visible hair length from an uploaded image using a convolutional neural network (CNN) and a Streamlit web interface.

## Project Overview

The application analyzes an uploaded image and predicts one of four hair-length categories:

- **BALD**
- **SHORT**
- **MEDIUM**
- **LONG**

It displays the predicted category, confidence score, and probability distribution across all four classes.

## Features

- Image upload through a Streamlit interface
- Four-class hair-length classification
- Deep learning model built with TensorFlow and Keras
- MobileNetV2 preprocessing
- Prediction confidence and class probabilities
- Simple, interactive web interface

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit

## Project Structure

```text
Long_Hair_Identification_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── hair_classifier_4class_best.keras
│   ├── hair_4class_names.json
│   └── hair_4class_test_results.json
│
└── src/
    ├── Dataset preparation scripts
    ├── Model training scripts
    ├── Evaluation scripts
    └── Prediction scripts
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Long_Hair_Identification_AI.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Long_Hair_Identification_AI
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Model

The application uses a trained Keras model saved at:

`models/hair_classifier_4class_best.keras`

The model predicts visible hair-length categories. It does not identify a person's gender, identity, or other personal attributes.

## Limitations

- Predictions depend on image quality, lighting, pose, and visibility of hair.
- Confidence scores are model outputs and should not be interpreted as guarantees of correctness.
- The application is intended for educational and demonstration purposes.

## Responsible Use

This project is designed for image-based hair-length classification only. It should not be used to infer sensitive personal characteristics or make consequential decisions about individuals.

## Author

**Piya Shaikh**

MCA Graduate | AI & Machine Learning

