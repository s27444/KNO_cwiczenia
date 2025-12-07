import pickle
import numpy as np
from tensorflow.keras.models import load_model
import argparse


def predict_wine():
    parser = argparse.ArgumentParser(description="Wine Class Predictor")

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    columns = [
        "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
        "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
        "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"
    ]

    for feature in columns:
        parser.add_argument(f"--{feature}", type=float, required=True)

    args = parser.parse_args()
    input_data = np.array([[getattr(args, f) for f in columns]])

    # NIE SKALUJEMY — model ma Normalization w środku
    model = load_model("best_model_tuned.keras")

    prediction = model.predict(input_data)
    class_idx = np.argmax(prediction)
    class_label = encoder.categories_[0][class_idx]

    print(f"\n>>> Przewidywana klasa wina: {class_label}")
