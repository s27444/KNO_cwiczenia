import argparse
import pickle

import numpy as np
from tensorflow.keras.models import load_model


def predict_wine():
    parser = argparse.ArgumentParser(description="Wine Class Predictor")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # Stałe nazwy kolumn
    columns = [
        "class",
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280/OD315_of_diluted_wines",
        "Proline",
    ]

    # Tworzymy argumenty CLI dla 13 cech
    for feature in columns[1:]:
        parser.add_argument(f"--{feature}", type=float, required=True)

    args = parser.parse_args()

    input_data = np.array([[getattr(args, f) for f in columns[1:]]])
    input_scaled = scaler.transform(input_data)

    model = load_model("best_model.h5")
    prediction = model.predict(input_scaled)
    class_idx = np.argmax(prediction)
    class_label = encoder.categories_[0][class_idx]

    print(f"\n>>> Przewidywana klasa wina: {class_label}")
