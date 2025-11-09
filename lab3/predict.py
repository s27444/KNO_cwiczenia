import argparse
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_loader import load_and_prepare_data

def predict_wine():
    parser = argparse.ArgumentParser(description="Wine Class Predictor")

    local_data_path = "data/wine.data"
    _, _, _, _, scaler, encoder, columns = load_and_prepare_data(local_data_path)

    # 🔹 Tworzymy argumenty CLI dla wszystkich cech wina
    for feature in columns[1:]:
        parser.add_argument(f'--{feature}', type=float, required=True)

    args = parser.parse_args()

    # 🔹 Przygotowanie wejścia użytkownika
    input_data = np.array([[getattr(args, f) for f in columns[1:]]])
    input_scaled = scaler.transform(input_data)

    # 🔹 Wczytanie wytrenowanego modelu
    model = load_model("best_model.h5")
    prediction = model.predict(input_scaled)
    class_idx = np.argmax(prediction)
    class_label = encoder.categories_[0][class_idx]

    print(f"\n>>> Przewidywana klasa wina: {class_label}")
