import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Nazwy klas Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess_image(image_path: Path):
    """
    Przetwarza obraz do formatu Fashion MNIST:
    - Skalowanie do 28x28
    - Konwersja na odcienie szarości
    - Negatyw (Fashion MNIST używa negatywów)
    - Normalizacja do [0, 1]
    """
    # Wczytaj obraz
    img = Image.open(image_path)

    # Konwersja na grayscale jeśli nie jest
    if img.mode != "L":
        img = img.convert("L")

    # Skalowanie do 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Konwersja do numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalizacja do [0, 1]
    img_array = img_array / 255.0

    # Negatyw (Fashion MNIST używa negatywów)
    img_array = 1.0 - img_array

    # Reshape dla modelu (28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


def predict_image(model_path: Path, image_path: Path):
    """Wykonuje predykcję na obrazie"""
    # Sprawdź czy pliki istnieją
    if not model_path.exists():
        print(f"Błąd: Model nie istnieje: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not image_path.exists():
        print(f"Błąd: Obraz nie istnieje: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Wczytaj model
    model = tf.keras.models.load_model(model_path)

    # Przetwórz obraz
    processed_image = preprocess_image(image_path)

    # Wykonaj predykcję
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])

    # Wyświetl wynik
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    print(f"Klasa: {predicted_class_name}")
    print(f"Pewność: {confidence:.4f}")


def predict_fashion():
    """Funkcja do wywołania z main.py - parsuje argumenty"""
    import argparse

    parser = argparse.ArgumentParser(description="Fashion MNIST Predictor")
    parser.add_argument("image", type=Path, help="Ścieżka do obrazu")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fashion_mnist_best.keras"),
        help="Ścieżka do modelu",
    )
    args = parser.parse_args()
    predict_image(args.model, args.image)


def main():
    """Główna funkcja - pobiera argumenty z linii komend"""
    if len(sys.argv) < 2:
        print(
            "Użycie: python predict.py <ścieżka_do_obrazu> [--model ścieżka_do_modelu]",
            file=sys.stderr,
        )
        print("Domyślny model: models/fashion_mnist_best.keras", file=sys.stderr)
        sys.exit(1)

    predict_fashion()


if __name__ == "__main__":
    main()
