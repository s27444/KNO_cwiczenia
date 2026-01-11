import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def load_and_preprocess_image(image_path, image_size=(128, 128)):
    """Ładuje i przetwarza obraz do formatu autoenkodera"""
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalizacja do [0, 1]
    img_array = tf.expand_dims(img_array, 0)  # Dodaj wymiar batch
    return img_array


def visualize_reconstruction(
    model_path, image_path, output_path="outputs/reconstruction.png"
):
    """
    Wizualizuje rekonstrukcję obrazu przez autoenkoder

    Args:
        model_path: Ścieżka do modelu autoenkodera
        image_path: Ścieżka do obrazu do rekonstrukcji
        output_path: Ścieżka do zapisu wyniku wizualizacji
    """
    if not Path(model_path).exists():
        print(f"❌ Błąd: Model nie istnieje: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"❌ Błąd: Obraz nie istnieje: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📥 Ładowanie modelu: {model_path}")
    autoencoder = tf.keras.models.load_model(model_path)

    print(f"📷 Ładowanie obrazu: {image_path}")
    original = load_and_preprocess_image(image_path)

    print("🔮 Generowanie rekonstrukcji...")
    reconstructed = autoencoder.predict(original, verbose=0)

    # Konwersja do formatu dla wyświetlenia
    original_img = original[0].numpy()
    reconstructed_img = reconstructed[0]

    # Wizualizacja
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_img)
    axes[0].set_title("Oryginalny obraz")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_img)
    axes[1].set_title("Odtworzony obraz")
    axes[1].axis("off")

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Wynik zapisany: {output_path}")

    # Oblicz metryki
    mse = np.mean((original_img - reconstructed_img) ** 2)
    mae = np.mean(np.abs(original_img - reconstructed_img))

    print(f"\n📊 Metryki rekonstrukcji:")
    print(f"   MSE (Mean Squared Error): {mse:.6f}")
    print(f"   MAE (Mean Absolute Error): {mae:.6f}")


def generate_from_latent(
    decoder_path, latent_vector, output_path="outputs/generated.png"
):
    """
    Generuje obraz z przestrzeni latentnej używając dekodera

    Args:
        decoder_path: Ścieżka do modelu dekodera
        latent_vector: Wektor z przestrzeni latentnej (numpy array)
        output_path: Ścieżka do zapisu wygenerowanego obrazu
    """
    if not Path(decoder_path).exists():
        print(f"❌ Błąd: Decoder nie istnieje: {decoder_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📥 Ładowanie dekodera: {decoder_path}")
    decoder = tf.keras.models.load_model(decoder_path)

    # Upewnij się że latent_vector ma odpowiedni kształt
    if isinstance(latent_vector, list):
        latent_vector = np.array(latent_vector)

    if latent_vector.ndim == 1:
        latent_vector = np.expand_dims(latent_vector, 0)

    print(f"🔮 Generowanie obrazu z przestrzeni latentnej...")
    print(f"   Wymiar latentny: {latent_vector.shape}")
    generated = decoder.predict(latent_vector, verbose=0)

    # Wizualizacja
    generated_img = generated[0]

    plt.figure(figsize=(6, 6))
    plt.imshow(generated_img)
    plt.title(f"Wygenerowany obraz (latent: {latent_vector[0]})")
    plt.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Obraz zapisany: {output_path}")


def encode_image(encoder_path, image_path):
    """
    Koduje obraz do przestrzeni latentnej używając enkodera

    Args:
        encoder_path: Ścieżka do modelu enkodera
        image_path: Ścieżka do obrazu do zakodowania

    Returns:
        latent_vector: Wektor z przestrzeni latentnej
    """
    if not Path(encoder_path).exists():
        print(f"❌ Błąd: Encoder nie istnieje: {encoder_path}", file=sys.stderr)
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"❌ Błąd: Obraz nie istnieje: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📥 Ładowanie enkodera: {encoder_path}")
    encoder = tf.keras.models.load_model(encoder_path)

    print(f"📷 Ładowanie obrazu: {image_path}")
    image = load_and_preprocess_image(image_path)

    print("🔮 Kodowanie do przestrzeni latentnej...")
    latent_vector = encoder.predict(image, verbose=0)

    print(f"\n📊 Wektor latentny: {latent_vector[0]}")
    return latent_vector[0]


def predict_autoencoder():
    """Funkcja do wywołania z main.py - parsuje argumenty"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoenkoder - Rekonstrukcja i generowanie"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Komenda")

    # Komenda: reconstruct - rekonstrukcja obrazu
    reconstruct_parser = subparsers.add_parser(
        "reconstruct", help="Rekonstrukcja obrazu przez autoenkoder"
    )
    reconstruct_parser.add_argument("image", type=Path, help="Ścieżka do obrazu")
    reconstruct_parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/autoencoder.keras"),
        help="Ścieżka do modelu autoenkodera",
    )
    reconstruct_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction.png"),
        help="Ścieżka do zapisu wyniku",
    )

    # Komenda: encode - kodowanie obrazu
    encode_parser = subparsers.add_parser(
        "encode", help="Kodowanie obrazu do przestrzeni latentnej"
    )
    encode_parser.add_argument("image", type=Path, help="Ścieżka do obrazu")
    encode_parser.add_argument(
        "--encoder",
        type=Path,
        default=Path("models/encoder.keras"),
        help="Ścieżka do modelu enkodera",
    )

    # Komenda: generate - generowanie z przestrzeni latentnej
    generate_parser = subparsers.add_parser(
        "generate", help="Generowanie obrazu z przestrzeni latentnej"
    )
    generate_parser.add_argument(
        "latent",
        type=float,
        nargs="+",
        help="Wektor z przestrzeni latentnej (2 wartości dla latent_dim=2)",
    )
    generate_parser.add_argument(
        "--decoder",
        type=Path,
        default=Path("models/decoder.keras"),
        help="Ścieżka do modelu dekodera",
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/generated.png"),
        help="Ścieżka do zapisu wygenerowanego obrazu",
    )

    args = parser.parse_args()

    if args.command == "reconstruct":
        visualize_reconstruction(args.model, args.image, args.output)
    elif args.command == "encode":
        encode_image(args.encoder, args.image)
    elif args.command == "generate":
        if len(args.latent) != 2:
            print(
                "❌ Błąd: Dla latent_dim=2 potrzebujesz dokładnie 2 wartości",
                file=sys.stderr,
            )
            sys.exit(1)
        generate_from_latent(args.decoder, args.latent, args.output)


def main():
    """Główna funkcja - pobiera argumenty z linii komend"""
    if len(sys.argv) < 2:
        print("Użycie: python predict.py <komenda> [opcje]", file=sys.stderr)
        print("Komendy: reconstruct, encode, generate", file=sys.stderr)
        sys.exit(1)

    predict_autoencoder()


if __name__ == "__main__":
    main()
