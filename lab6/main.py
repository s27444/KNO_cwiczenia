import argparse
import sys

from predict import predict_autoencoder
from train import train_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Autoenkoder - Trening i generowanie")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Tryb działania: 'train' lub 'predict'",
    )

    args, unknown = parser.parse_known_args()

    if args.mode == "train":
        # Proste parsowanie argumentów dla treningu
        data_dir = "data/images"
        epochs = 50
        batch_size = 32
        latent_dim = 2

        i = 0
        while i < len(unknown):
            if unknown[i] == "--data-dir" and i + 1 < len(unknown):
                data_dir = unknown[i + 1]
                i += 2
            elif unknown[i] == "--epochs" and i + 1 < len(unknown):
                epochs = int(unknown[i + 1])
                i += 2
            elif unknown[i] == "--batch-size" and i + 1 < len(unknown):
                batch_size = int(unknown[i + 1])
                i += 2
            elif unknown[i] == "--latent-dim" and i + 1 < len(unknown):
                latent_dim = int(unknown[i + 1])
                i += 2
            else:
                i += 1

        print("🚀 Rozpoczynam trening autoenkodera...")
        train_autoencoder(
            data_dir=data_dir,
            image_size=(128, 128),
            latent_dim=latent_dim,
            batch_size=batch_size,
            epochs=epochs,
        )

    elif args.mode == "predict":
        # Przekazujemy argumenty do predict.py
        sys.argv = [sys.argv[0]] + unknown
        print("🔮 Tryb predykcji/generowania...")
        predict_autoencoder()


if __name__ == "__main__":
    main()
