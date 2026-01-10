import argparse
import sys

from predict import predict_wine
from train import train_models


def main():
    parser = argparse.ArgumentParser(description="Fashion MNIST - Trening i predykcja")
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
        epochs = 10
        use_augmentation = False

        i = 0
        while i < len(unknown):
            if unknown[i] == "--epochs" and i + 1 < len(unknown):
                epochs = int(unknown[i + 1])
                i += 2
            elif unknown[i] == "--augmentation":
                use_augmentation = True
                i += 1
            else:
                i += 1

        print("🚀 Rozpoczynam trenowanie modeli Fashion MNIST...")
        train_models(epochs=epochs, use_augmentation=use_augmentation)

    elif args.mode == "predict":
        # Przekazujemy argumenty do predict.py (podobnie jak w lab3/lab4)
        sys.argv = [sys.argv[0]] + unknown
        print("🔮 Tryb predykcji...")
        predict_wine()


if __name__ == "__main__":
    main()
