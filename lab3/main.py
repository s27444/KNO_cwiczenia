import argparse
import sys

from predict import predict_wine
from train import train_models


def main():
    parser = argparse.ArgumentParser(
        description="Wine Classification - Training or Prediction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Wybierz tryb działania programu: 'train' lub 'predict'.",
    )
    args, unknown = parser.parse_known_args()

    match args.mode:
        case "train":
            print("🚀 Rozpoczynam trenowanie modeli...")
            train_models()
        case "predict":
            print("🔮 Tryb predykcji uruchomiony...")
            sys.argv = [sys.argv[0]] + unknown
            predict_wine()
        case _:
            print(f"❌ Nieobsługiwany tryb: {args.mode}")


if __name__ == "__main__":
    main()
