import argparse
from train import train_models
from predict import predict_wine

def main():
    parser = argparse.ArgumentParser(description="Wine Classification - Training or Prediction")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Wybierz tryb działania programu: 'train' lub 'predict'."
    )
    args, unknown = parser.parse_known_args()

    if args.mode == "train":
        print("🚀 Rozpoczynam trenowanie modeli...")
        train_models()
    elif args.mode == "predict":
        print("🔮 Tryb predykcji uruchomiony...")
        # Przekazujemy pozostałe argumenty do funkcji predict_wine()
        import sys
        sys.argv = [sys.argv[0]] + unknown
        predict_wine()

if __name__ == "__main__":
    main()
