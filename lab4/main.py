import argparse
import sys
from train import train_models
from train_tuner import train_tuner  # funkcja w tym pliku do uruchomienia tunera
from predict import predict_wine

def main():
    parser = argparse.ArgumentParser(description="Wine Classification - Train/Predict/Tuner")
    parser.add_argument("--mode", type=str, required=True, choices=["train","predict","tuner"])
    args, unknown = parser.parse_known_args()

    if args.mode == "train":
        train_models()
    elif args.mode == "tuner":
        train_tuner()
    elif args.mode == "predict":
        sys.argv = [sys.argv[0]] + unknown
        predict_wine()

if __name__ == "__main__":
    main()
