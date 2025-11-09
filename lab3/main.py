import argparse
from train import train_models
from predict import predict_wine

def main():
    parser = argparse.ArgumentParser(description="Wine Classification Project")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help="Wybierz tryb działania: train / predict")
    args, unknown = parser.parse_known_args()

    if args.mode == 'train':
        train_models()
    elif args.mode == 'predict':
        predict_wine()

if __name__ == "__main__":
    main()
