import argparse
import sys

from predict import predict_time_series
from train import train_models


def main():
    parser = argparse.ArgumentParser(
        description="Prognozowanie szeregów czasowych - Trening i predykcja"
    )
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
        csv_file = None
        value_column = None
        lookback = 30
        epochs = 50
        batch_size = 32
        test_split = 0.2
        use_tuner = False
        compare_fc = False

        i = 0
        while i < len(unknown):
            if unknown[i] == "--csv" and i + 1 < len(unknown):
                csv_file = unknown[i + 1]
                i += 2
            elif unknown[i] == "--column" and i + 1 < len(unknown):
                value_column = unknown[i + 1]
                i += 2
            elif unknown[i] == "--lookback" and i + 1 < len(unknown):
                lookback = int(unknown[i + 1])
                i += 2
            elif unknown[i] == "--epochs" and i + 1 < len(unknown):
                epochs = int(unknown[i + 1])
                i += 2
            elif unknown[i] == "--batch-size" and i + 1 < len(unknown):
                batch_size = int(unknown[i + 1])
                i += 2
            elif unknown[i] == "--tuner":
                use_tuner = True
                i += 1
            elif unknown[i] == "--compare-fc":
                compare_fc = True
                i += 1
            else:
                i += 1

        if csv_file is None:
            print("❌ Błąd: Musisz podać --csv <plik.csv>", file=sys.stderr)
            sys.exit(1)

        print("🚀 Rozpoczynam trening modelu szeregów czasowych...")
        train_models(
            csv_file=csv_file,
            value_column=value_column,
            lookback=lookback,
            epochs=epochs,
            batch_size=batch_size,
            test_split=test_split,
            use_tuner=use_tuner,
            compare_fc=compare_fc,
        )

    elif args.mode == "predict":
        # Przekazujemy argumenty do predict.py
        sys.argv = [sys.argv[0]] + unknown
        print("🔮 Tryb prognozowania...")
        predict_time_series()


if __name__ == "__main__":
    main()
