import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import enrich_features, load_time_series_csv, prepare_data_for_fc


def predict_future(
    csv_file,
    model_path="models/time_series_lstm.keras",
    n_predictions=10,
    output_file="outputs/predictions.csv",
    value_column=None,
    use_fc=False,
):
    """
    Generuje N kolejnych predykcji na podstawie danych historycznych

    Args:
        csv_file: Plik CSV z danymi historycznymi
        model_path: Ścieżka do modelu
        n_predictions: Liczba kolejnych predykcji do wygenerowania
        output_file: Plik wyjściowy CSV z predykcjami
        value_column: Nazwa kolumny z wartościami
        use_fc: Czy użyć modelu FC zamiast LSTM
    """
    print("=" * 60)
    print("Prognozowanie szeregów czasowych")
    print("=" * 60)

    # Sprawdź czy pliki istnieją
    if not Path(csv_file).exists():
        print(f"❌ Błąd: Plik CSV nie istnieje: {csv_file}", file=sys.stderr)
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"❌ Błąd: Model nie istnieje: {model_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Ładowanie danych historycznych
    print(f"\n📁 Ładowanie danych historycznych: {csv_file}")
    values, dates, col_name = load_time_series_csv(csv_file, value_column)
    print(f"   Kolumna: {col_name}")
    print(f"   Liczba punktów: {len(values)}")

    # 2. Ładowanie modelu i parametrów
    print(f"\n📥 Ładowanie modelu: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"📥 Ładowanie parametrów...")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/lookback.pkl", "rb") as f:
        lookback = pickle.load(f)
    with open("models/enriched_features.pkl", "rb") as f:
        n_features = pickle.load(f)

    print(f"   Lookback: {lookback}")
    print(f"   Liczba cech: {n_features}")

    # 3. Normalizacja danych
    values_scaled = scaler.transform(values.reshape(-1, 1)).flatten()

    # 4. Wzbogacenie danych
    print(f"\n🔄 Wzbogacanie danych...")
    enriched_data = enrich_features(values_scaled, lookback=lookback)

    # 5. Generowanie predykcji
    print(f"\n🔮 Generowanie {n_predictions} kolejnych predykcji...")

    predictions = []
    current_data = enriched_data.copy()

    # Używamy ostatnich 'lookback' punktów jako wejścia
    for i in range(n_predictions):
        # Weź ostatnie 'lookback' kroków
        sequence = current_data[-lookback:].reshape(1, lookback, n_features)

        # Dla modelu FC - spłaszcz sekwencję
        if use_fc or "fc" in str(model_path).lower():
            sequence_fc = prepare_data_for_fc(sequence)
            pred_scaled = model.predict(sequence_fc, verbose=0)[0, 0]
        else:
            pred_scaled = model.predict(sequence, verbose=0)[0, 0]

        # Denormalizacja
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(pred)

        # Dodaj predykcję do danych (dla kolejnej predykcji)
        # Musimy zaktualizować wszystkie cechy wzbogacone
        new_value_scaled = pred_scaled

        # Aktualizacja danych wzbogaconych
        # 1. Oryginalna wartość
        new_enriched = [new_value_scaled]

        # 2. Przesunięcia czasowe (aktualizujemy ostatnie wartości)
        enriched_values = current_data[:, 0]  # Oryginalne wartości
        enriched_values = np.append(enriched_values, new_value_scaled)

        # 3. Przesunięcia (lag features)
        for lag in [1, 2, 3, 7]:
            lagged = (
                enriched_values[-lag - 1]
                if len(enriched_values) > lag
                else enriched_values[0]
            )
            new_enriched.append(lagged)

        # 4. Średnie kroczące (uproszczone - używamy ostatnich wartości)
        window_sizes = [3, 7, 14]
        for window in window_sizes:
            if len(enriched_values) >= window:
                rolling_mean = np.mean(enriched_values[-window:])
            else:
                rolling_mean = np.mean(enriched_values)
            new_enriched.append(rolling_mean)

        # 5. Różnice
        if len(enriched_values) > 1:
            diff = enriched_values[-1] - enriched_values[-2]
        else:
            diff = 0.0
        new_enriched.append(diff)

        # 6. Cechy harmoniczne (dla aktualnego indeksu)
        n = len(enriched_values)
        for period in [7, 30, 365]:
            if n > period:
                t = n - 1
                sin_feature = np.sin(2 * np.pi * t / period)
                cos_feature = np.cos(2 * np.pi * t / period)
                new_enriched.append(sin_feature)
                new_enriched.append(cos_feature)
            else:
                # Jeśli za mało danych, użyj prostych wartości
                new_enriched.extend([0.0, 1.0])

        # Dodaj nowy wiersz do danych
        new_row = np.array(
            new_enriched[:n_features]
        )  # Upewnij się że mamy tyle cech co potrzeba
        current_data = np.vstack([current_data, new_row])

        print(f"   Predykcja {i+1}/{n_predictions}: {pred:.2f}")

    # 6. Zapis wyników
    print(f"\n💾 Zapisuję predykcje do: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Utwórz DataFrame z predykcjami
    predictions_df = pd.DataFrame(
        {
            "prediction_number": range(1, n_predictions + 1),
            "predicted_value": predictions,
        }
    )

    predictions_df.to_csv(output_file, index=False)
    print(f"   ✅ Zapisano {n_predictions} predykcji")

    # 7. Wizualizacja (opcjonalnie)
    output_file_str = str(output_file)
    if output_file_str.endswith(".csv"):
        plot_file = output_file_str.replace(".csv", ".png")
    else:
        plot_file = output_file_str + ".png"

    print(f"\n📊 Tworzenie wizualizacji...")
    plt.figure(figsize=(12, 6))

    # Ostatnie punkty danych historycznych (dla kontekstu)
    n_show = min(100, len(values))
    history_to_show = values[-n_show:]
    history_indices = np.arange(len(history_to_show))
    prediction_indices = np.arange(
        len(history_to_show), len(history_to_show) + n_predictions
    )

    plt.plot(history_indices, history_to_show, label="Dane historyczne", alpha=0.7)
    plt.plot(
        prediction_indices,
        predictions,
        "ro-",
        label=f"Predykcje ({n_predictions} kroków)",
        markersize=6,
    )
    plt.axvline(
        x=len(history_to_show) - 1,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Koniec danych",
    )
    plt.xlabel("Czas")
    plt.ylabel("Wartość")
    plt.title(f"Prognoza - {n_predictions} kolejnych kroków")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"   ✅ Wykres zapisany: {plot_file}")

    # 8. Wyświetl predykcje na standardowe wyjście
    print(f"\n📈 Predykcje:")
    for i, pred in enumerate(predictions, 1):
        print(f"   Krok {i}: {pred:.2f}")

    print(f"\n{'='*60}")
    print(f"✅ Prognozowanie zakończone!")
    print(f"{'='*60}")

    return predictions


def predict_time_series():
    """Funkcja do wywołania z main.py - parsuje argumenty"""
    import argparse

    parser = argparse.ArgumentParser(description="Prognozowanie szeregów czasowych")
    parser.add_argument(
        "--history", type=Path, required=True, help="Plik CSV z danymi historycznymi"
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Liczba kolejnych predykcji do wygenerowania",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("outputs/predictions.csv"),
        help="Plik wyjściowy CSV",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/time_series_lstm.keras"),
        help="Ścieżka do modelu",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Nazwa kolumny z wartościami (None = pierwsza numeryczna)",
    )
    parser.add_argument(
        "--fc", action="store_true", help="Użyj modelu Fully Connected zamiast LSTM"
    )

    args = parser.parse_args()

    predict_future(
        csv_file=args.history,
        model_path=args.model,
        n_predictions=args.n,
        output_file=args.result,
        value_column=args.column,
        use_fc=args.fc,
    )
