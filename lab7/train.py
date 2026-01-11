import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.model_fc import build_model as build_fc_model
from models.model_lstm import build_model as build_lstm_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import (create_sequences, enrich_features, load_time_series_csv,
                   prepare_data_for_fc)


def generate_experiment_report(
    csv_file,
    col_name,
    n_samples,
    lookback,
    epochs,
    batch_size,
    test_split,
    use_tuner,
    compare_fc,
    metrics_lstm,
    metrics_fc=None,
    best_hps=None,
    output_file="outputs/experiment_report.md",
):
    """
    Generuje elegancki raport z eksperymentów w formacie Markdown
    """
    from datetime import datetime

    report = f"""# Raport z Eksperymentów - Prognozowanie Szeregów Czasowych

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Dane Wejściowe

- **Plik danych:** `{Path(csv_file).name}`
- **Kolumna użyta:** `{col_name}`
- **Liczba próbek:** {n_samples:,}
- **Podział train/test:** {1-test_split:.0%} / {test_split:.0%}

---

## ⚙️ Parametry Eksperymentu

- **Lookback:** {lookback} kroków
- **Liczba epok:** {epochs}
- **Rozmiar batcha:** {batch_size}
- **Keras Tuner:** {"✅ Tak" if use_tuner else "❌ Nie"}
- **Porównanie z FC:** {"✅ Tak" if compare_fc else "❌ Nie"}

"""

    if use_tuner and best_hps:
        report += f"""### 🎯 Najlepsze Hiperparametry (Keras Tuner)

- **Units LSTM 1:** {best_hps.get('units1')}
- **Units LSTM 2:** {best_hps.get('units2')}
- **Dropout:** {best_hps.get('dropout'):.3f}
- **Learning Rate:** {best_hps.get('learning_rate'):.6f}

"""

    report += f"""---

## 📈 Wyniki Modelu LSTM

- **Test Loss (MSE):** {metrics_lstm['test_loss']:.6f}
- **Test MAE:** {metrics_lstm['test_mae']:.6f}

"""

    if compare_fc and metrics_fc:
        report += f"""---

## 📊 Porównanie Modeli

| Metryka | LSTM | FC | Różnica |
|---------|------|-----|---------|
| **Test Loss (MSE)** | {metrics_lstm['test_loss']:.6f} | {metrics_fc['test_loss']:.6f} | {metrics_lstm['test_loss'] - metrics_fc['test_loss']:+.6f} |
| **Test MAE** | {metrics_lstm['test_mae']:.6f} | {metrics_fc['test_mae']:.6f} | {metrics_lstm['test_mae'] - metrics_fc['test_mae']:+.6f} |

### 📝 Wnioski

"""

        if metrics_lstm["test_loss"] < metrics_fc["test_loss"]:
            report += f"- ✅ **Model LSTM osiągnął lepsze wyniki** (mniejsza strata: {((metrics_fc['test_loss'] - metrics_lstm['test_loss']) / metrics_fc['test_loss'] * 100):.1f}%)\n"
        else:
            report += f"- ✅ **Model FC osiągnął lepsze wyniki** (mniejsza strata: {((metrics_lstm['test_loss'] - metrics_fc['test_loss']) / metrics_lstm['test_loss'] * 100):.1f}%)\n"

        report += f"- LSTM jest lepszy dla szeregów czasowych ze względu na zdolność do zapamiętywania długoterminowych zależności\n"
        report += f"- FC jest szybszy w treningu i inferencji, ale może mieć gorszą zdolność do modelowania zależności czasowych\n"

    report += f"""
---

## 📁 Pliki Wyjściowe

- **Model LSTM:** `models/time_series_lstm.keras`
"""

    if compare_fc:
        report += f"- **Model FC:** `models/time_series_fc.keras`\n"

    report += f"""- **Wykresy treningu:** `outputs/training_results.png`
- **Metryki (JSON):** `outputs/metrics.json`

---

## 🔄 Wzbogacenie Danych

Model wykorzystuje następujące techniki wzbogacenia danych:

1. **Przesunięcia czasowe (lags):** 1, 2, 3, 7 kroków
2. **Średnie kroczące:** okna 3, 7, 14
3. **Różnice:** zmiana względem poprzedniej wartości
4. **Cechy harmoniczne:** sin/cos dla cykliczności (7, 30, 365 dni)

---

*Raport wygenerowany automatycznie po zakończeniu treningu.*
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"   📄 Raport zapisany: {output_file}")


def train_models(
    csv_file,
    value_column=None,
    lookback=30,
    forecast_horizon=1,
    test_split=0.2,
    epochs=50,
    batch_size=32,
    use_tuner=False,
    compare_fc=False,
):
    """
    Główna funkcja treningu modeli szeregów czasowych

    Args:
        csv_file: Ścieżka do pliku CSV z danymi
        value_column: Nazwa kolumny z wartościami (None = pierwsza numeryczna)
        lookback: Liczba kroków wstecz dla modelu
        forecast_horizon: Liczba kroków do przodu (obecnie 1)
        test_split: Proporcja danych testowych
        epochs: Liczba epok
        batch_size: Rozmiar batcha
        use_tuner: Czy użyć Keras Tuner
        compare_fc: Czy porównać z modelem FC (*)
    """
    print("=" * 60)
    print("Trening modelu szeregów czasowych")
    print("=" * 60)

    # 1. Ładowanie danych
    print(f"\n📁 Ładowanie danych z: {csv_file}")
    values, dates, col_name = load_time_series_csv(csv_file, value_column)
    print(f"   Kolumna: {col_name}")
    print(f"   Liczba próbek: {len(values)}")
    print(f"   Zakres wartości: [{values.min():.2f}, {values.max():.2f}]")

    # 2. Normalizacja
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    # 3. Wzbogacenie danych
    print("\n🔄 Wzbogacanie danych...")
    enriched_data = enrich_features(values_scaled, lookback=lookback)
    print(f"   Oryginalne cechy: 1")
    print(f"   Wzbogacone cechy: {enriched_data.shape[1]}")
    print(f"   (dodano: przesunięcia czasowe, średnie kroczące, różnice, harmoniczne)")

    # 4. Tworzenie sekwencji
    print(f"\n📊 Tworzenie sekwencji...")
    print(f"   Lookback: {lookback} kroków")
    print(f"   Forecast horizon: {forecast_horizon} kroków")
    X, y = create_sequences(
        enriched_data, lookback=lookback, forecast_horizon=forecast_horizon
    )
    print(f"   Sekwencje: {X.shape}")
    print(f"   Wartości docelowe: {y.shape}")

    # 5. Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_split,
        shuffle=False,  # shuffle=False dla szeregów czasowych!
    )
    print(f"\n📈 Podział danych:")
    print(f"   Treningowe: {X_train.shape[0]} próbek")
    print(f"   Testowe: {X_test.shape[0]} próbek")

    # 6. Trening modelu LSTM
    print("\n" + "=" * 60)
    print("Trening modelu LSTM...")
    print("=" * 60)

    input_shape = (lookback, enriched_data.shape[1])

    best_hps = None
    if use_tuner:
        print("\n🔧 Używam Keras Tuner do optymalizacji hiperparametrów...")
        from keras_tuner import RandomSearch
        from models.model_tuner import build_model_hp

        def build_tuner_model(hp):
            hp.values["input_shape"] = input_shape
            return build_model_hp(hp, model_type="lstm")

        tuner = RandomSearch(
            build_tuner_model,
            objective="val_loss",
            max_trials=10,
            executions_per_trial=1,
            directory="outputs/tuner",
            project_name="time_series_lstm",
        )

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=batch_size,
            verbose=1,
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"\n✅ Najlepsze hiperparametry:")
        print(f"   units1: {best_hps.get('units1')}")
        print(f"   units2: {best_hps.get('units2')}")
        print(f"   dropout: {best_hps.get('dropout')}")
        print(f"   learning_rate: {best_hps.get('learning_rate')}")

        model_lstm = tuner.get_best_models(num_models=1)[0]
    else:
        model_lstm = build_lstm_model(input_shape)

    history_lstm = model_lstm.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Ewaluacja LSTM
    test_loss_lstm, test_mae_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Wyniki modelu LSTM:")
    print(f"   Test Loss (MSE): {test_loss_lstm:.6f}")
    print(f"   Test MAE: {test_mae_lstm:.6f}")

    # 7. Porównanie z modelem FC (opcjonalnie)
    model_fc = None
    history_fc = None
    test_loss_fc = None
    test_mae_fc = None

    if compare_fc:
        print("\n" + "=" * 60)
        print("Trening modelu Fully Connected (porównanie)...")
        print("=" * 60)

        X_train_fc = prepare_data_for_fc(X_train)
        X_test_fc = prepare_data_for_fc(X_test)

        input_shape_fc = (lookback * enriched_data.shape[1],)
        model_fc = build_fc_model(input_shape_fc)

        history_fc = model_fc.fit(
            X_train_fc,
            y_train,
            validation_data=(X_test_fc, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        test_loss_fc, test_mae_fc = model_fc.evaluate(X_test_fc, y_test, verbose=0)
        print(f"\n📊 Wyniki modelu FC:")
        print(f"   Test Loss (MSE): {test_loss_fc:.6f}")
        print(f"   Test MAE: {test_mae_fc:.6f}")

    # 8. Zapis modeli
    print("\n💾 Zapisuję modele...")
    Path("models").mkdir(parents=True, exist_ok=True)

    model_lstm.save("models/time_series_lstm.keras")
    print("   ✅ Model LSTM zapisany: models/time_series_lstm.keras")

    if model_fc:
        model_fc.save("models/time_series_fc.keras")
        print("   ✅ Model FC zapisany: models/time_series_fc.keras")

    # Zapis skalera i parametrów
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/lookback.pkl", "wb") as f:
        pickle.dump(lookback, f)
    with open("models/enriched_features.pkl", "wb") as f:
        pickle.dump(enriched_data.shape[1], f)

    # 9. Zapis wyników
    print("\n📊 Zapisuję wyniki...")
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Zapis wykresów
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history_lstm.history["loss"], label="Train Loss (LSTM)")
    axes[0, 0].plot(history_lstm.history["val_loss"], label="Val Loss (LSTM)")
    if history_fc:
        axes[0, 0].plot(history_fc.history["loss"], label="Train Loss (FC)")
        axes[0, 0].plot(history_fc.history["val_loss"], label="Val Loss (FC)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (MSE)")
    axes[0, 0].set_title("Training History - Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE
    axes[0, 1].plot(history_lstm.history["mae"], label="Train MAE (LSTM)")
    axes[0, 1].plot(history_lstm.history["val_mae"], label="Val MAE (LSTM)")
    if history_fc:
        axes[0, 1].plot(history_fc.history["mae"], label="Train MAE (FC)")
        axes[0, 1].plot(history_fc.history["val_mae"], label="Val MAE (FC)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].set_title("Training History - MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Przykładowa predykcja na danych testowych
    y_pred_lstm = model_lstm.predict(X_test, verbose=0)

    # Denormalizacja
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_lstm_denorm = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

    # Wykres predykcji (pierwsze 100 punktów)
    n_plot = min(100, len(y_test))
    axes[1, 0].plot(y_test_denorm[:n_plot], label="Rzeczywiste", alpha=0.7)
    axes[1, 0].plot(y_pred_lstm_denorm[:n_plot], label="Predykcja LSTM", alpha=0.7)
    if model_fc:
        y_pred_fc = model_fc.predict(X_test_fc, verbose=0)
        y_pred_fc_denorm = scaler.inverse_transform(y_pred_fc.reshape(-1, 1)).flatten()
        axes[1, 0].plot(y_pred_fc_denorm[:n_plot], label="Predykcja FC", alpha=0.7)
    axes[1, 0].set_xlabel("Czas")
    axes[1, 0].set_ylabel("Wartość")
    axes[1, 0].set_title(f"Przykładowa predykcja (pierwsze {n_plot} punktów)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Scatter plot: rzeczywiste vs predykcja
    axes[1, 1].scatter(y_test_denorm, y_pred_lstm_denorm, alpha=0.5, label="LSTM", s=10)
    if model_fc:
        axes[1, 1].scatter(y_test_denorm, y_pred_fc_denorm, alpha=0.5, label="FC", s=10)
    axes[1, 1].plot(
        [y_test_denorm.min(), y_test_denorm.max()],
        [y_test_denorm.min(), y_test_denorm.max()],
        "r--",
        label="Idealna predykcja",
    )
    axes[1, 1].set_xlabel("Rzeczywiste wartości")
    axes[1, 1].set_ylabel("Predykcje")
    axes[1, 1].set_title("Rzeczywiste vs Predykcja")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("outputs/training_results.png", dpi=150, bbox_inches="tight")
    print("   ✅ Wykres zapisany: outputs/training_results.png")

    # Zapis metryk
    metrics = {
        "lstm": {
            "test_loss": float(test_loss_lstm),
            "test_mae": float(test_mae_lstm),
        }
    }

    if model_fc:
        metrics["fc"] = {
            "test_loss": float(test_loss_fc),
            "test_mae": float(test_mae_fc),
        }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📈 Metryki zapisane: outputs/metrics.json")

    # Generuj elegancki raport z eksperymentów
    generate_experiment_report(
        csv_file=csv_file,
        col_name=col_name,
        n_samples=len(values),
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        test_split=test_split,
        use_tuner=use_tuner,
        compare_fc=compare_fc,
        metrics_lstm={"test_loss": test_loss_lstm, "test_mae": test_mae_lstm},
        metrics_fc=(
            {"test_loss": test_loss_fc, "test_mae": test_mae_fc} if model_fc else None
        ),
        best_hps=best_hps,
    )

    print(f"\n{'='*60}")
    print(f"✅ Trening zakończony!")
    print(f"{'='*60}")

    return model_lstm, model_fc, scaler, lookback, enriched_data.shape[1]
