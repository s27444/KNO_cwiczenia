from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, units=128, dropout=0.2):
    """
    Buduje model Fully Connected do prognozowania szeregów czasowych
    (dla porównania z modelem rekurencyjnym)

    Args:
        input_shape: Kształt danych wejściowych (lookback * features)
        units: Liczba jednostek w warstwie Dense
        dropout: Wartość dropout
    """
    model = keras.Sequential(
        [
            layers.Dense(units, activation="relu", input_shape=input_shape),
            layers.Dropout(dropout),
            layers.Dense(units // 2, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),  # Output: 1 wartość
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model
