from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, units=64, dropout=0.2):
    """
    Buduje model rekurencyjny LSTM do prognozowania szeregów czasowych

    Args:
        input_shape: Kształt danych wejściowych (lookback, features)
        units: Liczba jednostek LSTM
        dropout: Wartość dropout dla regularyzacji
    """
    model = keras.Sequential(
        [
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout),
            layers.LSTM(units // 2, return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(1),  # Output: 1 wartość (kolejna predykcja)
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def build_model_hp(hp):
    """
    Model dla Keras Tuner - wersja LSTM z hiperparametrami
    input_shape musi być określone podczas tworzenia
    """
    units1 = hp.Int("units1", min_value=32, max_value=128, step=32)
    units2 = hp.Int("units2", min_value=16, max_value=64, step=16)
    dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)

    model = keras.Sequential(
        [
            layers.LSTM(
                units1, return_sequences=True, input_shape=hp.values["input_shape"]
            ),
            layers.Dropout(dropout),
            layers.LSTM(units2, return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )

    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
