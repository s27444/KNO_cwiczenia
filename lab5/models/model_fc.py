from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(28, 28, 1)):
    """Model z warstwami w pełni połączonymi (Fully Connected)"""
    model = keras.Sequential(
        [
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def build_model_hp(hp):
    """Model dla Keras Tuner - wersja Fully Connected"""
    model = keras.Sequential(
        [
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(
                units=hp.Int("dense_1", min_value=64, max_value=256, step=32),
                activation=hp.Choice("activation_1", ["relu", "tanh"]),
            ),
            layers.Dropout(rate=hp.Float("dropout_1", 0.1, 0.5, step=0.1)),
            layers.Dense(
                units=hp.Int("dense_2", min_value=32, max_value=128, step=16),
                activation=hp.Choice("activation_2", ["relu", "tanh"]),
            ),
            layers.Dropout(rate=hp.Float("dropout_2", 0.1, 0.5, step=0.1)),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
