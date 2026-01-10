from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(28, 28, 1)):
    """Model z warstwami splotowymi (Convolutional Neural Network)"""
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def build_model_hp(hp):
    """Model dla Keras Tuner - wersja Convolutional"""
    model = keras.Sequential(
        [
            layers.Conv2D(
                filters=hp.Int("conv1_filters", 16, 64, step=16),
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                filters=hp.Int("conv2_filters", 32, 128, step=32),
                kernel_size=(3, 3),
                activation="relu",
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(
                units=hp.Int("dense_units", 64, 256, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            ),
            layers.Dropout(rate=hp.Float("dropout", 0.1, 0.5, step=0.1)),
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
