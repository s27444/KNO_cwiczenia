from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_dim, normalizer):
    """Model z wbudowaną warstwą normalizacji"""
    model = Sequential([
        Input(shape=(input_dim,)),
        normalizer,
        Dense(32, activation="relu", name="Hidden_1"),
        Dense(16, activation="relu", name="Hidden_2"),
        Dense(3, activation="softmax", name="Output")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_model_hp(hp, normalizer):
    """Model tunera z warstwą normalizacji"""
    model = Sequential()
    model.add(Input(shape=(13,)))
    model.add(normalizer)

    model.add(Dense(
        units=hp.Int('units1', min_value=16, max_value=128, step=16),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        name="Hidden_1"
    ))
    model.add(Dense(
        units=hp.Int('units2', min_value=8, max_value=64, step=8),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        name="Hidden_2"
    ))
    model.add(Dense(3, activation="softmax", name="Output"))

    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

