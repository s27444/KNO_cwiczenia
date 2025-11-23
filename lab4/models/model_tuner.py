from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_dim, learning_rate=0.01, hidden_units1=32, hidden_units2=16, activation="relu"):
    """Tworzenie modelu dla podanych parametrów"""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_units1, activation=activation, name="Hidden_1"),
        Dense(hidden_units2, activation=activation, name="Hidden_2"),
        Dense(3, activation="softmax", name="Output")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_model_hp(hp):
    """Tworzenie modelu dla Keras Tuner"""
    model = Sequential()
    model.add(Input(shape=(13,)))
    model.add(Dense(
        units=hp.Int('units1', 16, 128, step=16),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        name="Hidden_1"
    ))
    model.add(Dense(
        units=hp.Int('units2', 8, 64, step=8),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        name="Hidden_2"
    ))
    model.add(Dense(3, activation='softmax', name="Output"))
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
