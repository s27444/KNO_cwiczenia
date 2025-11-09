from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(input_dim: int):
    model = Sequential(name="Model_Simple")
    model.add(Dense(16, input_shape=(input_dim,), activation="relu", name="Hidden_1"))
    model.add(Dense(3, activation="softmax", name="Output"))
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
