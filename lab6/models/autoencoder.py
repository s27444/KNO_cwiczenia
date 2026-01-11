from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_shape=(128, 128, 3), latent_dim=2):
    """
    Buduje autoenkoder z bardzo małą przestrzenią latentną (latent_dim=2)

    Args:
        input_shape: Rozmiar obrazków wejściowych (wysokość, szerokość, kanały)
        latent_dim: Wymiar przestrzeni latentnej (bottleneck) - bardzo mały dla lab6
    """
    # Encoder - kompresuje obraz do przestrzeni latentnej
    encoder_input = keras.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 64x64
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 32x32
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 16x16
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 8x8

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = keras.Model(encoder_input, z, name="encoder")

    # Decoder - odtwarza obraz z przestrzeni latentnej
    decoder_input = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(512, activation="relu")(decoder_input)
    x = layers.Dense(8 * 8 * 256, activation="relu")(x)
    x = layers.Reshape((8, 8, 256))(x)

    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 16x16
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 32x32
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 64x64
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 128x128
    decoder_output = layers.Conv2DTranspose(
        input_shape[2],
        (3, 3),
        activation="sigmoid",
        padding="same",
        name="decoder_output",
    )(x)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder - połączenie encoder + decoder
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

    autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return autoencoder, encoder, decoder
