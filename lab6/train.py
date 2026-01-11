from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.autoencoder import build_autoencoder


def load_data_from_directory(data_dir, image_size=(128, 128), batch_size=32):
    """
    Ładuje obrazy z katalogu używając image_dataset_from_directory
    Pomija etykiety (label_mode=None) ponieważ autoenkoder nie potrzebuje klas
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode=None,  # Brak etykiet - autoenkoder uczy się tylko rekonstrukcji
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    # Normalizacja obrazków do zakresu [0, 1]
    dataset = dataset.map(lambda x: x / 255.0)

    return dataset


def create_augmentation_model():
    """
    Tworzy model augmentacji danych dla autoenkodera
    Augmentacja jest stosowana podczas treningu
    """
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )
    return augmentation


def train_autoencoder(
    data_dir="data/images",
    image_size=(128, 128),
    latent_dim=2,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
):
    """
    Główna funkcja treningu autoenkodera

    Args:
        data_dir: Katalog ze zdjęciami
        image_size: Rozmiar obrazków (wysokość, szerokość)
        latent_dim: Wymiar przestrzeni latentnej (bottleneck) - bardzo mały
        batch_size: Rozmiar batcha
        epochs: Liczba epok
        validation_split: Proporcja danych walidacyjnych
    """
    print("=" * 60)
    print("Autoenkoder - Trening")
    print("=" * 60)

    # Sprawdź czy katalog istnieje
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Błąd: Katalog {data_dir} nie istnieje!")
        print(f"   Utwórz katalog {data_dir} i umieść w nim minimum 20 zdjęć")
        return None

    # Ładowanie danych
    print(f"\n📁 Ładowanie obrazów z katalogu: {data_dir}")
    dataset = load_data_from_directory(
        data_dir, image_size=image_size, batch_size=batch_size
    )

    # Podział na treningowy i walidacyjny
    dataset_size = sum(1 for _ in dataset)
    print(f"   Znaleziono {dataset_size} batchy danych")

    # Tworzymy dataset walidacyjny (ostatnie 20% batchy)
    val_size = int(dataset_size * validation_split)
    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    print(f"   Zbiór treningowy: {dataset_size - val_size} batchy")
    print(f"   Zbiór walidacyjny: {val_size} batchy")

    # Augmentacja danych - dla autoenkodera używamy augmentacji jako input, output pozostaje oryginalny
    print("\n🔄 Przygotowanie augmentacji danych...")

    # Funkcja do augmentacji dla zbioru treningowego
    # Dla autoenkodera: input (augmentowany) -> output (oryginalny)
    # To pomaga modelowi być bardziej odpornym na transformacje
    augmentation = create_augmentation_model()

    def augment_and_pair(x):
        """Augmentuje input, ale output pozostaje oryginalny"""
        augmented = augmentation(x, training=True)
        # Clip values to [0, 1] range after augmentation
        augmented = tf.clip_by_value(augmented, 0.0, 1.0)
        return (augmented, x)

    train_ds_augmented = train_ds.map(augment_and_pair)

    # Zbiór walidacyjny bez augmentacji (input = output dla autoenkodera)
    val_ds = val_ds.map(lambda x: (x, x))

    # Sprawdź czy jest wystarczająco dużo danych dla walidacji
    has_validation_data = val_size > 0

    # Budowa modelu
    print(f"\n🏗️  Budowa autoenkodera...")
    print(f"   Rozmiar obrazków: {image_size}")
    print(f"   Wymiar latentny: {latent_dim}")
    print(f"   Kanały: 3 (RGB)")

    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=(*image_size, 3), latent_dim=latent_dim
    )

    autoencoder.summary()

    # Callbacks - tylko jeśli są dane walidacyjne
    callbacks = []
    if has_validation_data:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
            ),
        ]
    else:
        print("   ⚠️  Za mało danych dla walidacji - trenuję bez zbioru walidacyjnego")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=5, min_lr=1e-7
            ),
        ]

    # Trening
    print("\n🚀 Rozpoczynam trening...")
    print(f"   Epoki: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Augmentacja: Włączona (tylko dla zbioru treningowego)\n")

    # Fit z walidacją tylko jeśli są dane walidacyjne
    if has_validation_data:
        history = autoencoder.fit(
            train_ds_augmented,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = autoencoder.fit(
            train_ds_augmented,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # Zapis modeli
    print("\n💾 Zapisuję modele...")
    Path("models").mkdir(parents=True, exist_ok=True)

    autoencoder.save("models/autoencoder.keras")
    encoder.save("models/encoder.keras")
    decoder.save("models/decoder.keras")

    print("   ✅ Autoenkoder zapisany: models/autoencoder.keras")
    print("   ✅ Encoder zapisany: models/encoder.keras")
    print("   ✅ Decoder zapisany: models/decoder.keras")

    # Zapis historii treningu
    print("\n📊 Zapisuję wyniki...")
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Sprawdź czy są dane walidacyjne
    has_validation = "val_loss" in history.history

    # Zapis wykresów
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    if has_validation:
        axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training History - Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["mae"], label="Train MAE")
    if has_validation and "val_mae" in history.history:
        axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Training History - MAE")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=150, bbox_inches="tight")
    print("   ✅ Wykres zapisany: outputs/training_history.png")

    # Zapis metryk
    final_train_loss = history.history["loss"][-1]
    final_train_mae = history.history["mae"][-1]

    print(f"\n📈 Wyniki końcowe:")
    print(f"   Train Loss (MSE): {final_train_loss:.6f}")
    print(f"   Train MAE: {final_train_mae:.6f}")

    if has_validation:
        final_val_loss = history.history["val_loss"][-1]
        print(f"   Val Loss (MSE): {final_val_loss:.6f}")
        if "val_mae" in history.history:
            final_val_mae = history.history["val_mae"][-1]
            print(f"   Val MAE: {final_val_mae:.6f}")
    else:
        print(f"   ⚠️  Brak danych walidacyjnych (za mało danych)")

    return autoencoder, encoder, decoder, history
