import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from models.model_cnn import build_model as build_cnn_model
from models.model_fc import build_model as build_fc_model
from sklearn.metrics import confusion_matrix

# Nazwy klas Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist():
    """Ładuje dane Fashion MNIST"""
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalizacja do zakresu [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape dla CNN (dodanie wymiaru kanału)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def create_augmentation_layer():
    """Tworzy warstwę augmentacji danych"""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )


def train_model(
    model, x_train, y_train, x_test, y_test, epochs=10, use_augmentation=False
):
    """Trenuje model i zwraca historię oraz metryki"""
    if use_augmentation:
        # Dodajemy augmentację do modelu
        augmentation = create_augmentation_layer()
        model_with_aug = tf.keras.Sequential([augmentation, model])
        model_with_aug.compile(
            optimizer=model.optimizer, loss=model.loss, metrics=model.metrics_names
        )
        history = model_with_aug.fit(
            x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1
        )
        # Używamy oryginalnego modelu do ewaluacji
        model.set_weights(model_with_aug.layers[1].get_weights())
    else:
        history = model.fit(
            x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1
        )

    # Ewaluacja
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Macierz pomyłek
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": cm,  # Zostawiamy jako numpy array dla łatwiejszej obsługi
    }

    return history, metrics


def save_metrics(metrics, filepath):
    """Zapisuje metryki do pliku JSON"""
    # Konwertujemy numpy array do listy jeśli potrzeba
    cm = metrics["confusion_matrix"]
    if isinstance(cm, np.ndarray):
        cm_shape = list(cm.shape)
    else:
        # Jeśli już lista, konwertujemy do numpy żeby sprawdzić shape
        cm_array = np.array(cm)
        cm_shape = list(cm_array.shape)
        cm = cm_array

    # Zapisujemy tylko liczby, nie całą macierz (może być duża)
    metrics_to_save = {
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "confusion_matrix_shape": cm_shape,
    }

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    # Zapisujemy pełną macierz pomyłek do osobnego pliku
    cm_file = filepath.replace(".json", "_confusion_matrix.txt")
    np.savetxt(cm_file, cm, fmt="%d")
    print(f"Macierz pomyłek zapisana: {cm_file}")


def train_models(epochs=10, use_augmentation=False):
    """Główna funkcja treningu - trenuje oba modele"""
    print("Ładowanie danych Fashion MNIST...")
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    print(f"Zbiór treningowy: {x_train.shape[0]} przykładów")
    print(f"Zbiór testowy: {x_test.shape[0]} przykładów")

    # Trening modelu Fully Connected
    print("\n" + "=" * 50)
    print("Trening modelu Fully Connected...")
    print("=" * 50)
    model_fc = build_fc_model()
    history_fc, metrics_fc = train_model(
        model_fc,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
        use_augmentation=use_augmentation,
    )

    model_fc.save("models/fashion_mnist_fc.keras")
    print(f"\nModel FC zapisany: models/fashion_mnist_fc.keras")
    save_metrics(metrics_fc, "outputs/fashion_mnist_fc_metrics.json")
    print(f"Test loss: {metrics_fc['test_loss']:.4f}")
    print(f"Test accuracy: {metrics_fc['test_accuracy']:.4f}")

    # Trening modelu CNN
    print("\n" + "=" * 50)
    print("Trening modelu Convolutional...")
    print("=" * 50)
    model_cnn = build_cnn_model()
    history_cnn, metrics_cnn = train_model(
        model_cnn,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
        use_augmentation=use_augmentation,
    )

    model_cnn.save("models/fashion_mnist_cnn.keras")
    print(f"\nModel CNN zapisany: models/fashion_mnist_cnn.keras")
    save_metrics(metrics_cnn, "outputs/fashion_mnist_cnn_metrics.json")
    print(f"Test loss: {metrics_cnn['test_loss']:.4f}")
    print(f"Test accuracy: {metrics_cnn['test_accuracy']:.4f}")

    # Wybierz lepszy model
    if metrics_cnn["test_accuracy"] > metrics_fc["test_accuracy"]:
        best_model = model_cnn
        best_metrics = metrics_cnn
        best_name = "CNN"
    else:
        best_model = model_fc
        best_metrics = metrics_fc
        best_name = "FC"

    best_model.save("models/fashion_mnist_best.keras")
    save_metrics(best_metrics, "outputs/fashion_mnist_best_metrics.json")
    print(f"\n{'='*50}")
    print(f"Lepszy model: {best_name}")
    print(f"Zapisano jako: models/fashion_mnist_best.keras")
    print(f"{'='*50}")

    if use_augmentation:
        print("\nUwaga: Użyto augmentacji danych podczas treningu.")

    return best_model, best_metrics
