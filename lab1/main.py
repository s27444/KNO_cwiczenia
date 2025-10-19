import sys
from pathlib import Path
import argparse

import tensorflow as tf
from PIL import Image


def build_mnist_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def ensure_parent_dir(path: Path) -> None:
    if path.parent.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def train(save_path: Path, epochs: int) -> None:
    (x_train, y_train), _ = load_mnist()
    model = build_mnist_model()
    model.fit(x_train, y_train, epochs=epochs, verbose=1)
    ensure_parent_dir(save_path)
    model.save(save_path)
    print(f"Model zapisany: {save_path}")


def evaluate(model_path: Path) -> None:
    if not model_path.exists():
        print(f"Brak modelu: {model_path}")
        sys.exit(1)
    _, (x_test, y_test) = load_mnist()
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Wynik: loss={loss:.4f}, accuracy={acc:.4f}")


def preprocess_image(image_path: Path) -> tf.Tensor:
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    arr = tf.convert_to_tensor(img, dtype=tf.float32)
    arr = arr / 255.0
    arr = tf.expand_dims(arr, axis=0)
    return arr


def predict(model_path: Path, image_path: Path) -> None:
    if not model_path.exists():
        print(f"Brak modelu: {model_path}")
        sys.exit(1)
    if not image_path.exists():
        print(f"Brak pliku obrazu: {image_path}")
        sys.exit(1)
    model = tf.keras.models.load_model(model_path)
    inp = preprocess_image(image_path)
    probs = model.predict(inp, verbose=0)[0]
    pred = int(tf.argmax(probs).numpy())
    confidence = float(tf.reduce_max(probs).numpy())
    print(f"Predykcja: {pred} (pewność {confidence:.4f})")


def cpu_test() -> None:
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lab1",
        description="Lab1: MNIST trening, ewaluacja i predykcja",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Trenuj model i zapisz do pliku")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument(
        "--save",
        type=Path,
        default=Path("models/mnist.keras"),
    )
    p_train.add_argument(
        "--use-existing",
        action="store_true",
        help="Użyj istniejącego modelu jeśli plik istnieje",
    )

    p_eval = sub.add_parser("evaluate", help="Oceń model na zbiorze testowym")
    p_eval.add_argument(
        "--model",
        type=Path,
        default=Path("models/mnist.keras"),
    )

    p_pred = sub.add_parser("predict", help="Rozpoznaj cyfrę z pliku obrazu")
    p_pred.add_argument(
        "--model",
        type=Path,
        default=Path("models/mnist.keras"),
    )
    p_pred.add_argument("--image", type=Path, required=True)

    sub.add_parser("cpu-test", help="Test działania TensorFlow (CPU)")

    args = parser.parse_args()

    match args.cmd:
        case "train":
            save_path: Path = args.save
            if args.use_existing and save_path.exists():
                print(f"Model istnieje, pomijam trening: {save_path}")
            else:
                train(save_path, args.epochs)
            evaluate(save_path)
            return 0

        case "evaluate":
            evaluate(args.model)
            return 0

        case "predict":
            predict(args.model, args.image)
            return 0

        case "cpu-test":
            cpu_test()
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
