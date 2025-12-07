import pickle
import numpy as np
from kerastuner.tuners import RandomSearch
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Normalization

from utils.data_loader import load_and_prepare_data
from models.model_tuner import build_model_hp
from models import model_deep


def print_comparison(y_true, y_pred_baseline, y_pred_tuned, best_hp=None):
    acc_baseline = (y_pred_baseline == y_true).mean()
    acc_tuned = (y_pred_tuned == y_true).mean()

    print("\n=== PORÓWNANIE MODELI ===")
    print(f"Dokładność baseline: {acc_baseline:.4f}")
    print(f"Dokładność tunera:  {acc_tuned:.4f}\n")

    print("Macierz pomyłek - baseline:")
    print(confusion_matrix(y_true, y_pred_baseline))

    print("\nMacierz pomyłek - tuner:")
    print(confusion_matrix(y_true, y_pred_tuned))

    if best_hp:
        print("\nNajlepsze hiperparametry tunera:")
        for param in best_hp.values:
            print(f"{param}: {best_hp.get(param)}")


def train_tuner():
    # ============================
    # 1) Załaduj dane
    # ============================
    X_train, X_test, y_train, y_test, normalizer_unused, encoder, columns = \
        load_and_prepare_data("data/wine.data")

    # ============================
    # 2) Stwórz i dopasuj normalizer
    # ============================
    normalizer = Normalization()
    normalizer.adapt(X_train)

    # ============================
    # 3) Baseline (model klasyczny)
    # ============================
    print("Trening baseline (model z normalizerem w środku)...")

    baseline_model = model_deep.build_model(X_train.shape[1], normalizer)

    try:
        baseline_model.load_weights("best_model.h5")
        print("Załadowano wagę baseline modelu z best_model.h5")
    except:
        print("Nie znaleziono best_model.h5. Trenuję baseline od nowa...")
        baseline_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50, batch_size=16, verbose=0
        )
        baseline_model.save("best_model.h5")

    y_pred_baseline = np.argmax(baseline_model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ============================
    # 4) Keras Tuner
    # ============================
    print("\nRozpoczynam tuning Keras Tuner...")

    tuner = RandomSearch(
        lambda hp: build_model_hp(hp, normalizer),
        objective='val_accuracy',
        max_trials=3,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='wine_classification'
    )

    tuner.search(
        X_train, y_train,
        epochs=20,
        validation_split=0.2,
        batch_size=16
    )

    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    # ============================
    # 5) Ewaluacja modelu tunera
    # ============================
    y_pred_tuned = np.argmax(best_model.predict(X_test), axis=1)

    print_comparison(y_true, y_pred_baseline, y_pred_tuned, best_hp)

    # ============================
    # 6) Zapis modelu
    # ============================
    best_model.save("best_model_tuned.keras")

    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("\nZapisano best_model_tuned.keras i encoder.pkl (normalizer zapisany w modelu automatycznie!)")


if __name__ == "__main__":
    train_tuner()
