from models import model_deep, model_simple
from utils.data_loader import load_and_prepare_data
from utils.plot_utils import plot_histories
import pickle


def train_models():
    local_data_path = "data/wine.data"
    X_train, X_test, y_train, y_test, scaler, encoder, columns = \
        load_and_prepare_data(local_data_path)

    print(">>> Trening Modelu 1 (prosty)...")
    model1 = model_simple.build_model(X_train.shape[1])
    history1 = model1.fit(
        X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0
    )

    print(">>> Trening Modelu 2 (złożony)...")
    model2 = model_deep.build_model(X_train.shape[1])
    history2 = model2.fit(
        X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0
    )

    plot_histories([history1, history2], ["Model prosty", "Model złożony"])

    acc1 = model1.evaluate(X_test, y_test, verbose=0)[1]
    acc2 = model2.evaluate(X_test, y_test, verbose=0)[1]

    print(f"\nModel 1 – dokładność testowa: {acc1:.4f}")
    print(f"Model 2 – dokładność testowa: {acc2:.4f}")

    better_model = model2 if acc2 > acc1 else model1
    better_model.save("best_model.h5")
    print(f"\nZapisano lepszy model jako best_model.h5")

    # Zapis obiektów scaler i encoder
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("Zapisano scaler.pkl i encoder.pkl")

    return better_model, scaler, encoder, columns
