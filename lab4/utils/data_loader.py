import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_and_prepare_data(path):
    columns = [
        "class","Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
        "Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins",
        "Color_intensity","Hue","OD280/OD315_of_diluted_wines","Proline"
    ]
    data = pd.read_csv(path, header=None, names=columns)
    data = data.sample(frac=1, random_state=42)

    X = data.drop("class", axis=1).values
    y = data["class"].values.reshape(-1,1)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Normalizacja z Keras
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(X_train)

    return X_train, X_test, y_train, y_test, normalizer, encoder, columns[1:]
