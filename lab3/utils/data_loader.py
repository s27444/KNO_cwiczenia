import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_prepare_data(url: str):
    columns = ['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash',
               'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
               'Proanthocyanins', 'Color_intensity', 'Hue',
               'OD280/OD315_of_diluted_wines', 'Proline']

    data = pd.read_csv(url, header=None, names=columns)
    data = data.sample(frac=1, random_state=42)

    X = data.drop('class', axis=1).values
    y = data['class'].values.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, encoder, columns
