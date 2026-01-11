import numpy as np
import pandas as pd


def load_time_series_csv(file_path, value_column=None):
    """
    Ładuje szereg czasowy z pliku CSV

    Args:
        file_path: Ścieżka do pliku CSV
        value_column: Nazwa kolumny z wartościami (jeśli None, użyje 'close' lub pierwszej kolumny numerycznej)

    Returns:
        values: Tablica wartości szeregu czasowego (od najstarszych do najnowszych)
        dates: Daty (jeśli dostępne) lub indeksy
        column_name: Nazwa użytej kolumny
    """
    # Próbuj wczytać CSV - sprawdź czy pierwszy wiersz to URL
    try:
        # Wczytaj pierwsze 2 wiersze żeby sprawdzić
        first_lines = pd.read_csv(file_path, nrows=2, header=None)
        first_cell = str(first_lines.iloc[0, 0]).lower() if len(first_lines) > 0 else ""

        # Jeśli pierwszy wiersz zawiera URL, pomiń go i użyj drugiego jako nagłówka
        if (
            "http" in first_cell
            or "www" in first_cell
            or "cryptodatadownload" in first_cell
        ):
            # Pierwszy wiersz to URL, drugi to nagłówek
            df = pd.read_csv(file_path, skiprows=1)
        else:
            # Normalny plik CSV
            df = pd.read_csv(file_path)
    except Exception as e:
        # Jeśli błąd, spróbuj normalnie
        try:
            df = pd.read_csv(file_path, skiprows=1)
        except:
            df = pd.read_csv(file_path)

    # Wyczyść nazwy kolumn ze spacji
    df.columns = df.columns.str.strip()

    # Jeśli nie podano kolumny, użyj 'close' (dla danych giełdowych) lub pierwszej kolumny numerycznej
    if value_column is None:
        # Sprawdź różne warianty nazwy kolumny 'close'
        if "close" in df.columns.str.lower():
            close_col = [col for col in df.columns if col.lower() == "close"][0]
            value_column = close_col
        else:
            # Spróbuj znaleźć kolumny numeryczne
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Jeśli brak numerycznych, spróbuj przekonwertować
            if len(numeric_cols) == 0:
                # Spróbuj przekonwertować kolumny które wyglądają na numeryczne
                for col in df.columns:
                    if col.lower() in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "volume btc",
                        "volume usd",
                    ]:
                        try:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            numeric_cols.append(col)
                        except:
                            pass

                if len(numeric_cols) == 0:
                    raise ValueError(
                        f"Nie znaleziono kolumny numerycznej w CSV. Dostępne kolumny: {df.columns.tolist()}"
                    )

            # Preferuj 'close' jeśli istnieje, w przeciwnym razie pierwsza numeryczna
            close_candidates = [col for col in numeric_cols if "close" in col.lower()]
            if close_candidates:
                value_column = close_candidates[0]
            else:
                value_column = numeric_cols[0]

    # Jeśli jest kolumna z datami, użyj jej
    date_columns = [
        "date",
        "Date",
        "DATE",
        "time",
        "Time",
        "TIME",
        "timestamp",
        "Timestamp",
    ]
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break

    # Upewnij się, że kolumna jest numeryczna
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    # Usuń wartości NaN
    df = df.dropna(subset=[value_column])

    if len(df) == 0:
        raise ValueError(
            f"Brak poprawnych wartości numerycznych w kolumnie '{value_column}'"
        )

    values = df[value_column].values.astype(np.float32)

    # Odwróć kolejność jeśli dane są od najnowszych do najstarszych
    # (dla szeregów czasowych lepiej mieć od najstarszych do najnowszych)
    # Sprawdź czy pierwsza data jest nowsza niż ostatnia
    if date_col and len(df) > 1:
        try:
            dates_parsed = pd.to_datetime(df[date_col])
            if dates_parsed.iloc[0] > dates_parsed.iloc[-1]:
                # Dane są odwrócone - odwróć je
                values = values[::-1]
                dates_parsed = dates_parsed[::-1]
                dates = dates_parsed.values
            else:
                dates = dates_parsed.values
        except:
            dates = np.arange(len(values))
    elif date_col:
        try:
            dates = pd.to_datetime(df[date_col]).values
        except:
            dates = np.arange(len(values))
    else:
        dates = np.arange(len(values))

    return values, dates, value_column


def enrich_features(data, lookback=30):
    """
    Wzbogaca dane o cechy harmoniczne i przesunięcia czasowe

    Args:
        data: Tablica wartości szeregu czasowego
        lookback: Liczba kroków wstecz do uwzględnienia

    Returns:
        enriched_data: Wzbogacone dane z dodatkowymi cechami (n_samples, n_features)
    """
    enriched = []
    n = len(data)

    # 1. Oryginalne wartości
    enriched.append(data)

    # 2. Przesunięcia czasowe (lag features)
    for lag in [1, 2, 3, 7]:
        lagged = np.roll(data, lag)
        lagged[:lag] = data[0]  # Wypełnij początek
        enriched.append(lagged)

    # 3. Średnia krocząca
    window_sizes = [3, 7, 14]
    for window in window_sizes:
        rolling_mean = (
            pd.Series(data).rolling(window=window, min_periods=1).mean().values
        )
        enriched.append(rolling_mean)

    # 4. Różnice (zmiana względem poprzedniej wartości)
    diff1 = np.diff(data, prepend=data[0])
    enriched.append(diff1)

    # 5. Cechy harmoniczne (sin/cos dla cykliczności)
    for period in [7, 30, 365]:  # Tygodniowy, miesięczny, roczny (jeśli dane długie)
        if n > period:
            t = np.arange(n)
            sin_feature = np.sin(2 * np.pi * t / period)
            cos_feature = np.cos(2 * np.pi * t / period)
            enriched.append(sin_feature)
            enriched.append(cos_feature)

    # Łączymy wszystkie cechy
    enriched_data = np.column_stack(enriched)

    return enriched_data


def create_sequences(data, lookback=30, forecast_horizon=1):
    """
    Tworzy sekwencje dla uczenia LSTM

    Args:
        data: Wzbogacone dane (n_samples, n_features)
        lookback: Liczba kroków wstecz (wejście)
        forecast_horizon: Liczba kroków do przodu (wyjście)

    Returns:
        X: Sekwencje wejściowe (n_samples, lookback, n_features)
        y: Wartości docelowe (n_samples, forecast_horizon)
    """
    X, y = [], []

    for i in range(len(data) - lookback - forecast_horizon + 1):
        # Input: lookback kroków z wszystkimi cechami
        X.append(data[i : i + lookback])
        # Output: wartość docelowa (pierwsza cecha - oryginalna wartość)
        y.append(data[i + lookback : i + lookback + forecast_horizon, 0])

    return np.array(X), np.array(y)


def prepare_data_for_fc(X_lstm):
    """
    Przygotowuje dane dla modelu Fully Connected
    Spłaszcza sekwencje do wektorów
    """
    n_samples, lookback, n_features = X_lstm.shape
    X_fc = X_lstm.reshape(n_samples, lookback * n_features)
    return X_fc
