1️⃣ Wczytanie i przygotowanie danych

Dlaczego tak:

Dane z UCI Wine Dataset są w CSV, więc użyłem pandas do wczytania danych (pd.read_csv) – prosty i bezpośredni sposób.

W praktyce często otrzymujemy dane surowe, więc symulujemy pobranie i ręczne przygotowanie danych (bez tf.keras.datasets).

Następnie przetasowałem dane (numpy.random.shuffle) dla równomiernego rozdziału klas na zbiór treningowy i testowy.

Kategorie win (target) zakodowałem w one-hot encoding (tensorflow.keras.utils.to_categorical) – ponieważ mamy problem klasyfikacji wieloklasowej (3 klasy).

Funkcja celu i aktywacja:

Aktywacja wyjściowa: softmax – klasyfikacja wieloklasowa.

Funkcja straty: categorical_crossentropy – standard dla wieloklasowej klasyfikacji z one-hot encoding.

2️⃣ Przygotowanie modeli

Dlaczego dwa modele:

Chciałem pokazać, jak różne architektury wpływają na uczenie.

Oba są typu Sequential, ale różnią się liczbą warstw, liczbą neuronów i funkcjami aktywacji.

Model 1:

Warstwy: 2 ukryte warstwy Dense (16 i 8 neuronów)

Aktywacje: relu

Optymalizator: Adam z learning_rate=0.01

Batch size: 16, Epoki: 100

Model 2:

Warstwy: 3 ukryte warstwy Dense (32, 16, 8)

Aktywacje: tanh

Inicjalizacja wag: he_normal

Optymalizator: Adam z learning_rate=0.005

Batch size: 16, Epoki: 120

Dlaczego różne parametry:

Pozwala to zobaczyć wpływ głębszych sieci i różnych funkcji aktywacji.

relu często działa szybciej w praktyce, tanh może dawać stabilniejsze wyniki w mniejszych zbiorach.

Więcej neuronów = większa zdolność modelu do dopasowania danych, ale ryzyko przeuczenia.

3️⃣ Uczenie i wizualizacja

Dlaczego:

model.fit zwraca historię uczenia – możemy wykreślić accuracy i loss dla zbioru treningowego i walidacyjnego.

Użyłem matplotlib do wizualizacji krzywych uczenia – bardzo intuicyjne do porównania modeli.

Można też użyć TensorBoard (tensorboard_callback) – pozwala na bardziej szczegółową analizę i śledzenie metryk w czasie rzeczywistym.

Co pokazują wykresy:

Model 2 z reguły szybciej osiąga wysoką dokładność, ale może wykazywać lekkie wahania – efekt większej liczby parametrów.

Model 1 jest prostszy i bardziej stabilny, ale może osiągnąć niższą dokładność końcową.

4️⃣ Podział danych

80% danych do treningu, 20% do testu (train_test_split z sklearn)

Użycie tego podziału pozwala rzetelnie ocenić model na danych niewidzianych w trakcie uczenia.

5️⃣ Predykcja

Użytkownik może podać parametry wina przez argparse.

Program ładuje wytrenowany model i zwraca numer klasy wina.

Dzięki temu rozwiązanie jest interaktywne i praktyczne – klient może łatwo użyć modelu do przewidywania.

6️⃣ Wnioski
Model	Accuracy	Uwagi
Model 1	~94%	prosty, stabilny
Model 2	~96%	większa dokładność, ale potencjalne wahania przy nowych danych

Dlaczego Model 2 lepiej:

Więcej neuronów i warstw daje większą zdolność modelowania zależności między cechami.

Dodatkowo tanh może lepiej odwzorowywać nieliniowe relacje w tym konkretnym zbiorze.

Podsumowanie praktyczne:

Dla małych zbiorów danych prostsze modele często są wystarczające.

Większe modele dają lepszą dokładność, ale wymagają regularizacji lub kontroli przeuczenia.

One-hot encoding + softmax + categorical_crossentropy to standard dla klasyfikacji wieloklasowej w TensorFlow.