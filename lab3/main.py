from sklearn.datasets import make_blobs
import numpy as np
from lab3.MySVM import LinearSVM

RANDOM_STATE = 42

X, y_01 = make_blobs(
    n_samples=60,
    centers=[(-2.5, -2.0), (2.5, 2.0)],
    cluster_std=0.9,
    random_state=RANDOM_STATE
)

y = np.where(y_01 == 0, -1, 1)

model = LinearSVM(tempo_uczenia=0.001, C=1.0, ilosc_iteracji=3000)
model.fit(X, y)

predykcje = model.predict(X)
wartosci_decyzyjne = model.decision_function(X)

print("Czy w ma tyle samo elementów ile jest cech?")
print(len(model.w) == X.shape[1])

print("Czy predict(X) zwraca tyle etykiet ile jest obserwacji?")
print(len(predykcje) == X.shape[0])

unikalne_klasy = np.unique(predykcje)
print("Czy predykcje należą wyłącznie do {-1, 1}?")
print(set(unikalne_klasy).issubset({-1, 1}))

print("Czy decision_function(X) zwraca po jednej wartości na obserwację?")
print(len(wartosci_decyzyjne) == X.shape[0])

dokladnosc = np.mean(predykcje == y)
print("Czy model potrafi rozdzielić prosty zbiór liniowo separowalny?")
print("Dokładność klasyfikacji (%): ", {dokladnosc * 100})

model_duze_c = LinearSVM(tempo_uczenia=0.001, C=100.0, ilosc_iteracji=3000)
model_duze_c.fit(X, y)
print("Co dzieje się po zmianie C?")
print(f"Norma w dla C=1.0: {np.linalg.norm(model.w):.4f}")
print(f"Norma w dla C=100.0: {np.linalg.norm(model_duze_c.w):.4f}")

model_szybki = LinearSVM(tempo_uczenia=0.5, C=1.0, ilosc_iteracji=3000)
model_szybki.fit(X, y)
print("Co dzieje się po zmianie learning_rate?")
print(f"Duży learning_rate, wektor w: {model_szybki.w}")

model_malo_iteracji = LinearSVM(tempo_uczenia=0.001, C=1.0, ilosc_iteracji=10)
model_malo_iteracji.fit(X, y)
dokladnosc_malo_iter = np.mean(model_malo_iteracji.predict(X) == y)
print("Co dzieje się po zmianie liczby iteracji?")
print(f"Dokładność przy 10 iteracjach: {dokladnosc_malo_iter * 100}%")
