import numpy as np


class LinearSVM:
    def __init__(self, tempo_uczenia, C, ilosc_iteracji):
        self.w = None
        self.b = None
        self.C = C
        self.tempo_uczenia = tempo_uczenia
        self.ilosc_iteracji = ilosc_iteracji

    def fit(self, macierz_cech, klasy):
        ilosc_probek, ilosc_cech = macierz_cech.shape
        self.w = np.zeros(ilosc_cech)
        self.b = 0.0
        for i in range(self.ilosc_iteracji):
            for indeks, x in enumerate(macierz_cech):
                warunek = klasy[indeks] * (np.dot(x, self.w) + self.b)
                if warunek >= 1:
                    self.w -= self.tempo_uczenia * self.w
                else:
                    self.w -= self.tempo_uczenia * (self.w - self.C * klasy[indeks] * x)
                    self.b -= self.tempo_uczenia * (-self.C * klasy[indeks])
        return self

    def decision_function(self, X):
        wartosci_decyzyjne = X @ self.w + self.b
        return wartosci_decyzyjne

    def predict(self, X):
        wartosci_decyzyjne = self.decision_function(X)
        predykcje = np.where(wartosci_decyzyjne >= 0, 1, -1)
        return predykcje
