import numpy as np


class MyPCA:
    def __init__(self, k):
        self.k = k
        self.srednie_cech = None
        self.dane_wycentrowane = None
        self.macierz_kowariancji = None
        self.glowne_skladowe = None
        self.wektory_wlasne = None
        self.udzial_zmiennosci = None

    def redukcja(self, X):
        srednie = np.mean(X, axis=0)
        self.srednie_cech = srednie
        C = X - srednie
        self.dane_wycentrowane = C
        n = C.shape[0]
        macierz_kowariancji = (C.T @ C) / (n - 1)
        self.macierz_kowariancji = macierz_kowariancji
        wartosci_wlasne, wektory_wlasne = np.linalg.eigh(macierz_kowariancji)
        self.glowne_skladowe = wartosci_wlasne[:: -1]
        self.wektory_wlasne = wektory_wlasne
        self.udzial_zmiennosci = (wartosci_wlasne / wartosci_wlasne.sum())
        self.udzial_zmiennosci = self.udzial_zmiennosci[:: -1]
        kolejnosc = np.argsort(wartosci_wlasne)[:: -1]
        eigenvectors_sorted = wektory_wlasne[:, kolejnosc]
        V_k = eigenvectors_sorted[:, :self.k]
        wynik_kD = C @ V_k
        return wynik_kD