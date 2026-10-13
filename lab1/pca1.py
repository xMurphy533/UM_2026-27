import numpy as np
from sklearn.datasets import load_iris
def pca_components(X):
    srednia_kazdej_cechy = np.mean(X, axis=0)
    wycentrowane_dane = X - srednia_kazdej_cechy
    n = wycentrowane_dane.shape[0]
    macierz_kowariancji = (wycentrowane_dane.T @ wycentrowane_dane) / (n - 1)
    wartosci_wlasne, wektory_wlasne = np.linalg.eigh(macierz_kowariancji)
    kolejnosc = np.argsort(wartosci_wlasne)[::-1]
    wartosci_wlasne_posortowane = wartosci_wlasne[kolejnosc]
    wektory_wlasne_posortowane = wektory_wlasne[:, kolejnosc]
    udzial = (wartosci_wlasne_posortowane / wartosci_wlasne_posortowane.sum())
    return {
        "mean": srednia_kazdej_cechy.shape,
        "X_centered": wycentrowane_dane.shape,
        "covariance_matrix": macierz_kowariancji.shape,
        "eigenvalues": wartosci_wlasne_posortowane.shape,
        "eigenvectors": wektory_wlasne_posortowane.shape,
        "explained_variance_ratio": udzial
    }

dane = load_iris()
wynik = pca_components(dane.data)

print(wynik)