import numpy as np
import matplotlib.pyplot as plt

class MyPCA:
    def __init__(self):
        self.srednie_cech = None
        self.dane_wycentrowane = None
        self.macierz_kowariancji = None
        self.glowne_skladowe = None
        self.wektory_wlasne = None
        self.etykiety = None
        self.nazwy_kwiatow = None
        self.udzial_zmiennosci = None

    def redukcja_do_2_wymiarow(self, iris):
        dane = iris.data
        etykiety = iris.target
        self.etykiety = etykiety
        nazwy_kwiatow = iris.target_names
        self.nazwy_kwiatow = nazwy_kwiatow
        dane_srednia = np.mean(dane, axis=0)
        self.srednie_cech = dane_srednia
        C = dane - dane_srednia
        self.dane_wycentrowane = C

        rozmiar = C.shape[0]
        macierz_kowariancji = (C.T @ C) / (rozmiar - 1)
        self.macierz_kowariancji = macierz_kowariancji

        wartosci_wlasne, wektory_wlasne = np.linalg.eigh(macierz_kowariancji)
        self.glowne_skladowe = wartosci_wlasne[:: -1]
        self.wektory_wlasne = wektory_wlasne
        self.udzial_zmiennosci = (wartosci_wlasne / wartosci_wlasne.sum())
        self.udzial_zmiennosci = self.udzial_zmiennosci[:: -1]

        kolejnosc = np.argsort(wartosci_wlasne)[:: -1]
        wektory_wlasne_posortowane = wektory_wlasne[:, kolejnosc]

        V_2 = wektory_wlasne_posortowane[:, :2]
        wynik_2D = C @ V_2

        return wynik_2D

    def redukcja_do_3_wymiarow(self, iris):
        dane = iris.data
        etykiety = iris.target
        self.etykiety = etykiety
        nazwy_kwiatow = iris.target_names
        self.nazwy_kwiatow = nazwy_kwiatow

        dane_srednia = np.mean(dane, axis=0)
        self.srednie_cech = dane_srednia
        C = dane - dane_srednia
        self.dane_wycentrowane = C

        rozmiar = C.shape[0]
        macierz_kowariancji = (C.T @ C) / (rozmiar - 1)
        self.macierz_kowariancji = macierz_kowariancji

        wartosci_wlasne, wektory_wlasne = np.linalg.eigh(macierz_kowariancji)
        self.glowne_skladowe = wartosci_wlasne
        self.wektory_wlasne = wektory_wlasne

        kolejnosc = np.argsort(wartosci_wlasne)[:: -1]
        wektory_wlasne_posortowane = wektory_wlasne[:, kolejnosc]

        V_3 = wektory_wlasne_posortowane[:, :3]
        wynik_3D = C @ V_3

        return wynik_3D

    def wykres_2D(self, wynik_2D):
        x = wynik_2D[:, 0]
        y = wynik_2D[:, 1]
        kolory = ['red', 'blue', 'green']
        for i in range(3):
            punkty_x = x[self.etykiety == i]
            punkty_y = y[self.etykiety == i]
            plt.scatter(punkty_x, punkty_y, colorizer=kolory, label=self.nazwy_kwiatow[i])
        plt.title("PCA - Zbiór Iris")
        plt.xlabel("Pierwsza główna składowa")
        plt.ylabel("Druga główna składowa")
        plt.legend()
        plt.grid()
        plt.show()

    def wykres_3D(self, wynik_3D):
        wykres_3d = plt.axes(projection='3d')
        kolory = ['red', 'blue', 'green']
        for etykieta, nazwa in enumerate(self.nazwy_kwiatow):
            mask = self.etykiety == etykieta
            wykres_3d.scatter(
                wynik_3D[mask, 0],
                wynik_3D[mask, 1],
                wynik_3D[mask, 2],
                label=nazwa,
                colorizer= kolory
            )
        wykres_3d.set_xlabel("PC1")
        wykres_3d.set_ylabel("PC2")
        wykres_3d.set_zlabel("PC3")
        wykres_3d.set_title("PCA dla zbioru IRIS w 3D")
        wykres_3d.legend()
        plt.show()