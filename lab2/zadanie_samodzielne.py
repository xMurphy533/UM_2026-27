from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from lab2.MyPCA import MyPCA

iris = load_iris()
X = iris.data
moje_PCA_2D = MyPCA(k=2)
moje_PCA_3D = MyPCA(k=3)

X_3D = moje_PCA_3D.redukcja(X)
X_2D = moje_PCA_2D.redukcja(X)

print("Wartości własne 2D:")
print(moje_PCA_2D.glowne_skladowe)

print("Udział zmienności 2D:")
print(moje_PCA_2D.udzial_zmiennosci)

print("Udział skumulowany 2D:")
print(moje_PCA_2D.udzial_zmiennosci.sum())

print("Generuje wykres 2D ...")
x = X_2D[:, 0]
y = X_2D[:, 1]
kolory = ['red', 'blue', 'green']
for i in range(3):
    punkty_x = x[iris.target == i]
    punkty_y = y[iris.target == i]
    plt.scatter(punkty_x, punkty_y, colorizer=kolory, label=iris.target_names[i])
plt.title("PCA - Zbiór Iris")
plt.xlabel("Pierwsza główna składowa")
plt.ylabel("Druga główna składowa")
plt.legend()
plt.grid()
plt.show()

print("Generuje wykres 3D ...")
wykres_3d = plt.axes(projection='3d')
kolory = ['red', 'blue', 'green']
for etykieta, nazwa in enumerate(iris.target_names):
    mask = iris.target == etykieta
    wykres_3d.scatter(
            X_3D[mask, 0],
            X_3D[mask, 1],
            X_3D[mask, 2],
            label=nazwa,
            colorizer=kolory
        )
wykres_3d.set_xlabel("PC1")
wykres_3d.set_ylabel("PC2")
wykres_3d.set_zlabel("PC3")
wykres_3d.set_title("PCA dla zbioru IRIS w 3D")
wykres_3d.legend()
plt.show()