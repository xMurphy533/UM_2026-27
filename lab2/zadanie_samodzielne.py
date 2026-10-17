from sklearn.datasets import load_iris

from lab2.MyPCA import MyPCA

iris = load_iris()
moje_PCA = MyPCA()

X_3D = moje_PCA.redukcja_do_3_wymiarow(iris)
X_2D = moje_PCA.redukcja_do_2_wymiarow(iris)

print("Wartości własne:")
print(moje_PCA.glowne_skladowe)

print("Udział zmienności:")
print(moje_PCA.udzial_zmiennosci)

print("Udział skumulowany:")
print(moje_PCA.udzial_zmiennosci.sum())

print("Generuje wykres 2D ...")
moje_PCA.wykres_2D(X_2D)

print("Generuje wykres 3D ...")
moje_PCA.wykres_3D(X_3D)