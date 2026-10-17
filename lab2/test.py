from sklearn.datasets import load_iris
from MyPCA import MyPCA
from sklearn.decomposition import PCA

iris = load_iris()

moje_pca = MyPCA()
X_mojePCA = moje_pca.redukcja_do_2_wymiarow(iris)

sklearn_pca = PCA(n_components=2)
X = iris.data

X_sklearn = sklearn_pca.fit_transform(X)

print("Udział zmienności z biblioteki sklearn:")
print(sklearn_pca.explained_variance_ratio_)
print("\nPierwsze 5 obserwacji z biblioteki sklearn:")
print(X_sklearn[: 5])

print("\nUdział zmienności z własnej implementacji:")
print(moje_pca.udzial_zmiennosci[: 2])
print("\nPierwsze 5 obserwacji z własnej implementacji:")
print(X_mojePCA[: 5])