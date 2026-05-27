from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# O Agglomerative Clustering é um algoritmo de agrupamento hierárquico que constrói uma hierarquia de clusters.
# O algoritmo começa com cada ponto de dados como um cluster separado e, em seguida, mescla os clusters mais próximos 
# iterativamente até que o número desejado de clusters seja alcançado. 
# O Agglomerative Clustering é um algoritmo de aprendizado não supervisionado, pois não requer rótulos para os dados.
iris = datasets.load_iris()
print(iris.data)

agglo = AgglomerativeClustering(n_clusters=3)
print(agglo.fit(iris.data).labels_)
print("------------------------")
print(confusion_matrix(iris.target, agglo.fit(iris.data).labels_))