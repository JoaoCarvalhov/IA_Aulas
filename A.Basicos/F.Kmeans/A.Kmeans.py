# No algoritmo de K-means, o número de clusters é definido pelo usuário, 
# e o algoritmo tenta encontrar os centros dos clusters que minimizam a distância entre os pontos de dados e os centros dos clusters. 
# O K-means é um algoritmo de aprendizado não supervisionado, pois não requer rótulos para os dados.

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

iris = datasets.load_iris()
print(iris.data)

kmeans = KMeans(n_clusters=3, n_init="auto")

print(kmeans.fit(iris.data).labels_)
print("------------------------")
print(confusion_matrix(iris.target, kmeans.fit(iris.data).labels_))