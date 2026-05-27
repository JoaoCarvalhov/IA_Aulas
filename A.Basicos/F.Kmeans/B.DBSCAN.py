from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

iris = datasets.load_iris()
print(iris.data)
dbscan = DBSCAN(eps=0.5, min_samples=5) # eps é a distância máxima entre dois pontos para que eles sejam considerados vizinhos, e 
                                        # min_samples é o número mínimo de pontos necessários para formar um cluster.
print(dbscan.fit_predict(iris.data)) # fit_predict é um método que ajusta o modelo aos dados e retorna os rótulos dos clusters 
                                     # para cada ponto de dados.
