from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

# O KNN é um algoritmo de classificação que classifica um novo ponto de dados com base na maioria dos rótulos dos seus vizinhos mais próximos.

data = pd.read_csv("C:/projetos/GitHub/IA_Aulas/Basicos/A.Reg_Linear/mt_cars.csv")
print(data)

X = data[["mpg","hp"]].values #usando o miles per galon e o potency para
y = data["cyl"].values        #prever a cilindrada

Modelo = KNeighborsClassifier(n_neighbors=3) #é bom que o numero de k serja impar para não haver empate
Modelo = Modelo.fit(X, y)

y_prev = Modelo.predict(X) # Avaliação do modelo
print("Acurácia:", accuracy_score(y, y_prev))
print("Precisão:", precision_score(y, y_prev, average='weighted'))
print("Recall:", recall_score(y, y_prev, average='weighted'))
print("F1-Score:", f1_score(y, y_prev, average='weighted'))
print("Matriz de Confusão:\n", confusion_matrix(y, y_prev))


print("--------------------------------------")
print("teste com o dado 19.3 de mgp e 105 de hp:")
teste = np.array([[19.3, 105]])
print(f'previsão do modelo: {Modelo.predict(teste)}')
print(f'com k=3 os 3 mais proximos são os de indice:\n{Modelo.kneighbors(teste)}\n Que são:\n{data.loc[[1,5,31],["cyl", "mpg", "hp"]]}')