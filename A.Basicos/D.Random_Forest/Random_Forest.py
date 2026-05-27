# Random forest é um algoritmo de aprendizado de máquina que utiliza múltiplas árvores de decisão para melhorar a precisão e evitar o overfitting. 
# Ele funciona criando várias árvores de decisão a partir de amostras aleatórias dos dados e, em seguida, combina as previsões dessas árvores para obter uma previsão final 
# mais robusta.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = pd.read_csv("C:/projetos/GitHub/IA_Aulas/A.Basicos/B.Naive_Bayes/insurance.csv", keep_default_na=False)
data = data.drop(columns=['Unnamed: 0'], index=1)

X = data.drop(columns=["Accident"]).values

y = data["Accident"].values

#print(X)
#print(y)

labelencoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:, i] = labelencoder.fit_transform(X[:, i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

modelo = RandomForestClassifier(random_state=1, max_depth=20, max_leaf_nodes=12, n_estimators=500) # essa parte é bem parecida com arvore de decisão
modelo.fit(X_train, y_train)                                                                       # n_estimators define a quantidade de arvores que vão ser "treinadas" 

tree_index = 0
tree_to_visualise = modelo.estimators_[tree_index]
#plt.figure(figsize=(20,20))
#plot_tree(tree_to_visualise, filled=True, feature_names= data.columns[:-1], class_names=True, rounded=True)
#plt.show()

previsao = modelo.predict(X_test)
print(accuracy_score(y_test, previsao))
print(precision_score(y_test, previsao, average="weighted"))
print(recall_score(y_test, previsao, average="weighted"))
print(f1_score(y_test, previsao, average="weighted"))
print()
print(classification_report(y_test, previsao))