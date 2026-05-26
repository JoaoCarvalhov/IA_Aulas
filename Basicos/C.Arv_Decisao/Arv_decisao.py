import pandas as pd # leitura de dados
import numpy as np # vetores e matrizes
from sklearn.model_selection import train_test_split # divisão dos dados
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder # codificação de rótulos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report# avaliação do modelo
from sklearn.tree import export_graphviz
import graphviz

data = pd.read_csv("C:/Projetos/GitHub/IA_Aulas/Basicos/B.Naive_Bayes/insurance.csv", keep_default_na=False)
#print(data)
data = data.drop(columns=["Unnamed: 0"])
#print(data)
X = data.drop(columns=["Accident"]).values
y = data["Accident"].values
#print(X)
#print(y)
labelencoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:, i] = labelencoder.fit_transform(X[:, i])
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

modelo = DecisionTreeClassifier(random_state=1, max_depth=8, max_leaf_nodes=6) # cria modelo de árvore de decisão, limitando a profundidade e o número de folhas 
                                                                               # para evitar overfitting
modelo.fit(X_train, y_train)
7
dot_data = export_graphviz(modelo, out_file=None, filled=True, feature_names=data.columns[:-1], class_names=True, rounded=True)
y_pred = modelo.predict(X_test) # faz previsões com os dados de teste
print("Acurácia:", accuracy_score(y_test, y_pred)) # avalia a acurácia do modelo
print("Precisão:", precision_score(y_test, y_pred, average='weighted')) # avalia a precisão do modelo
print("Recall:", recall_score(y_test, y_pred, average='weighted')) # avalia o recall do modelo 
print("F1-Score:", f1_score(y_test, y_pred, average='weighted')) # avalia o F1-Score do modelo 
print("Relatório de Classificação:\n", classification_report(y_test, y_pred)) # exibe o relatório de cl