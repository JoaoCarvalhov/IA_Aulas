import pandas as pd # leitura de dados
import numpy as np # vetores e matrizes
from sklearn.model_selection import train_test_split # divisão dos dados
from sklearn.naive_bayes import GaussianNB # modelo de Naive Bayes
from sklearn.preprocessing import LabelEncoder # codificação de rótulos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report# avaliação do modelo

#Naive Bayes é um algoritmo de classificação baseado no teorema de Bayes, 
# que assume que as características são independentes entre si. 
# Basicamente ele verifica a probabilidade de um caso acontecer levando em consideração o que já aconteceu
#P(A|B) = P(B|A) * P(A) / P(B) 
# P(A|B) é a probabilidade de A ocorrer dado que B ocorreu

data = pd.read_csv("C:/Projetos/GitHub/IA_Aulas/Basicos/B.Naive_Bayes/insurance.csv", keep_default_na=False) # lê o arquivo CSV e trata valores ausentes
#print(data.head())
data = data.drop(columns=['Unnamed: 0'])
#print(data.shape)
#print(data.isnull().sum())
y = data["Accident"].values
X = data.drop(columns=["Accident"]).values
#print(X)
#print(y)

labelencoder = LabelEncoder() # codificação de rótulos para variáveis categóricas
for i in range(X.shape[1]):#para cada coluna de x
    if X[:, i].dtype == 'object': #verifica se a coluna é do tipo objeto (categórica)
        X[:, i] = labelencoder.fit_transform(X[:, i]) #se sim, codifica a coluna usando LabelEncoder deixando numerico
#print(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
modelo = GaussianNB() # cria modelo Naive Bayes
modelo.fit(x_train, y_train) # treina o modelo com os dados de treinamento
y_pred = modelo.predict(x_test) # faz previsões com os dados de teste
print("Acurácia:", accuracy_score(y_test, y_pred)) # avalia a acurácia do modelo
print("Precisão:", precision_score(y_test, y_pred, average='weighted')) # avalia a precisão do modelo
print("Recall:", recall_score(y_test, y_pred, average='weighted')) # avalia o recall do modelo 
print("F1-Score:", f1_score(y_test, y_pred, average='weighted')) # avalia o F1-Score do modelo 
print("Relatório de Classificação:\n", classification_report(y_test, y_pred)) # exibe o relatório de classificação detalhado