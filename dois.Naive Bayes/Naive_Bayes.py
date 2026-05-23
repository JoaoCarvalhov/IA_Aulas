import pandas as pd # leitura de dados
import numpy as np # vetores e matrizes
from sklearn.model_selection import train_test_split # divisão dos dados
from sklearn.naive_bayes import GaussianNB # modelo de Naive Bayes
from sklearn.preprocessing import LabelEncoder # codificação de rótulos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report# avaliação do modelo

data = pd.read_csv("C:/Users/proje/OneDrive/Documentos/GitHubUEA/IA_Aulas/dois.Naive Bayes/insurance.csv", keep_default_na=False) # lê o arquivo CSV e trata valores ausentes
print(data.head())
data = data.drop(columns=['Unnamed: 0'])
#print(data.shape)
#print(data.isnull().sum())
y = data["Accident"]
X = data.drop(columns=["Accident"])
print(X.head())
print(y.head())