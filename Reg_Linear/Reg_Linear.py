import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

"""def correlacao(x,y):
    return np.cov(x,y) / (np.std(x) * np.std(y))

def inclinacao(x,y):
    return np.std(y) / np.std(x)

def interceptacao(x,y):
    return np.mean(y) - inclinacao(x,y) * np.mean(x)

def previsao(x,y, v):
    return interceptacao(x,y) + inclinacao(x,y) * v

x = [1,2,3,4,5]
y = [2,4,6,8,10]

print(previsao(x,y, 6))"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

base = pd.read_csv("C:/Users/proje/Downloads/CursoIA/CursoIA/3.Algoritmos de Machine Learning/mt_cars.csv")

#print(base.head())

base = base.drop(["Unnamed: 0"], axis=1)

#print(base.head())

corr = base.corr()
"""sns.heatmap(corr, annot=True)
plt.show()"""

modelo = sm.ols()