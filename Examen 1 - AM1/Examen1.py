#Importar librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#Leer dataset
df = pd.read_csv('./penguins_size.csv')

#Mostrar las primeras filas del dataset
print(df.head())

#Imprimir descripcion estadistica de las variables numericas
df.describe()

#Graficar distribucion de especies


#Eliminar nulos
df = df.replace(['.', 'NA', 'NaN', 'nan', ' '], np.nan)

for col in df.columns:
    df = df.dropna(subset=[col])

#Codificacion de variables
categoricas = df.select_dtypes(include=['object', 'category']).columns
df['sex'] = df['sex'].map({'MALE': 1, 'FEMALE': 0})
df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df['island'] = df['island'].map({'Torgersen': 0, 'Dream': 1, 'Biscoe': 2})
print(df.columns)

#Entrenamiento
y = df['species']
X = df.drop(columns=['species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modelo = LogisticRegression(random_state=16, max_iter=1000)

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

matriz_confusion = metrics.confusion_matrix(y_test, y_pred)
matriz_confusion

nombres_clases=[0,1]
fig, ax = plt.subplots()
marcas_verificacion = np.arange(len(nombres_clases))
plt.xticks(marcas_verificacion, nombres_clases)
plt.yticks(marcas_verificacion, nombres_clases)

sns.heatmap(pd.DataFrame(matriz_confusion), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz de Confusion',y=1.1)
plt.ylabel('Actual')
plt.xlabel('Prediccion')
plt.show()

y_pred = y_pred.astype(int)
Accuracy =metrics.accuracy_score(y_test, y_pred)
Precision = metrics.precision_score(y_test, y_pred)
Sensitivity_recall = metrics.recall_score(y_test, y_pred)
Specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
F1_score = metrics.f1_score(y_test, y_pred)
print({"Exactitud":Accuracy,"Precision":Precision,
      "Sensibilidad_recall":Sensitivity_recall,
      "Especificidad":Specificity,"F1_score":F1_score})

