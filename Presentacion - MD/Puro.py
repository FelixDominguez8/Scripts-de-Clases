import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

ruta_archivo = "transacciones.csv"
df = pd.read_csv(ruta_archivo, nrows=100000)

def motor_de_reglas(row):
    if row['amount'] == row['oldbalanceOrg'] and row['amount'] > 1000:
        return 1 
    
    if row['amount'] > 500000 and row['type'] == 'TRANSFER':
        return 1
        
    if row['type'] == 'TRANSFER' and row['oldbalanceOrg'] > 0 and row['newbalanceOrig'] == 0:
        if row['amount'] > 100000:
            return 1 
        
    return 0

df['prediccion_reglas'] = df.apply(motor_de_reglas, axis=1)

matriz_sklearn = confusion_matrix(df['isFraud'], df['prediccion_reglas'])

plt.figure(figsize=(6, 5))
sns.heatmap(matriz_sklearn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimo (0)', 'Fraude (1)'], 
            yticklabels=['Legitimo (0)', 'Fraude (1)'])

plt.title('Matriz de Confusion', fontsize=14, pad=15)
plt.xlabel('Prediccion', fontsize=12)
plt.ylabel('Realidad', fontsize=12)
plt.tight_layout()
plt.show()