import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("transacciones.csv")
df = df.sample(n=100000, random_state=42).reset_index(drop=True)

X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df['isFraud'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_ml = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_ml.fit(X_train, y_train)

probabilidades = modelo_ml.predict_proba(X_test)[:, 1] 

resultados = X_test.copy()
resultados['score_fraude'] = probabilidades
resultados['is_fraud_real'] = y_test
resultados['tipo_transaccion'] = df.loc[X_test.index, 'type']

def decision_negocio(row):
    if row['score_fraude'] > 0.80:
        if row['tipo_transaccion'] == 'DEBIT' and row['amount'] < 500:
            return "AUTENTICACION_SMS"
        return "RECHAZAR"
    elif row['score_fraude'] > 0.50 and row['amount'] < 100:
        return "AUTENTICACION_SMS"
    else:
        return "APROBAR"

resultados['accion_final'] = resultados.apply(decision_negocio, axis=1)
print(resultados[['amount', 'score_fraude', 'tipo_transaccion', 'accion_final']].head(20))

print("\nConteo de decisiones finales de negocio:")
print(resultados['accion_final'].value_counts())

print("\nFraudes reales presentes en el set de prueba:")
print(resultados['is_fraud_real'].value_counts())

filtro_sospechosos = resultados[resultados['score_fraude'] > 0.0]
print(filtro_sospechosos[['amount', 'score_fraude', 'tipo_transaccion', 'accion_final']].head(20))

y_pred_clase = modelo_ml.predict(X_test)
print(classification_report(y_test, y_pred_clase, target_names=['Legítimo (0)', 'Fraude (1)']))

print("\nGenerando gráfico de la Matriz de Confusión...")
matriz_num = confusion_matrix(y_test, y_pred_clase)

plt.figure(figsize=(6, 5))
sns.heatmap(matriz_num, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimo (0)', 'Fraude (1)'], 
            yticklabels=['Legitimo (0)', 'Fraude (1)'])

plt.title('Matriz de Confusión', fontsize=14, pad=15)
plt.xlabel('Prediccion', fontsize=12)
plt.ylabel('Realidad', fontsize=12)
plt.tight_layout()

plt.show()