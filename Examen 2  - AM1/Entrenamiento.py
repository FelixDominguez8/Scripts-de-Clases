#Librerias
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.svm import SVC

# 1. EDA
def EDA(df):
    # 1. Clases
    print("1. ¿Cuáles son las posibles clases del conjunto de datos?")
    print(df['quality'].unique())
    print("\n")

    # 2. Grafico de Barras
    print("2. ¿Las clases están equilibradas o no?")
    plt.figure(figsize=(6,4))
    sns.countplot(x=df['quality'])
    plt.title('Fraude o Legitimo (Vino)')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad')
    plt.show()

    print("No estan equilibradas, hay mucho mas casos de vino legitimo que de fraudulento\n")

    # 3. Comparacion de fraude
    print("3. ¿Que porcentaje de vinos tintos y blancos son fraude?")

    tinto_df = df[df['type'] == 'red']
    blanco_df = df[df['type'] == 'white']

    tinto_counts = tinto_df['quality'].value_counts().sort_index()
    blanco_counts = blanco_df['quality'].value_counts().sort_index()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.pie(tinto_counts, labels=tinto_counts.index, autopct='%1.1f%%')
    plt.title('Porcentajes de Calidad para Vino Tinto')

    plt.subplot(1, 2, 2)
    plt.pie(blanco_counts, labels=blanco_counts.index, autopct='%1.1f%%')
    plt.title('Porcentajes de Calidad para Vino Blanco')

    plt.tight_layout()
    plt.show()

    print("Porcentaje de fraude en vino tintos: " + str(((tinto_counts.get('Fraud', 0))/(len(tinto_df))) * 100) + " %")
    print("Porcentaje de fraude en vino blancos: " + str(((blanco_counts.get('Fraud', 0))/(len(blanco_df))) * 100) + " %")
    print("\n")

    # 4. Valores de Correlacion
    print("4. ¿Cual es la correlacion entre las variables continuas y la dependiente?")

    df['quality_target'] = df['quality'].map({'Fraud': 1, 'Legit': 0})

    variables_continuas = df.select_dtypes(include=['float64', 'int64']).columns
    variables_continuas = variables_continuas.drop('quality_target')

    correlaciones = {}

    for col in variables_continuas:
        correlaciones[col] = df[col].corr(df['quality_target'])

    print(pd.DataFrame.from_dict(correlaciones, orient='index', columns=['correlacion']))
    print("\n")

    # 5. Matriz de Correlacion
    print("5. ¿Cual es la matriz de  correlacion entre las variables continuas y la dependiente?")

    corr_df = df[variables_continuas.tolist() + ['quality_target']]
    matriz_corr = corr_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de correlación')
    plt.show()

# 3. Preprocesamiento de Datos
def preprocesamiento(df):
    df['quality'] = df['quality'].map({'Fraud': 1, 'Legit': 0})
    df['type'] = df['type'].map({'red': 1, 'white': 0})

    variables_continuas = df.select_dtypes(include=['float64', 'int64']).columns
    variables_continuas = variables_continuas.drop('quality')
    variables_continuas = variables_continuas.drop('type')

    scaler = StandardScaler()
    df[variables_continuas] = scaler.fit_transform(df[variables_continuas])

    return df

# 4. Division de los Datos
def division(df):
    y = df['quality']
    X = df.drop(columns=['quality'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101, stratify=y
    )

    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    return X_train, X_test, y_train, y_test

# 5. Entrenamiento del Modelo
def entrenar_svc(X_train, y_train):
    svc = SVC(class_weight='balanced')

    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_

    joblib.dump(mejor_modelo, 'modelo_svc.pkl')

def main():
    df = pd.read_csv('./wine_fraud.csv')
    EDA(df)
    df_nuevo = preprocesamiento(df)
    X_train, X_test, y_train, y_test = division(df_nuevo)
    entrenar_svc(X_train, y_train)

if __name__ == "__main__":
    main()

# ----------- Entrenamiento (Random Forest) -------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_random_forest(X_train, y_train):
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=101
    )
    modelo.fit(X_train, y_train)
    return modelo

# ----------- Evaluación -------------
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

# ----------- Entrenamiento (AdaBoost) -------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_adaboost(X_train, y_train):
    modelo = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=1.0,
        random_state=101
    )
    modelo.fit(X_train, y_train)
    return modelo

# ----------- Evaluación -------------
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

# ----------- Entrenamiento (Gradient Boosting) -------------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_gradient_boosting(X_train, y_train):
    modelo = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=101
    )
    modelo.fit(X_train, y_train)
    return modelo

# ----------- Evaluación -------------
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

def algo(X_train, y_train, X_test, y_test):
    modelo = entrenar_random_forest(X_train, y_train)
    modelo = entrenar_adaboost(X_train, y_train)
    modelo = entrenar_gradient_boosting(X_train, y_train)
    evaluar_modelo(modelo, X_test, y_test)
