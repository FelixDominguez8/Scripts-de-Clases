import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Inciso 1
def preprocesamiento(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

# Inciso 2
def division(df):
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=98, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Inciso 3 y 5
def PrimerEnfoque(X_train, y_train, X_test, y_test):

    pipeline = Pipeline([('kpca', KernelPCA()), ('logreg', LogisticRegression(max_iter=1000))])

    param_grid = {'kpca__n_components': [2, 3, 5, 10],
        'kpca__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
        'logreg__C': [0.1, 1, 10, 100],
        'logreg__solver': ['lbfgs', 'liblinear']}

    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid.fit(X_train, y_train)

    print("Hiperprametros:")
    print(grid.best_params_)

    y_pred = grid.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicho')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusion')
    plt.show()

    print("Reporte")
    print(classification_report(y_test, y_pred))

    return grid.best_estimator_

# Inciso 4 y 5
def SegundoEnfoque(X_train, y_train, X_test, y_test):
    gradboost = GradientBoostingClassifier(
        n_estimators=100,
        min_samples_split=5,
        random_state=42
    )

    gradboost.fit(X_train, y_train)

    y_pred = gradboost.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.xlabel('Predicho')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusion')
    plt.show()

    print("Reporte:")
    print(classification_report(y_test, y_pred))

    return gradboost

def main():
    if len(sys.argv) < 2:
        print("Ponga la ruta del archivo")
        sys.exit(1)

    ruta_csv = sys.argv[1]

    df = pd.read_csv(ruta_csv)

    X_train, X_test, y_train, y_test = division(df)
    X_train, X_test = preprocesamiento(X_train, X_test)
    PrimerEnfoque(X_train, y_train, X_test, y_test)
    SegundoEnfoque(X_train, y_train, X_test, y_test)

    # Inciso 6
    print("\nInciso 6:\n")
    print("Entre las tecnicas evaluadas podemos ver que el segundo enfoque Gradient Boosting tira mejores "
          "resultados obteniendo 0.96 en casi todas las metricas en comparacion al casi 0.85 de el primer "
          "enfoque\n")
    print("Para esto tambien siento que se podria aplicar random forest ya que es tambien un modelo "
          "supervisado el cual puede generar bastantes buenos resultados")

if __name__ == "__main__":
    main()