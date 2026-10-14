# housing_pipeline.py
# Unificación de todos los módulos en uno solo.

import argparse
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
import joblib


# ==========================================================
# ================ Funciones EDA (antes en eda.py) =========
# ==========================================================

def null_values(df):
    missing = df.isnull().sum()
    missing_filtered = missing[missing > 0].sort_values(ascending=False)
    missing_percentage = (missing_filtered.values / len(df) * 100).round(2)

    drop_labels = []
    for col, percentage in zip(missing_filtered.index, missing_percentage):
        if percentage > 60:
            drop_labels.append(col)

    create_hist(
        missing_percentage,
        missing_filtered.index,
        "nulls_hist.png",
        title="Porcentaje de valores nulos por variable",
    )


def categorical_values(df):
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    create_hist(
        [df[col].nunique() for col in categorical_cols],
        categorical_cols,
        "categorical_hist.png",
        title="Categorias por variable",
    )


def corr_matrix(df):
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

    df_dummies = pd.get_dummies(df, drop_first=True)
    corr = df_dummies.corr()["SalePrice"].sort_values(ascending=False)
    corr_filtered = corr[corr.index != "SalePrice"]

    cols_len = len(corr_filtered.index)

    create_hist(
        corr_filtered.values[:10],
        corr_filtered.index[:10],
        "best_corr_vars.png",
        "Correlación de las variables independientes",
    )
    create_hist(
        corr_filtered.values[cols_len - 11 : -1],
        corr_filtered.index[cols_len - 11 : -1],
        "worst_corr_vars.png",
        "Correlación de las variables independientes",
    )


def generate(df):
    null_values(df)
    categorical_values(df)
    corr_matrix(df)


def create_hist(x, y, output, title, xlabel=""):
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(x)), x)
    plt.yticks(range(len(x)), y, fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\nGráfico guardado como '{output}'")


def fix_nulls(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    imputer_num = SimpleImputer(strategy="mean")
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])


def drop_nulls(df):
    missing = df.isnull().sum()
    missing_filtered = missing[missing > 0].sort_values(ascending=False)
    missing_percentage = (missing_filtered.values / len(df) * 100).round(2)

    drop_labels = ["Order", "PID"]
    threshold = 60

    for col, percentage in zip(missing_filtered.index, missing_percentage):
        if percentage > threshold:
            drop_labels.append(col)

    if has_columns(df, drop_labels):
        df.drop(labels=drop_labels, axis="columns")
        return drop_labels
    else:
        return None


def drop_low_corr(df):
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

    df_dummies = pd.get_dummies(df, drop_first=True)
    corr = df_dummies.corr()["SalePrice"].sort_values(ascending=False)
    corr_filtered = corr[corr.index != "SalePrice"]

    drop_labels = ["Order", "PID"]
    threshold = 0.2
    for col, val in zip(corr_filtered.index, corr_filtered.values):
        if val < threshold:
            drop_labels.append(col)

    if has_columns(df, drop_labels):
        df.drop(labels=drop_labels, axis="columns")
        return drop_labels
    else:
        return None


def has_columns(df, labels):
    for col in labels:
        if not col in df.columns:
            return False
    return True


# ==========================================================
# ================ Entrenamiento (train.py) ================
# ==========================================================

def fit(X, y, output_file):
    param_grid = {
        "alpha": [0.1, 1, 5, 10, 50, 100],
        "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    }

    regr = ElasticNet()

    grid_search = GridSearchCV(
        estimator=regr,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    grid_search.fit(X, y)
    joblib.dump(grid_search.best_estimator_, output_file)

    print(f"Modelo guardado en {output_file}")


# ==========================================================
# ================ Evaluación (evaluate.py) ================
# ==========================================================

def eval(X_test, y_test, model_file):
    model = joblib.load(model_file)
    y_pred = model.predict(X_test)

    resultados = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    plot(y_test, y_pred)
    return resultados


def plot(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Predicción ideal",
    )

    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title("Valores Reales vs Predichos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("preds_vs_real.png", dpi=300, bbox_inches="tight")
    plt.show()


# ==========================================================
# ================ Script Principal (main.py) ==============
# ==========================================================

def preprocess_data(df):
    drop_low_corr(df)
    drop_nulls(df)
    fix_nulls(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


def main():
    print("🔹 Descargando dataset desde KaggleHub...")
    path = kagglehub.dataset_download("marcopale/housing")
    df = pd.read_csv(path + "/AmesHousing.csv")
    print("✅ Dataset cargado correctamente.")

    print("\n🔹 Ejecutando análisis exploratorio de datos (EDA)...")
    generate(df)

    print("\n🔹 Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n🔹 Entrenando modelo...")
    model_file = "model.pkl"
    fit(X_train, y_train, output_file=model_file)

    print("\n🔹 Evaluando modelo...")
    resultados = eval(X_test, y_test, model_file=model_file)

    print(f'\n📊 MSE = {resultados["mse"]:.4f}')
    print(f'📊 MAE = {resultados["mae"]:.4f}')
    print(f'📊 RMSE = {resultados["rmse"]:.4f}')

    mean = y_test.mean()
    mae = 100 - ((resultados["mae"] / mean) * 100)
    print(f"🏁 Rendimiento promedio del modelo: {mae:.2f}%")

    print("\n🎉 Proceso completo finalizado con éxito.")


if __name__ == "__main__":
    main()
