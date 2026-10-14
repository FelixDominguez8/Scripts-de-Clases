import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


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
