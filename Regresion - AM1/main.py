import argparse
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import eda
import train
import evaluate


def preprocess_data(df):
    eda.drop_low_corr(df)
    eda.drop_nulls(df)
    eda.fix_nulls(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
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
    parser = argparse.ArgumentParser(description="Housing Price Prediction Pipeline")
    parser.add_argument(
        "-e",
        "--eda",
        action="store_true",
        help="Ejecutar análisis exploratorio de datos",
    )
    parser.add_argument(
        "-t",
        "--train",
        help="Entrenar y guardar el modelo",
        nargs="?",
        const="model.pkl",
    )
    parser.add_argument(
        "-v", "--eval", help="Cargar y evaluar un modelo", nargs="?", const="model.pkl"
    )

    args = parser.parse_args()

    path = kagglehub.dataset_download("marcopale/housing")
    df = pd.read_csv(path + "/AmesHousing.csv")
    if not args.eda:
        X_train, X_test, y_train, y_test = preprocess_data(df)

    if args.eda:
        eda.generate(df)
    if args.train:
        train.fit(X_train, y_train, output_file=args.train)
    if args.eval:
        resultados = evaluate.eval(X_test, y_test, model_file=args.eval)

        print(f'\nMSE = {resultados["mse"]:.4f}')
        print(f'\nMAE = {resultados["mae"]:.4f}')
        print(f'\nRMSE = {resultados["rmse"]:.4f}')

        mean = y_test.mean()
        mae = 100 - ((resultados["mae"] / mean) * 100)

        print(f"Rendimiento promedio del modelo: {mae}%")

    if not (args.eda or args.train or args.eval):
        parser.print_help()


if __name__ == "__main__":
    main()
