import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)


def eval(X_test, y_test, model_file):
    model = joblib.load(model_file)
    y_pred = model.predict(X_test)

    resultados = {}

    resultados["mse"] = mean_squared_error(y_test, y_pred)
    resultados["rmse"] = root_mean_squared_error(y_test, y_pred)
    resultados["mae"] = mean_absolute_error(y_test, y_pred)

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
