from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import joblib


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
