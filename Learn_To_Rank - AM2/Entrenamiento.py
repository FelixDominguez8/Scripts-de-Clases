import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib
import sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import random

def cargar_letor(path):
    X = []
    y = []
    qids = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            y.append(int(parts[0]))
            qids.append(int(parts[1].split(":")[1]))

            features = np.zeros(136)
            for item in parts[2:]:
                idx, val = item.split(":")
                features[int(idx) - 1] = float(val)

            X.append(features)

    return np.array(X), np.array(y), np.array(qids)

def generar_pairwise(X, y, qids, max_pairs_total=20000):
    X_pairs = []
    y_pairs = []

    by_query = defaultdict(list)
    for i, qid in enumerate(qids):
        by_query[qid].append(i)

    total_pairs = 0

    for qid, indices in by_query.items():
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if y[indices[i]] == y[indices[j]]:
                    continue
                diff = X[indices[i]] - X[indices[j]]
                label = 1 if y[indices[i]] > y[indices[j]] else -1
                pairs.append((diff, label))

        if len(pairs) + total_pairs > max_pairs_total:
            remaining = max_pairs_total - total_pairs
            if remaining <= 0:
                break
            pairs = random.sample(pairs, remaining)

        for diff, label in pairs:
            X_pairs.append(diff)
            y_pairs.append(label)
        total_pairs += len(pairs)

        if total_pairs >= max_pairs_total:
            break

    return np.array(X_pairs), np.array(y_pairs)

def crear_grupos(qids):
    grupos = []
    unique_qids, counts = np.unique(qids, return_counts=True)
    grupos = counts.tolist()
    return grupos

def main():
    if len(sys.argv) < 2:
        print("Use la ruta del archivo")
        sys.exit(1)

    ruta = sys.argv[1]

    X, y, qids = cargar_letor(ruta)

    pointwise_model = Ridge()

    param_grid_pw = {
        "alpha": [0.1, 1.0, 10.0]
    }

    grid_pw = GridSearchCV(
        pointwise_model,
        param_grid_pw,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_pw.fit(X, y)
    joblib.dump(grid_pw.best_estimator_, "pointwise_model.joblib")

    X_pairs, y_pairs = generar_pairwise(X, y, qids, max_pairs_total=20000)

    pairwise_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(dual=False, max_iter=5000))
    ])

    param_grid_pwsvm = {
        "svm__C": [0.01, 0.1, 1.0]
    }

    grid_pairwise = GridSearchCV(
        pairwise_pipeline,
        param_grid_pwsvm,
        cv=3,
        n_jobs=-1
    )

    grid_pairwise.fit(X_pairs, y_pairs)
    joblib.dump(grid_pairwise.best_estimator_, "pairwise_model.joblib")

    order = np.argsort(qids)
    X_sorted = X[order]
    y_sorted = y[order]
    qids_sorted = qids[order]

    grupos = crear_grupos(qids_sorted)

    listwise_model = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,  
        verbosity=0
    )

    listwise_model.fit(
        X_sorted,
        y_sorted,
        group=grupos
    )

    joblib.dump(listwise_model, "listwise_model.joblib")


if __name__ == "__main__":
    main()