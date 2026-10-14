import pandas as pd
import numpy as np
import sys
import joblib
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score, average_precision_score
from collections import defaultdict
from sklearn.metrics import ndcg_score, average_precision_score

def cargar_letor(path):
    X = []
    y = []
    qids = []

    with open(path, "r") as f:
        for linea in f:
            partes = linea.strip().split()
            y.append(int(partes[0]))
            qids.append(int(partes[1].split(":")[1]))

            features = np.zeros(136)
            for item in partes[2:]:
                idx, val = item.split(":")
                features[int(idx) - 1] = float(val)

            X.append(features)

    return np.array(X), np.array(y), np.array(qids)

def evaluar_ranking(y_true, y_scores, qids, k_values=[5,10]):

    by_query = defaultdict(list)
    for idx, qid in enumerate(qids):
        by_query[qid].append(idx)

    resultados = {f"nDCG@{k}": [] for k in k_values}
    resultados.update({f"MAP@{k}": [] for k in k_values})

    for qid, indices in by_query.items():
        if len(indices) < 2:
            continue 

        y_true_q = y_true[indices]
        y_scores_q = y_scores[indices]

        for k in k_values:
            ndcg = ndcg_score([y_true_q], [y_scores_q], k=k)
            resultados[f"nDCG@{k}"].append(ndcg)

        y_true_bin = (y_true_q > 0).astype(int)
        sorted_indices = np.argsort(-y_scores_q)
        y_true_sorted = y_true_bin[sorted_indices]
        for k in k_values:
            y_true_at_k = y_true_sorted[:k]
            if y_true_at_k.sum() == 0:
                ap = 0
            else:
                ap = average_precision_score(y_true_at_k, np.arange(len(y_true_at_k),0,-1))
            resultados[f"MAP@{k}"].append(ap)

    resultados_globales = {metric: np.mean(vals) if len(vals) > 0 else float('nan') for metric, vals in resultados.items()}
    return resultados_globales

def main():
    if len(sys.argv) < 2:
        print("Use la ruta del archivo")
        sys.exit(1)

    val_path = sys.argv[1]
    X_val, y_val, qids_val = cargar_letor(val_path)
    print(f"Queries totales: {len(np.unique(qids_val))}, filas: {len(X_val)}")

    pointwise_model = joblib.load("pointwise_model.joblib")
    pairwise_model = joblib.load("pairwise_model.joblib")
    listwise_model = joblib.load("listwise_model.joblib")

    y_scores_pw = pointwise_model.predict(X_val)
    metrics_pw = evaluar_ranking(y_val, y_scores_pw, qids_val)
    for metric, value in metrics_pw.items():
        print(f"{metric}: {value:.4f}")

    scaler = pairwise_model.named_steps["scaler"]
    X_val_scaled = scaler.transform(X_val)
    y_scores_pair = pairwise_model.named_steps["svm"].decision_function(X_val_scaled)
    metrics_pair = evaluar_ranking(y_val, y_scores_pair, qids_val)
    for metric, value in metrics_pair.items():
        print(f"{metric}: {value:.4f}")

    y_scores_lw = listwise_model.predict(X_val)
    metrics_lw = evaluar_ranking(y_val, y_scores_lw, qids_val)
    for metric, value in metrics_lw.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
