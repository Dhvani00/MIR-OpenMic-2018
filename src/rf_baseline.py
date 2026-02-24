import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score

import joblib


# --------------------------------------------------
# Feature loading
# --------------------------------------------------
def load_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    prefer = ['arr_0', 'X', 'features', 'vggish', 'embeddings']
    for k in prefer:
        if k in keys:
            X = data[k]
            break
    else:
        raise ValueError("No valid feature array found in npz file.")

    if X.ndim == 3:
        X = X.mean(axis=1)

    meta = {}
    for k in ['clip_id', 'clip_ids', 'sample_key', 'filenames']:
        if k in keys:
            meta['clip_id'] = np.asarray(data[k]).astype(str)
            break

    return X, meta


# --------------------------------------------------
# Label processing
# --------------------------------------------------
def load_and_binarize_labels(csv_path, threshold):
    df = pd.read_csv(csv_path)
    df['instrument'] = df['instrument'].astype(str).str.strip()

    pivot = df.pivot_table(
        index='sample_key',
        columns='instrument',
        values='relevance',
        aggfunc='max'
    ).fillna(0.0)

    labels = (pivot >= threshold).astype(int)
    return labels


# --------------------------------------------------
# Alignment
# --------------------------------------------------
def align_features_and_labels(X, labels, meta):
    if 'clip_id' not in meta:
        if X.shape[0] != labels.shape[0]:
            raise RuntimeError("Feature/label size mismatch.")
        return X, labels.values, labels.columns.tolist()

    label_index = labels.index.astype(str)
    index_map = {k: i for i, k in enumerate(label_index)}

    keep = [index_map[c] for c in meta['clip_id'] if c in index_map]
    X = X[:len(keep)]
    y = labels.values[keep]

    return X, y, labels.columns.tolist()


# --------------------------------------------------
# AUC helper
# --------------------------------------------------
def macro_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return float(np.mean(aucs))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--relevance-threshold', type=float, default=0.5)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--out', default='rf_weighted.joblib')
    args = parser.parse_args()

  # --------------------------------------------------
  # train, test data split
  # --------------------------------------------------
    X, meta = load_features(args.features)
    labels = load_and_binarize_labels(args.labels, args.relevance_threshold)

    X, y, label_names = align_features_and_labels(X, labels, meta)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

  # --------------------------------------------------
  #  simple RF. classifier trainig
  # --------------------------------------------------
    model = OneVsRestClassifier(
       RandomForestClassifier(
           n_estimators=500,
           n_jobs=-1,
           random_state=args.random_state
       )
    )

    model.fit(X_train, y_train)

  # --------------------------------------------------
  #  Evaluation metrics
  # --------------------------------------------------

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc = macro_auc(y_test, y_score)

    print("\n=== BASELINE RESULTS ===")
    print(f"Micro F1 : {micro_f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Macro AUC: {auc:.4f}")

    joblib.dump(model, args.out)
    pd.DataFrame(y_score, columns=label_names).to_csv("predicted_scores.csv", index=False)


if __name__ == "__main__":
    main()

