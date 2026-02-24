import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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

