import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from data import (
    load_features,
    load_and_binarize_labels,
    align_features_and_labels,
    macro_auc
)

from model import (
    build_baseline_model,
    build_weighted_model,
    build_oversampled_model
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--strategy',
                        choices=["baseline", "weighted", "oversample"],
                        default="baseline")
    parser.add_argument('--relevance-threshold', type=float, default=0.5)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--out', default='rf_model.joblib')

    args = parser.parse_args()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    X, meta = load_features(args.features)
    labels = load_and_binarize_labels(args.labels, args.relevance_threshold)
    X, y, label_names = align_features_and_labels(X, labels, meta)

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # --------------------------------------------------
    # Select model
    # --------------------------------------------------
    if args.strategy == "baseline":
        model = build_baseline_model(args.random_state)
    elif args.strategy == "weighted":
        model = build_weighted_model(args.random_state)
    else:
        model = build_oversampled_model(args.random_state)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc = macro_auc(y_test, y_score)

    print("\n=== RESULTS ===")
    print(f"Strategy : {args.strategy}")
    print(f"Micro F1 : {micro_f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Macro AUC: {auc:.4f}")

    joblib.dump(model, args.out)


if __name__ == "__main__":
    main()
