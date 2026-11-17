"""Check saved test predictions and summary metrics against an input split file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.splits)
    summary = json.loads((args.results / "summary.json").read_text())
    rows = []
    cols = ["cdr3_beta", "peptide", "binder"]
    for fold in range(5):
        prediction = pd.read_csv(args.results / f"fold_{fold}_predictions.csv")
        expected = data.loc[data.fold.eq(fold), cols].reset_index(drop=True)
        actual = prediction[cols].reset_index(drop=True)
        if not expected.equals(actual):
            raise AssertionError(f"Fold {fold}: saved prediction rows do not match the split")
        if not np.isfinite(prediction.score).all() or not prediction.score.between(0, 1).all():
            raise AssertionError(f"Fold {fold}: invalid prediction scores")
        y, score = prediction.binder.to_numpy(), prediction.score.to_numpy()
        label = score >= 0.5
        rows.append({
            "auroc": roc_auc_score(y, score), "auprc": average_precision_score(y, score),
            "f1_at_0_5": f1_score(y, label),
            "precision_at_0_5": precision_score(y, label, zero_division=0),
            "recall_at_0_5": recall_score(y, label, zero_division=0),
        })
    recomputed = pd.DataFrame(rows).agg(["mean", "std"]).to_dict()
    for metric, aggregates in recomputed.items():
        for key, value in aggregates.items():
            if not np.isclose(value, summary[metric][key], rtol=1e-8, atol=1e-8):
                raise AssertionError(f"{metric} {key}: {value} != {summary[metric][key]}")
    print(f"Verified five folds, prediction labels/scores, and summary metrics: {args.results}")


if __name__ == "__main__":
    main()
