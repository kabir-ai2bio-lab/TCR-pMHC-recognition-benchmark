"""Make an out-of-fold classification report from five saved prediction files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = pd.concat([
        pd.read_csv(args.results / f"fold_{fold}_predictions.csv").assign(test_fold=fold)
        for fold in range(5)
    ], ignore_index=True)
    labels = (predictions.score >= 0.5).astype(int)
    report = classification_report(predictions.binder, labels, labels=[0, 1],
                                   target_names=["synthetic_negative", "reported_positive"],
                                   output_dict=True, zero_division=0)
    result = {
        "description": "Pooled out-of-fold classification report; one prediction per held-out row",
        "threshold": 0.5,
        "negative_label_caveat": "Synthetic negatives are not verified non-binders.",
        "n_rows": len(predictions),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(predictions.binder, labels, labels=[0, 1]).tolist(),
        "classification_report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(classification_report(predictions.binder, labels, labels=[0, 1],
                                target_names=["synthetic_negative", "reported_positive"], zero_division=0))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
