"""Leakage-safe peptide splits and within-fold artificial negatives."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def assert_strict_no_leakage(data: pd.DataFrame) -> None:
    for fold in sorted(data.fold.unique()):
        held_out = set(data.loc[data.fold.eq(fold), "peptide"])
        training = set(data.loc[~data.fold.eq(fold), "peptide"])
        if not held_out.isdisjoint(training):
            raise AssertionError(f"Peptide leakage in strict fold {fold}")
    if data.duplicated(["cdr3_beta", "peptide"]).any():
        raise AssertionError("Duplicate CDR3β-peptide pairs exist across split rows")


def make_folds(positives: pd.DataFrame, strategy: str, n_folds: int = 5, seed: int = 13) -> pd.DataFrame:
    if strategy not in {"strict", "uniform"}:
        raise ValueError("strategy must be strict or uniform")
    data = positives.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if strategy == "uniform":
        low_count = data.peptide.value_counts().loc[lambda counts: counts < n_folds]
        if not low_count.empty:
            raise ValueError(
                "Uniform five-fold splitting requires at least five observations per peptide; "
                f"{len(low_count)} peptides do not meet this criterion. Filter and audit them first."
            )
        # Each peptide is shuffled independently then spread across folds.
        data["fold"] = -1
        for _, indexes in data.groupby("peptide", sort=False).groups.items():
            indexes = np.asarray(list(indexes))
            data.loc[rng.permutation(indexes), "fold"] = np.arange(len(indexes)) % n_folds
    else:
        # Greedy bin packing: a peptide is indivisible and folds have comparable row counts.
        counts = data.peptide.value_counts().sort_values(ascending=False)
        fold_sizes = np.zeros(n_folds, dtype=int)
        assignment: dict[str, int] = {}
        for peptide, count in counts.items():
            candidates = np.flatnonzero(fold_sizes == fold_sizes.min())
            fold = int(rng.choice(candidates))
            assignment[peptide] = fold
            fold_sizes[fold] += int(count)
        data["fold"] = data.peptide.map(assignment).astype(int)
        assert_strict_no_leakage(data)
    return data


def add_within_fold_negatives(fold_data: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create one non-cognate rearranged pair per positive without leaving its fold."""
    rng = np.random.default_rng(seed)
    positives = fold_data[["cdr3_beta", "peptide"]].drop_duplicates().copy()
    known = set(map(tuple, positives.itertuples(index=False, name=None)))
    tcr_pool = positives.cdr3_beta.unique()
    negative_rows = []
    for peptide, group in positives.groupby("peptide", sort=False):
        candidates = np.array([tcr for tcr in tcr_pool if (tcr, peptide) not in known])
        if len(candidates) < len(group):
            raise RuntimeError(
                f"Cannot generate {len(group)} unique fold-local negatives for peptide {peptide}; "
                f"only {len(candidates)} non-cognate TCR candidates exist"
            )
        selected = rng.choice(candidates, size=len(group), replace=False)
        negative_rows.extend((str(tcr), peptide) for tcr in selected)
    negatives = pd.DataFrame(negative_rows, columns=["cdr3_beta", "peptide"])
    negatives["binder"] = 0
    positives["binder"] = 1
    return pd.concat([positives, negatives], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--strategy", choices=("strict", "uniform"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    positive = pd.read_csv(args.input)
    folds = make_folds(positive, args.strategy, seed=args.seed)
    for fold in range(5):
        part = add_within_fold_negatives(folds.loc[folds.fold.eq(fold)], args.seed + fold)
        part["fold"] = fold
        args.output_dir.mkdir(parents=True, exist_ok=True)
        part.to_csv(args.output_dir / f"{args.strategy}_fold_{fold}.csv", index=False)
    print(f"Wrote {args.strategy} folds to {args.output_dir}")


if __name__ == "__main__":
    main()
