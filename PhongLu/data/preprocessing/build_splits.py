"""Create Deng-style uniform and strict five-fold VDJdb benchmark files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PhongLu.data.preprocessing.split import add_within_fold_negatives, assert_strict_no_leakage, make_folds


def downsample_for_strict(data: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Cap dominant peptides below one-third of an average fold (a conservative Deng safeguard)."""
    rng = np.random.default_rng(seed)
    current = data.copy()
    audit = []
    while current.peptide.value_counts().iloc[0] > len(current) / (3 * n_folds):
        cap = int(len(current) // (3 * n_folds))
        pieces = []
        for _, group in current.groupby("peptide", sort=False):
            if len(group) > cap:
                group = group.iloc[rng.permutation(len(group))[:cap]]
            pieces.append(group)
        next_data = pd.concat(pieces, ignore_index=True)
        audit.append({"rows_before": len(current), "rows_after": len(next_data), "per_peptide_cap": cap})
        current = next_data
    return current, audit


def _write_with_negatives(folds: pd.DataFrame, strategy: str, output_dir: Path, seed: int, all_positives: pd.DataFrame) -> None:
    parts = []
    known = set(map(tuple, all_positives[["cdr3_beta", "peptide"]].itertuples(index=False, name=None)))
    used = set()
    for fold in sorted(folds.fold.unique()):
        part = add_within_fold_negatives(folds.loc[folds.fold.eq(fold)], seed + int(fold), known, used)
        part["fold"] = fold
        parts.append(part)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(output_dir / f"{strategy}_five_fold.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/vdjdb_v1_positive.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    strict_input, strict_audit = downsample_for_strict(data, 5, args.seed)
    strict = make_folds(strict_input, "strict", seed=args.seed)
    assert_strict_no_leakage(strict)
    _write_with_negatives(strict, "strict", args.output_dir, args.seed, data)
    counts = data.peptide.value_counts()
    uniform_input = data.loc[data.peptide.isin(counts[counts >= 5].index)].copy()
    uniform = make_folds(uniform_input, "uniform", seed=args.seed)
    _write_with_negatives(uniform, "uniform", args.output_dir, args.seed, data)
    audit = {
        "seed": args.seed,
        "strict": {"positive_rows": len(strict_input), "downsampling": strict_audit},
        "uniform": {"positive_rows": len(uniform_input), "removed_peptides_with_fewer_than_5": int((counts < 5).sum())},
        "negative_strategy": "one fold-local shuffled negative per positive; exclude full positive pool and prior-fold negatives",
    }
    (args.output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Strict positives: {len(strict_input):,}; uniform positives: {len(uniform_input):,}")


if __name__ == "__main__":
    main()
