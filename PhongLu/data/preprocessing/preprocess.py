"""Auditable preprocessing for the Deng et al. CDR3β benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED = ("cdr3_beta", "peptide")


def _normalise_sequence(value: object) -> str:
    return str(value).strip().upper() if pd.notna(value) else ""


def _is_valid(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(VALID_AA)


def _is_a02(value: object) -> bool:
    """Accept common HLA-A*02 spellings; retain subtypes such as A*02:01."""
    return "A*02" in str(value).upper().replace("HLA-", "")


def preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    for column in ("cdr3_beta", "peptide"):
        data[column] = data[column].map(_normalise_sequence)
    audit: list[dict[str, int | str]] = []

    def keep(mask: pd.Series, reason: str) -> None:
        nonlocal data
        before = len(data)
        data = data.loc[mask].copy()
        audit.append({"stage": reason, "rows_before": before,
                      "rows_removed": before - len(data), "rows_remaining": len(data)})

    keep(data.cdr3_beta.map(_is_valid) & data.peptide.map(_is_valid), "invalid_or_missing_amino_acid")
    keep(data.peptide.str.len().eq(9), "peptide_length_not_9")
    keep(data.cdr3_beta.str.len().between(10, 18), "cdr3_beta_length_not_10_to_18")
    if "mhc" in data.columns and data.mhc.notna().any() and data.mhc.astype(str).str.strip().ne("").any():
        keep(data.mhc.map(_is_a02), "mhc_not_hla_a02")
    else:
        audit.append({"stage": "mhc_filter_not_applied_missing_mhc", "rows_before": len(data),
                      "rows_removed": 0, "rows_remaining": len(data)})
    keep(~data.duplicated(["cdr3_beta", "peptide"], keep="first"), "duplicate_cdr3_beta_peptide")
    return data.reset_index(drop=True), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/dbase_positive.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/preprocessing_audit.json"))
    args = parser.parse_args()
    processed, audit = preprocess(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {len(processed)} positive rows to {args.output}")


if __name__ == "__main__":
    main()

