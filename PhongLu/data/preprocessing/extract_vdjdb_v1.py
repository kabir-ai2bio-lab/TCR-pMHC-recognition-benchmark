"""Produce the VDJdb-only positive candidates required for project Version 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from PhongLu.data.preprocessing.import_sources import import_vdjdb, import_vdjdb_release
from PhongLu.data.preprocessing.preprocess import preprocess


def _stage(frame: pd.DataFrame, mask: pd.Series, name: str) -> tuple[pd.DataFrame, dict[str, int | str]]:
    before = len(frame)
    filtered = frame.loc[mask].copy()
    return filtered, {"stage": name, "rows_before": before, "rows_removed": before - len(filtered), "rows_remaining": len(filtered)}


def build_vdjdb_v1(input_path: Path) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    raw = import_vdjdb(input_path) if input_path.is_dir() else import_vdjdb_release(input_path)
    human, human_audit = _stage(raw, raw.species.str.lower().eq("homosapiens"), "not_homo_sapiens")
    class_i, class_audit = _stage(human, human.mhc_class.eq("MHCI"), "mhc_class_not_i")
    processed, preprocess_audit = preprocess(class_i)
    return processed, [human_audit, class_audit, *preprocess_audit]


def main() -> None:
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--full-file", type=Path, help="Official release vdjdb_full.txt")
    sources.add_argument("--chunks-dir", type=Path, help="Legacy raw submission chunks")
    parser.add_argument("--output", type=Path, default=Path("data/processed/vdjdb_v1_positive.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/vdjdb_v1_audit.json"))
    args = parser.parse_args()
    data, audit = build_vdjdb_v1(args.full_file or args.chunks_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {len(data):,} VDJdb Version-1 candidate positives to {args.output}")


if __name__ == "__main__":
    main()
