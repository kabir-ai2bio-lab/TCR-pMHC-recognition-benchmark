"""Convert raw IEDB and VDJdb downloads into a common, traceable schema.

These are source adapters, not a final biological-quality filter. In particular,
the output does not claim experimental-negative labels and is not model input.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

COMMON_COLUMNS = [
    "cdr3_beta", "peptide", "mhc", "species", "mhc_class", "source", "source_record_id",
    "source_reference", "source_evidence",
]


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def import_iedb(path: Path) -> pd.DataFrame:
    """Extract beta-chain receptor–epitope associations from an IEDB TCR export."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        top, detail = list(csv.reader(handle))[:2]
    columns = [f"{group} | {field}" for group, field in zip(top, detail)]
    raw = pd.read_csv(path, header=None, names=columns, skiprows=2, low_memory=False)
    shared = pd.DataFrame({
        "peptide": _text(raw["Epitope | Name"]),
        "mhc": _text(raw["Assay | MHC Allele Names"]),
        "source": "iedb",
        "source_record_id": _text(raw["Receptor | CEDAR Receptor ID"]),
        "source_reference": _text(raw["Reference | CEDAR IRI"]),
        "source_evidence": "curated_tcr_epitope_association_outcome_not_exported",
    })
    beta_frames = []
    for chain in ("Chain 1", "Chain 2"):
        chain_columns = {
            f"{chain} | Type", f"{chain} | CDR3 Curated", f"{chain} | CDR3 Calculated"
        }
        if not chain_columns.issubset(raw.columns):
            continue
        is_beta = _text(raw[f"{chain} | Type"]).str.lower().eq("beta")
        cdr3 = _text(raw[f"{chain} | CDR3 Calculated"])
        cdr3 = cdr3.mask(cdr3.eq(""), _text(raw[f"{chain} | CDR3 Curated"]))
        selected = shared.loc[is_beta].copy()
        selected["cdr3_beta"] = cdr3.loc[is_beta]
        organism = raw.get(f"{chain} | Organism IRI", pd.Series("", index=raw.index))
        selected["species"] = _text(organism.loc[is_beta])
        selected["mhc_class"] = ""
        beta_frames.append(selected)
    if not beta_frames:
        raise ValueError("IEDB export has no complete Chain 1 or Chain 2 CDR3 fields")
    return pd.concat(beta_frames, ignore_index=True)[COMMON_COLUMNS]


def import_vdjdb(chunks_dir: Path) -> pd.DataFrame:
    """Read VDJdb positive submission chunks without invoking its 64-GB build."""
    records = []
    for path in sorted(chunks_dir.glob("*.txt")):
        raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
        required = {"cdr3.beta", "antigen.epitope", "mhc.a", "reference.id"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"{path} lacks required VDJdb columns: {sorted(missing)}")
        output = pd.DataFrame({
            "cdr3_beta": _text(raw["cdr3.beta"]),
            "peptide": _text(raw["antigen.epitope"]),
            "mhc": _text(raw["mhc.a"]),
            "species": _text(raw["species"]),
            "mhc_class": _text(raw["mhc.class"]),
            "source": "vdjdb",
            "source_record_id": path.name + ":" + _text(raw.get("chunk.id", pd.Series(raw.index, index=raw.index))),
            "source_reference": _text(raw["reference.id"]),
            "source_evidence": "vdjdb_positive_submission_chunk",
        })
        records.append(output)
    if not records:
        raise FileNotFoundError(f"No VDJdb .txt chunks found in {chunks_dir}")
    return pd.concat(records, ignore_index=True)[COMMON_COLUMNS]


def import_vdjdb_release(path: Path) -> pd.DataFrame:
    """Read the official paired-row release, preserving its 1-based file row."""
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    required = {"cdr3.beta", "antigen.epitope", "mhc.a", "mhc.class", "species", "reference.id"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} lacks required VDJdb columns: {sorted(missing)}")
    return pd.DataFrame({
        "cdr3_beta": _text(raw["cdr3.beta"]),
        "peptide": _text(raw["antigen.epitope"]),
        "mhc": _text(raw["mhc.a"]),
        "species": _text(raw["species"]),
        "mhc_class": _text(raw["mhc.class"]),
        "source": "vdjdb_official_release",
        "source_record_id": [f"{path.name}:{row + 2}" for row in range(len(raw))],
        "source_reference": _text(raw["reference.id"]),
        "source_evidence": "vdjdb_full_release_record",
    })[COMMON_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iedb", type=Path, default=Path("data/raw/iedb/tcr_full_v3.csv"))
    parser.add_argument("--vdjdb-chunks", type=Path, default=Path("data/raw/vdjdb/vdjdb-db-master/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    iedb = import_iedb(args.iedb)
    vdjdb = import_vdjdb(args.vdjdb_chunks)
    iedb.to_csv(args.output_dir / "iedb_associations.csv", index=False)
    vdjdb.to_csv(args.output_dir / "vdjdb_associations.csv", index=False)
    combined = pd.concat([iedb, vdjdb], ignore_index=True)
    combined.to_csv(args.output_dir / "source_associations_unfiltered.csv", index=False)
    print(f"IEDB associations: {len(iedb):,}")
    print(f"VDJdb associations: {len(vdjdb):,}")
    print(f"Combined unfiltered associations: {len(combined):,}")


if __name__ == "__main__":
    main()
