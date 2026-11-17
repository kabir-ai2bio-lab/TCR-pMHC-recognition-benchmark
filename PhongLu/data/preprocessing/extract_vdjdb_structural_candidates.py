"""Create an auditable paired-TCR candidate set for a STAG-style pilot.

This deliberately writes a *new* table.  It does not alter the CDR3beta-only
benchmark, its fixed split files, or any previous model results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from PhongLu.data.preprocessing.preprocess import VALID_AA, _is_a02


OUTPUT_COLUMNS = [
    "source_file", "source_chunk_id", "source_reference", "species", "mhc",
    "mhc_b", "mhc_class", "peptide", "cdr3_alpha", "v_alpha", "j_alpha",
    "cdr3_beta", "v_beta", "d_beta", "j_beta", "structure_id", "label",
]


def normalise(value: object) -> str:
    return str(value).strip().upper() if pd.notna(value) else ""


def valid_amino_acids(series: pd.Series) -> pd.Series:
    return series.map(lambda sequence: bool(sequence) and set(sequence).issubset(VALID_AA))


def stage(data: pd.DataFrame, mask: pd.Series, name: str) -> tuple[pd.DataFrame, dict[str, int | str]]:
    before = len(data)
    kept = data.loc[mask].copy()
    return kept, {"stage": name, "rows_before": before, "rows_removed": before - len(kept), "rows_remaining": len(kept)}


def read_chunks(chunks_dir: Path) -> pd.DataFrame:
    """Read a submission-chunk directory or one official vdjdb_full.txt file."""
    frames = []
    fields = {
        "chunk.id": "source_chunk_id", "reference.id": "source_reference",
        "species": "species", "mhc.a": "mhc", "mhc.b": "mhc_b",
        "mhc.class": "mhc_class", "antigen.epitope": "peptide",
        "cdr3.alpha": "cdr3_alpha", "v.alpha": "v_alpha", "j.alpha": "j_alpha",
        "cdr3.beta": "cdr3_beta", "v.beta": "v_beta", "d.beta": "d_beta",
        "j.beta": "j_beta", "meta.structure.id": "structure_id",
    }
    input_files = [chunks_dir] if chunks_dir.is_file() else sorted(chunks_dir.glob("*.txt"))
    for path in input_files:
        raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
        required = {"species", "mhc.a", "mhc.class", "antigen.epitope", "cdr3.alpha", "cdr3.beta"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"{path} lacks required VDJdb columns: {sorted(missing)}")
        frame = pd.DataFrame(index=raw.index)
        for source, destination in fields.items():
            frame[destination] = raw[source] if source in raw else ""
        if "chunk.id" not in raw:
            frame["source_chunk_id"] = [str(row + 2) for row in range(len(raw))]
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No VDJdb .txt chunks found in {chunks_dir}")
    result = pd.concat(frames, ignore_index=True)
    for column in result.columns:
        result[column] = result[column].map(normalise)
    return result


def build_structural_candidates(chunks_dir: Path) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    """Return human, class-I, A*02, 9-mer records with reconstructable TCRs."""
    data = read_chunks(chunks_dir)
    audit: list[dict[str, int | str]] = []
    data, item = stage(data, data.species.str.lower().eq("homosapiens"), "not_homo_sapiens")
    audit.append(item)
    data, item = stage(data, data.mhc_class.eq("MHCI"), "mhc_class_not_i")
    audit.append(item)
    data, item = stage(data, data.mhc.map(_is_a02), "mhc_not_hla_a02")
    audit.append(item)
    data, item = stage(data, data.peptide.str.len().eq(9), "peptide_length_not_9")
    audit.append(item)
    sequence_columns = ["cdr3_alpha", "cdr3_beta", "peptide"]
    sequence_mask = pd.concat([valid_amino_acids(data[column]) for column in sequence_columns], axis=1).all(axis=1)
    data, item = stage(data, sequence_mask, "missing_or_invalid_cdr3_alpha_beta_or_peptide")
    audit.append(item)
    gene_columns = ["v_alpha", "j_alpha", "v_beta", "j_beta"]
    complete_genes = data[gene_columns].ne("").all(axis=1)
    data, item = stage(data, complete_genes, "missing_alpha_or_beta_vj_gene")
    audit.append(item)
    data, item = stage(
        data,
        ~data.duplicated(["cdr3_alpha", "cdr3_beta", "peptide", "mhc"], keep="first"),
        "duplicate_alpha_beta_peptide_mhc",
    )
    audit.append(item)
    data["label"] = 1
    return data[OUTPUT_COLUMNS].reset_index(drop=True), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--chunks-dir", type=Path, help="Legacy submission chunk folder")
    sources.add_argument("--full-file", type=Path, help="Official vdjdb_full.txt release file")
    parser.add_argument("--output", type=Path, default=Path("data/interim/vdjdb_paired_structural_candidates.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/interim/vdjdb_paired_structural_audit.json"))
    args = parser.parse_args()
    candidates, audit = build_structural_candidates(args.full_file or args.chunks_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    report = {
        "purpose": "STAG-style structural pilot candidates; not a replacement for the existing CDR3beta-only benchmark.",
        "filters": audit,
        "final_rows": len(candidates),
        "unique_cdr3_alpha_beta_pairs": int(candidates[["cdr3_alpha", "cdr3_beta"]].drop_duplicates().shape[0]),
        "unique_peptides": int(candidates.peptide.nunique()),
        "unique_mhc_a": int(candidates.mhc.nunique()),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(candidates):,} structural pilot candidates to {args.output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
