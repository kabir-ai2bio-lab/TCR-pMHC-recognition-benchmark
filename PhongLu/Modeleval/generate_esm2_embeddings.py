"""Generate frozen protein-language-model embeddings for CDR3beta and peptides.

The pretrained ESM-2 model is never updated. It sees no binding labels, so using it
to encode all sequences before outer-fold evaluation does not leak supervised labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed_sequences(model, alphabet, sequences: list[str], batch_size: int, device: torch.device) -> np.ndarray:
    """Mean-pool residue embeddings, excluding ESM beginning/end special tokens."""
    converter = alphabet.get_batch_converter()
    layer = model.num_layers
    parts: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        _, raw, tokens = converter([(str(i), sequence) for i, sequence in enumerate(batch)])
        output = model(tokens.to(device), repr_layers=[layer], return_contacts=False)
        token_embeddings = output["representations"][layer]
        for position, sequence in enumerate(raw):
            # positions 0 and len(sequence)+1 are ESM's BOS/EOS tokens.
            parts.append(token_embeddings[position, 1 : len(sequence) + 1].mean(dim=0).cpu().numpy())
        print(f"Embedded {min(start + len(batch), len(sequences)):,}/{len(sequences):,} sequences")
    return np.stack(parts).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, nargs="+", required=True, help="One or more prepared split CSV files")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    parser.add_argument(
        "--model",
        default="esm2_t6_8M_UR50D",
        choices=("esm2_t6_8M_UR50D", "esm1v_t33_650M_UR90S_1"),
        help="Frozen ESM model. ESM-1v is the encoder used by TCR-ESM.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args()

    try:
        import esm
    except ImportError as error:
        raise SystemExit("Missing dependency. In Colab run: pip install fair-esm") from error

    data = pd.concat([pd.read_csv(path) for path in args.splits], ignore_index=True)
    required = {"cdr3_beta", "peptide"}
    if not required.issubset(data.columns):
        raise ValueError(f"Each split needs columns {sorted(required)}")
    tcrs = sorted(data.cdr3_beta.astype(str).unique())
    peptides = sorted(data.peptide.astype(str).unique())
    device = choose_device(args.device)
    loaders = {
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        # TCR-ESM (Yadav et al., 2023) uses this frozen ESM-1v checkpoint.
        "esm1v_t33_650M_UR90S_1": esm.pretrained.esm1v_t33_650M_UR90S_1,
    }
    model, alphabet = loaders[args.model]()
    model.eval().to(device)
    print(f"Device: {device}; unique CDR3beta: {len(tcrs):,}; unique peptides: {len(peptides):,}")
    tcr_embeddings = embed_sequences(model, alphabet, tcrs, args.batch_size, device)
    peptide_embeddings = embed_sequences(model, alphabet, peptides, args.batch_size, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"model": args.model, "pooling": "mean residue embeddings; no BOS/EOS", "tcr_count": len(tcrs), "peptide_count": len(peptides), "dimension": int(tcr_embeddings.shape[1])}
    np.savez_compressed(args.output, tcr_sequences=np.asarray(tcrs), tcr_embeddings=tcr_embeddings, peptide_sequences=np.asarray(peptides), peptide_embeddings=peptide_embeddings, metadata=json.dumps(metadata))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
