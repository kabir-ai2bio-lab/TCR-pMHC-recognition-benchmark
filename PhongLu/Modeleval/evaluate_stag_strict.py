"""Strict peptide-disjoint evaluation for the public STAG structural dataset.

This runner intentionally reuses the authors' STAG graph encoder and model, but
replaces their random-row cross-validation with a reproducible peptide-disjoint
train/validation/test evaluation. It is designed to run from the STAG repository
after this file is copied there.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def peptide_from_pdb(path: Path) -> str:
    """Read the antigen peptide (chain P) from an authors-supplied PDB."""
    structure = PDBParser(QUIET=True).get_structure("complex", path)
    try:
        chain = structure[0]["P"]
    except KeyError as error:
        raise ValueError(f"Missing peptide chain P in {path}") from error
    residues = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        code = protein_letters_3to1.get(residue.resname.upper())
        if code is None:
            raise ValueError(f"Unsupported peptide residue {residue.resname} in {path}")
        residues.append(code)
    peptide = "".join(residues)
    if not peptide:
        raise ValueError(f"Empty peptide chain P in {path}")
    return peptide


def prepare_metadata(table: Path, structures: Path, output: Path) -> pd.DataFrame:
    cache = output / "metadata_with_peptides.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="index")
    data = pd.read_csv(table, index_col=0)
    peptides = []
    for index in data.index:
        pdb = structures / f"{int(index)}.pdb"
        if not pdb.exists():
            raise FileNotFoundError(f"Missing structure for row {index}: {pdb}")
        peptides.append(peptide_from_pdb(pdb))
    data["peptide"] = peptides
    data.index.name = "index"
    data.to_csv(cache)
    return data


def make_folds(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign whole peptides to folds while balancing rows and class counts.

    StratifiedGroupKFold can be badly unbalanced when a few peptide groups are
    much larger than the rest. This deterministic greedy bin-packing procedure
    keeps every peptide intact, seeds all five folds with a large group, then
    assigns each remaining peptide to the fold closest to the global row and
    positive-label targets.
    """
    groups = (
        data.groupby("peptide", as_index=False)
        .agg(rows=("label", "size"), positives=("label", "sum"))
        .sort_values(["rows", "peptide"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(groups) < 5:
        raise ValueError("Strict five-fold evaluation requires at least five peptides")
    rng = np.random.default_rng(seed)
    # Randomness only resolves otherwise equal assignments; it does not inspect
    # validation/test labels outside this public dataset's grouping operation.
    groups["tie_break"] = rng.random(len(groups))
    groups = groups.sort_values(["rows", "tie_break"], ascending=[False, True]).reset_index(drop=True)
    target_rows = len(data) / 5
    target_positive = float(data.label.sum()) / 5
    fold_rows = np.zeros(5, dtype=float)
    fold_positives = np.zeros(5, dtype=float)
    peptide_to_fold: dict[str, int] = {}
    for group_number, row in groups.iterrows():
        candidates = range(5) if group_number >= 5 else [group_number]
        costs = []
        for fold in candidates:
            # Use the *change* in the overall imbalance objective.  Comparing
            # only the post-assignment distance to the target can repeatedly
            # place groups into a nearly-full fold while leaving a small fold
            # empty.  The marginal change instead rewards filling the folds
            # that are currently furthest below their targets.
            old_row_cost = ((fold_rows[fold] - target_rows) / target_rows) ** 2
            new_row_cost = ((fold_rows[fold] + row.rows - target_rows) / target_rows) ** 2
            old_positive_cost = ((fold_positives[fold] - target_positive) / max(target_positive, 1.0)) ** 2
            new_positive_cost = ((fold_positives[fold] + row.positives - target_positive) / max(target_positive, 1.0)) ** 2
            costs.append(((new_row_cost - old_row_cost) + 0.35 * (new_positive_cost - old_positive_cost), fold))
        _, chosen_fold = min(costs)
        peptide_to_fold[str(row.peptide)] = int(chosen_fold)
        fold_rows[chosen_fold] += row.rows
        fold_positives[chosen_fold] += row.positives
    folds = data.peptide.astype(str).map(peptide_to_fold).to_numpy(dtype=int)
    if np.any(folds < 0):
        raise RuntimeError("Every row must receive a fold")
    result = data.copy()
    result["fold"] = folds
    for fold in range(5):
        test_peptides = set(result.loc[result.fold.eq(fold), "peptide"])
        train_peptides = set(result.loc[result.fold.ne(fold), "peptide"])
        if test_peptides & train_peptides:
            raise AssertionError(f"Peptide leakage in fold {fold}")
    return result


def import_stag(repo: Path):
    sys.path.insert(0, str(repo))
    from models.STAG.encode_structure import add_edge_data, get_graph
    from models.STAG.model import STAG
    return get_graph, add_edge_data, STAG


def load_graphs(data: pd.DataFrame, structures: Path, cache: Path, get_graph, add_edge_data) -> list[Data]:
    if cache.exists():
        print(f"Loading cached graphs: {cache}")
        return torch.load(cache, weights_only=False)
    graphs = []
    for n, (index, row) in enumerate(data.iterrows(), start=1):
        graph = get_graph(str(structures / f"{int(index)}.pdb"), int(row.label), 0)
        add_edge_data(graph)
        graphs.append(Data(x=graph.get_node_data(), edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=torch.tensor(float(row.label)), sample_index=int(index)))
        if n % 500 == 0:
            print(f"Encoded {n:,}/{len(data):,} structures")
    torch.save(graphs, cache)
    return graphs


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "f1_at_0_5": float(f1_score(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, scores = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        labels.extend(batch.y.cpu().numpy().astype(int).tolist())
        scores.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
    return np.asarray(labels), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--table", type=Path, default=Path("datasets/pan_peptide/pan_peptide.csv"))
    parser.add_argument("--structures", type=Path, default=Path("datasets/pan_peptide/structures"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--graph-cache", type=Path, default=None, help="Reusable encoded-graph cache; all seeds may share it.")
    parser.add_argument("--rebuild-graphs", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for STAG.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    set_seed(args.seed)
    get_graph, add_edge_data, STAG = import_stag(args.repo)
    table = args.table if args.table.is_absolute() else args.repo / args.table
    structures = args.structures if args.structures.is_absolute() else args.repo / args.structures
    data = prepare_metadata(table, structures, args.outdir)
    data = make_folds(data, args.seed)
    data.to_csv(args.outdir / "strict_split.csv")
    audit = {
        "seed": args.seed,
        "rows": int(len(data)),
        "unique_peptides": int(data.peptide.nunique()),
        "fold_rows": {str(f): int(data.fold.eq(f).sum()) for f in range(5)},
        "fold_peptides": {str(f): int(data.loc[data.fold.eq(f), "peptide"].nunique()) for f in range(5)},
        "strict_assertion": "Every test-fold peptide is absent from all other folds.",
    }
    (args.outdir / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    graph_cache = args.graph_cache or (args.repo / ".stag_cache" / "pan_peptide_graphs.pt")
    graph_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild_graphs and graph_cache.exists():
        graph_cache.unlink()
    graphs = load_graphs(data, structures, graph_cache, get_graph, add_edge_data)
    graph_by_index = {int(graph.sample_index): graph for graph in graphs}

    all_metrics = []
    all_predictions = []
    for test_fold in range(5):
        validation_fold = (test_fold + 1) % 5
        train = data[~data.fold.isin([test_fold, validation_fold])]
        valid = data[data.fold.eq(validation_fold)]
        test = data[data.fold.eq(test_fold)]
        train_loader = DataLoader([graph_by_index[int(i)] for i in train.index], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader([graph_by_index[int(i)] for i in valid.index], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader([graph_by_index[int(i)] for i in test.index], batch_size=args.batch_size, shuffle=False)
        model = STAG().to(device)
        model.apply(lambda layer: nn.init.xavier_uniform_(layer.weight) if isinstance(layer, nn.Linear) else None)
        positive_weight = (len(train) - train.label.sum()) / train.label.sum()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight), device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        best_state, best_auc, stale = None, -np.inf, 0
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            valid_y, valid_scores = predict(model, valid_loader, device)
            valid_auc = roc_auc_score(valid_y, valid_scores)
            if valid_auc > best_auc:
                best_auc, stale = valid_auc, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        y, scores = predict(model, test_loader, device)
        fold_metrics = {
            "fold": test_fold,
            **metrics(y, scores),
            "best_validation_auroc": float(best_auc),
            "epochs_run": epoch + 1,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
        }
        all_metrics.append(fold_metrics)
        prediction_frame = pd.DataFrame({"index": test.index, "peptide": test.peptide, "label": y, "score": scores, "fold": test_fold})
        prediction_frame.to_csv(args.outdir / f"fold_{test_fold}_predictions.csv", index=False)
        all_predictions.append(prediction_frame)
        print(fold_metrics)
        del model
        torch.cuda.empty_cache()

    result = pd.DataFrame(all_metrics)
    result.to_csv(args.outdir / "fold_metrics.csv", index=False)
    metric_columns = ["auroc", "auprc", "f1_at_0_5", "precision_at_0_5", "recall_at_0_5"]
    summary = result[metric_columns].agg(["mean", "std"]).to_dict()
    out_of_fold = pd.concat(all_predictions, ignore_index=True)
    out_of_fold.to_csv(args.outdir / "out_of_fold_predictions.csv", index=False)
    summary["pooled_out_of_fold"] = metrics(out_of_fold.label.to_numpy(), out_of_fold.score.to_numpy())
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.outdir / "config.json").write_text(json.dumps({
        "method": "Authors' STAG encoder and model with project-specific strict peptide-disjoint outer folds",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": str(device),
    }, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
