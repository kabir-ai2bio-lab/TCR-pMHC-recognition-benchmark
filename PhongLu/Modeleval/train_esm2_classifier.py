"""Leakage-safe five-fold classifiers on frozen CDR3beta and peptide embeddings."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from PhongLu.Modeleval.models.peptide_balancing import make_peptide_sampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))


def load_maps(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=False)
    tcr_map = dict(zip(archive["tcr_sequences"].astype(str), archive["tcr_embeddings"]))
    peptide_map = dict(zip(archive["peptide_sequences"].astype(str), archive["peptide_embeddings"]))
    return tcr_map, peptide_map


def features(data: pd.DataFrame, tcr_map: dict[str, np.ndarray], peptide_map: dict[str, np.ndarray]) -> np.ndarray:
    missing_tcr = set(data.cdr3_beta.astype(str)) - set(tcr_map)
    missing_peptide = set(data.peptide.astype(str)) - set(peptide_map)
    if missing_tcr or missing_peptide:
        raise ValueError(f"Missing embeddings: {len(missing_tcr)} TCRs, {len(missing_peptide)} peptides")
    # Concatenation changes only the representation; the MLP learns the interaction.
    return np.concatenate([
        np.stack([tcr_map[s] for s in data.cdr3_beta.astype(str)]),
        np.stack([peptide_map[s] for s in data.peptide.astype(str)]),
    ], axis=1).astype(np.float32)


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.20), nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1))

    def forward(self, x):
        return self.network(x).squeeze(1)


class TCRESMClassifier(nn.Module):
    """CDR3beta--peptide network matching the published TCR-ESM topology.

    The original implementation has a 128-unit branch for each 1,280-dimensional
    ESM-1v embedding, concatenates them, then uses 512 and 256-unit dense layers.
    This PyTorch version preserves that topology while retaining this project's
    separate validation fold for early stopping.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim % 2:
            raise ValueError("TCR-ESM requires concatenated TCR and peptide embeddings")
        component_dim = input_dim // 2
        self.component_dim = component_dim
        self.tcr_branch = nn.Sequential(
            nn.Linear(component_dim, 128), nn.SiLU(), nn.BatchNorm1d(128), nn.Dropout(0.5)
        )
        self.peptide_branch = nn.Sequential(
            nn.Linear(component_dim, 128), nn.SiLU(), nn.BatchNorm1d(128), nn.Dropout(0.5)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1),
        )

    def forward(self, x):
        tcr, peptide = x[:, : self.component_dim], x[:, self.component_dim :]
        joined = torch.cat([self.tcr_branch(tcr), self.peptide_branch(peptide)], dim=1)
        return self.head(joined).squeeze(1)


def calculate_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = (scores >= 0.5).astype(int)
    return {"auroc": float(roc_auc_score(y, scores)), "auprc": float(average_precision_score(y, scores)), "f1_at_0_5": float(f1_score(y, labels)), "precision_at_0_5": float(precision_score(y, labels, zero_division=0)), "recall_at_0_5": float(recall_score(y, labels, zero_division=0))}


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--architecture", choices=("simple", "tcr-esm"), default="simple", help="simple is the existing ESM-2 MLP; tcr-esm matches Yadav et al.'s CDR3beta--peptide topology")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--balance-power", type=float, default=None, metavar="ALPHA", help="Peptide sampling exponent: 0=none, 0.5=gentle, 1=full inverse-frequency")
    parser.add_argument("--peptide-balanced", action="store_true", help="Deprecated compatibility alias for --balance-power 1")
    args = parser.parse_args()
    if args.peptide_balanced and args.balance_power not in (None, 1.0):
        parser.error("--peptide-balanced is an alias for --balance-power 1; do not combine it with another power")
    balance_power = 1.0 if args.peptide_balanced else (0.0 if args.balance_power is None else args.balance_power)
    if not 0.0 <= balance_power <= 1.0:
        parser.error("--balance-power must be between 0 and 1")
    # Existing ESM-2 defaults remain unchanged. TCR-ESM defaults follow the
    # reported CDR3beta--peptide configuration (Adam lr=0.009, batch=512,
    # up to 600 epochs); this project keeps a fold-local validation set.
    if args.epochs is None:
        args.epochs = 600 if args.architecture == "tcr-esm" else 100
    if args.batch_size is None:
        args.batch_size = 512 if args.architecture == "tcr-esm" else 256
    if args.learning_rate is None:
        args.learning_rate = 9e-3 if args.architecture == "tcr-esm" else 1e-3
    set_seed(args.seed)
    device = choose_device(args.device)
    data = pd.read_csv(args.splits)
    if sorted(data.fold.unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected exactly folds 0 through 4")
    tcr_map, peptide_map = load_maps(args.embeddings)
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for test_fold in range(5):
        validation_fold = (test_fold + 1) % 5
        train = data[~data.fold.isin([test_fold, validation_fold])].reset_index(drop=True)
        valid = data[data.fold.eq(validation_fold)].reset_index(drop=True)
        test = data[data.fold.eq(test_fold)].reset_index(drop=True)
        x_train, x_valid, x_test = (features(part, tcr_map, peptide_map) for part in (train, valid, test))
        # Scaling statistics are fitted on the training folds only.
        scaler = StandardScaler().fit(x_train)
        x_train, x_valid, x_test = (scaler.transform(x).astype(np.float32) for x in (x_train, x_valid, x_test))
        y_train = train.binder.to_numpy(dtype=np.float32)
        y_valid = valid.binder.to_numpy(dtype=np.float32)
        model_type = TCRESMClassifier if args.architecture == "tcr-esm" else EmbeddingMLP
        model = model_type(x_train.shape[1]).to(device)
        optimizer = (torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                     if args.architecture == "tcr-esm"
                     else torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4))
        loss_fn = nn.BCEWithLogitsLoss()
        train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        if balance_power > 0.0:
            sampler = make_peptide_sampler(train, balance_power, args.seed + test_fold)
            loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        best_state, best_loss, stale = None, float("inf"), 0
        for _epoch in range(args.epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
                loss.backward()
                optimizer.step()
            valid_scores = predict(model, x_valid, device)
            valid_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(valid_scores), torch.from_numpy(y_valid)))
            if valid_loss < best_loss:
                best_loss, stale = valid_loss, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        scores = predict(model, x_test, device)
        row = {"fold": test_fold, **calculate_metrics(test.binder.to_numpy(), scores), "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test)}
        all_metrics.append(row)
        pd.DataFrame({"cdr3_beta": test.cdr3_beta, "peptide": test.peptide, "binder": test.binder, "score": scores}).to_csv(args.outdir / f"fold_{test_fold}_predictions.csv", index=False)
        print(row)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary = metrics.drop(columns=["fold", "train_rows", "validation_rows", "test_rows"]).agg(["mean", "std"]).to_dict()
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.outdir / "config.json").write_text(json.dumps({"seed": args.seed, "architecture": args.architecture, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "balance_power": balance_power, "balancing_description": f"Training-fold row sampling proportional to peptide_count^-{balance_power:g}" if balance_power else "None"}, indent=2) + "\n")
    print("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()
