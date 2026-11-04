"""Framework-independent peptide-frequency balancing utilities."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def peptide_balanced_weights(train: pd.DataFrame, balance_power: float) -> np.ndarray:
    """Return per-row weights ``count(peptide) ** -balance_power``.

    The caller must pass training-fold rows only. At power 0 there is no
    balancing; 0.5 is inverse-square-root balancing; and 1 gives every peptide
    equal expected representation.
    """
    if not 0.0 <= balance_power <= 1.0:
        raise ValueError("balance_power must be between 0 and 1")
    counts = train.peptide.value_counts()
    return train.peptide.map(lambda peptide: counts[peptide] ** -balance_power).to_numpy(dtype=np.float64)


class DeterministicWeightedSampler:
    """Yield weighted row indexes reproducibly while advancing between epochs."""

    def __init__(self, weights: np.ndarray, num_samples: int, seed: int):
        if len(weights) == 0 or num_samples <= 0:
            raise ValueError("weights and num_samples must be non-empty")
        self.probabilities = np.array(weights, dtype=np.float64, copy=True)
        self.probabilities /= self.probabilities.sum()
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        indexes = self.rng.choice(len(self.probabilities), self.num_samples, replace=True, p=self.probabilities)
        return iter(indexes.tolist())

    def __len__(self) -> int:
        return self.num_samples


def make_peptide_sampler(train: pd.DataFrame, balance_power: float, seed: int) -> DeterministicWeightedSampler:
    """Create a deterministic sampler from training-fold peptide counts only."""
    return DeterministicWeightedSampler(peptide_balanced_weights(train, balance_power), len(train), seed)
