# Official VDJdb release: NetTCR-style evaluation

**Status:** All four NetTCR evaluations completed. These are new results on the 2026-06-03 official `vdjdb_full.txt`, not results from the previous chunk snapshot.

The model and five-fold procedure use seed 13, the next fold for validation, up to 30 epochs, and early stopping. Numbers are mean ± sample standard deviation across five test folds, not uncertainty across independently resampled splits or random seeds. AUPRC is average precision. Synthetic negatives are not measured non-binders.

| Evaluation | AUROC | AUPRC | Status |
| --- | ---: | ---: | --- |
| Matched strict / unseen peptide | 0.510 ± 0.011 | 0.509 ± 0.018 | Complete |
| Matched uniform / seen peptide | 0.671 ± 0.008 | 0.671 ± 0.010 | Complete |
| Original strict / unseen peptide | 0.533 ± 0.027 | 0.549 ± 0.020 | Complete |
| Original uniform / seen peptide | 0.940 ± 0.002 | 0.937 ± 0.002 | Complete |

The matched split comparisons use the same set of 11,296 positive pairs, but different fold allocations and fold-local negatives. The seen-peptide AUROC is approximately 0.161 higher in seed 13. This supports a gap between these two evaluation settings, not a causal proof that novelty alone explains it; test composition, training distribution, and synthetic-negative composition can also vary. The matched strict AUROC is near 0.5.

## Training-seed stability on fixed splits

All three seeds reuse the same split assignment and generated negatives. Each seed value below is the mean of its five test folds; the last row summarizes the three seed-level means. This does not measure variation due to new data splits.

| Seed | Strict AUROC | Strict AUPRC | Uniform AUROC | Uniform AUPRC |
| --- | ---: | ---: | ---: | ---: |
| 13 | 0.510 | 0.509 | 0.671 | 0.671 |
| 42 | 0.512 | 0.505 | 0.664 | 0.661 |
| 123 | 0.504 | 0.500 | 0.669 | 0.669 |
| Mean ± SD across seeds | 0.509 ± 0.004 | 0.505 ± 0.004 | 0.668 ± 0.004 | 0.667 ± 0.005 |

The observed seen/unseen gap persists across these training seeds. A threshold-based classification report for each completed seed-13 setting is saved next to its summary; it pools the five out-of-fold prediction sets and can therefore differ from the mean of fold-level F1 scores.

The split is designed around peptide identity, not TCR identity. Across the five folds, an average of approximately 3.8% of matched-strict test rows and 3.6% of matched-uniform test rows have a CDR3β sequence that occurs in the three training folds. This is allowed by the stated protocol; it means the experiment does not establish generalization to unseen TCRs as well as unseen peptides.

The earlier chunk-based matched results were 0.506 strict and 0.667 uniform AUROC. The new 0.510 and 0.671 values have the same broad pattern, but this is not a controlled head-to-head measurement: input records, sample sizes, and the negative exclusion rule changed. Rounding and fold variation should not be presented as a statistically significant improvement.

The official-release ESM-2 and TCR-ESM-style matched results were subsequently verified and are documented in [VERIFIED_MODEL_COMPARISON.md](VERIFIED_MODEL_COMPARISON.md). STAG requires paired TCRα/β, peptide, MHC, and a 3D complex for each evaluated example; a matching official-release structural dataset does not yet exist. Its earlier public-dataset results cannot be directly compared to these numbers.
