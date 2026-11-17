# Official VDJdb release: matched model comparison

## Purpose and data

We compared three local models on the same matched strict and uniform datasets derived from the official 2026-06-03 VDJdb release. The matched datasets each contain 11,296 positive TCRβ–peptide pairs and 11,296 synthetic negatives. Synthetic negatives are unobserved pairs, not experimentally confirmed non-binders. The two settings use the same positive pool, but their fold assignments and generated negatives differ.

The **strict** split holds out peptide identities: a test peptide is absent from training. The **uniform** split permits the same peptide identities in training and testing. Neither setting is designed to make all test TCRs unseen. For each test fold, the next fold is validation and the other three are training.

## Verification

The ESM result archive contained all 12 expected model/split/seed folders. For each run we checked that all five prediction files match the appropriate official-release split rows and labels, that prediction scores are finite and between zero and one, and that the AUROC, AUPRC, precision, recall, and F1 summaries can be recomputed from those predictions. All 12 checks passed. The saved `config.json` files confirm the recorded classifier settings and seeds. These checks do not independently establish the identity of the pretrained embedding checkpoint, package versions, or provenance of the training data used by the pretrained model; those details should be documented from the Colab execution environment.

## Results

Each number below is the mean ± sample standard deviation of the **three seed-level means** for seeds 13, 42, and 123. Each seed-level mean averages five outer test folds. The splits and synthetic negatives were fixed across seeds, so the reported standard deviation measures training-seed variation, not independent split uncertainty. AUPRC is average precision.

| Model | Setting | AUROC | AUPRC |
| --- | --- | ---: | ---: |
| NetTCR-2.0-style | Strict / unseen peptide | 0.509 ± 0.004 | 0.505 ± 0.004 |
| NetTCR-2.0-style | Uniform / seen peptide | 0.668 ± 0.004 | 0.667 ± 0.005 |
| ESM-2 embeddings + classifier | Strict / unseen peptide | 0.510 ± 0.003 | 0.509 ± 0.001 |
| ESM-2 embeddings + classifier | Uniform / seen peptide | 0.720 ± 0.001 | 0.731 ± 0.001 |
| ESM-1v embeddings + TCR-ESM-style classifier | Strict / unseen peptide | 0.513 ± 0.003 | 0.512 ± 0.002 |
| ESM-1v embeddings + TCR-ESM-style classifier | Uniform / seen peptide | 0.686 ± 0.008 | 0.686 ± 0.011 |

The checkpoint names are those specified by the prepared notebook; the uploaded classifier `config.json` files do not independently record the embedding checkpoint. The TCR-ESM-style implementation is an adaptation and is not claimed to be an exact reproduction of the published model.

## Classification reports

Each of the 12 ESM run folders under `esm_colab_verified/` contains `classification_report.json` with pooled out-of-fold precision, recall, F1, support, and a confusion matrix at threshold 0.5. This report pools all five held-out folds for one seed; its values may differ from the average of fold-level metrics in `summary.json`. A fixed threshold of 0.5 is a diagnostic and does not establish calibrated binding probabilities.

## Interpretation and limitations

All three models showed substantially higher AUROC for familiar peptides than for held-out peptide identities. Their matched strict AUROCs were close to 0.5, so these results do not demonstrate reliable generalization to unseen peptides. ESM-2 ranked better on the uniform split in this experiment. The small differences on the strict split should not be treated as proof that one model is superior, because folds, negatives, training distributions, and other aspects of the evaluation vary and no statistical comparison on independent splits was performed.

The official-release negative generator rejects a candidate found in the full filtered Version-1 positive pool and prevents duplicate negatives across folds. It does not exclude every possible positive in external databases or outside the filtered pool. The high apparent uniform performance should also be interpreted in the context of the constructed negatives and dataset composition.

ESM-2 and ESM-1v are pretrained models. Holding peptides out of supervised classifier training does not by itself prove that test sequences were absent from the pretrained encoder's data. This evaluation does not determine that overlap.

STAG is not included in this table. The earlier public structural evaluation used different data, and the official-release paired-chain candidates have not yet been converted into and evaluated as a matched structural dataset.
