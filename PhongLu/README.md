# Phong Lu: data preparation and model evaluation

This benchmark asks whether a model can predict TCR–peptide recognition for peptide identities absent during supervised training. Version 1 now uses the **official VDJdb Zenodo release**, not the earlier raw-chunk snapshot.

## Official-release data

The input is `vdjdb_full.txt` from [`vdjdb-2026-06-03.zip`](https://doi.org/10.5281/zenodo.22104776). The archive SHA-256 is `9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1`. Its 192,753 records yield **73,654 unique positive TCRβ–peptide pairs** after the documented filters.

- [Processed positives and filter audit](data/preprocessing/official_2026_06_03/processed/)
- [Four split datasets and audits](data/preprocessing/official_2026_06_03/splits/)
- [Release source, checksums, filtering and reproduction commands](data/preprocessing/OFFICIAL_RELEASE.md)
- [Paired-chain structural candidates](data/preprocessing/official_2026_06_03/structural/): 49,586 candidate rows, not completed structures or STAG predictions.

IEDB was inspected but not merged: the current importer did not establish a consistently usable assay outcome from the specific export. This is a limitation of the present pipeline, not a claim that IEDB lacks TCR data or outcomes.

## Evaluation design

| Protocol | Dataset | Positive pairs | Purpose |
| --- | --- | ---: | --- |
| P1 | Original strict | 12,586 | Test peptides are absent from supervised training. |
| P2 | Original uniform | 72,364 | Test peptide identities can also occur in training. |
| P3 | Matched strict | 11,296 | Peptide-unseen evaluation on the matched positive pool. |
| P4 | Matched uniform | 11,296 | Peptide-seen evaluation on the same positive pool. |

Each dataset contains one generated negative per positive. Folds are assigned before negative generation. For each test fold, the next fold is validation and the other three are used for training. The current generator excludes pairs in the complete **filtered Version-1 positive pool** and prevents duplicate generated negatives across folds. Synthetic negatives are unobserved pairs, not confirmed non-binders; external positives and records excluded by filtering are not in the exclusion pool. Matched settings share positives but differ in fold assignments and generated negatives.

## Model results and reproduction

The official-release NetTCR-style evaluations and matched ESM-2 and TCR-ESM-style runs have been completed. See the [verified matched model comparison](Modeleval/results/official_2026_06_03/VERIFIED_MODEL_COMPARISON.md) and [evaluation instructions](Modeleval/README.md). ESM prediction files and classification reports are in `Modeleval/results/official_2026_06_03/esm_colab_verified/`.

Files under `data/preprocessing/processed/`, `data/preprocessing/splits/`, and `Modeleval/results/nettcr2_*` are **historical chunk-based outputs**. Do not mix them with the new official-release results.

For a local data setup, run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r PhongLu/requirements.txt
python -m pytest -q PhongLu/tests
```

See [OFFICIAL_RELEASE.md](data/preprocessing/OFFICIAL_RELEASE.md) for raw-release download and rebuild commands. The base requirements do not include every model dependency. The exact Colab software environment for the completed ESM runs was not retained.

## Structural model status

The public STAG structural evaluation used a different dataset. The official-release candidate table does not itself contain the modeled TCR–pMHC complexes required for a comparable STAG evaluation. A fair comparison requires defining eligible pairs, generating or obtaining suitable structures, and using the same held-out examples and evaluation rules.
