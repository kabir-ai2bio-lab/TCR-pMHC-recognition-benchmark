# Phong Lu: data preparation and model evaluation

Version 1 uses human VDJdb records for HLA-A*02, 9-amino-acid peptides, and TCR CDR3β. The source archive produced 74,157 distinct positive CDR3β–peptide pairs after the filters in `data/preprocessing/preprocess.py`. These rules select a narrow benchmark; they do not mean excluded biological records are invalid.

## Source data and exact reproduction

VDJdb: https://github.com/antigenomics/vdjdb-db. The input used in the original run was the repository archive `vdjdb-db-master.zip`, read as tab-separated `chunks/*.txt` files. The local archive has SHA-256 `ae27aed7e44ffdbadca97540c91360c35941285463dac3d9c096d38d0f3a866e`. A newer GitHub download will not necessarily match this checksum or reproduce the counts. The exact archive is not included here; obtain that matching archive from the project owner and verify its checksum. A public, permanently accessible snapshot is still needed.

The committed `data/preprocessing/processed/` folder contains the processed positive table and filter audit. The `splits/` folder contains the four development split files and their audits. The filters and original input column mappings are implemented in `extract_vdjdb_v1.py`, `import_sources.py`, and `preprocess.py`.

The IEDB export was inspected but was not included in Version 1 because its selected export did not provide a consistently usable row-level outcome in the current pipeline.

## Installation and Colab

From the repository root, install the base processing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r PhongLu/requirements.txt
```

For Google Colab, clone this repository in a notebook, select a GPU if the model requires one, and run the same commands from the repository root. Mount Google Drive only when needed for persistent inputs or outputs. Record Python, package versions, accelerator type, seed, source archive checksum, and code revision. The available project scripts include local and Colab workflows; the precise environment for every historical run has not yet been recovered. The base requirements do not include TensorFlow, PyTorch, transformers, or PyTorch Geometric; install compatible versions for the selected model and record them before running. A fully pinned model environment is still needed.

## Rebuild data

Place the verified archive under `data/raw/vdjdb/`, extract it so that `data/raw/vdjdb/vdjdb-db-master/chunks/` exists, then run from the repository root. Use a new output directory to preserve the committed development files:

```bash
python -m PhongLu.data.preprocessing.extract_vdjdb_v1 \
  --chunks-dir data/raw/vdjdb/vdjdb-db-master/chunks \
  --output work/processed/vdjdb_v1_positive.csv \
  --audit work/processed/vdjdb_v1_audit.json
python -m PhongLu.data.preprocessing.build_splits \
  --input work/processed/vdjdb_v1_positive.csv --output-dir work/splits --seed 13
python -m PhongLu.data.preprocessing.build_matched_splits \
  --input work/processed/vdjdb_v1_positive.csv --output-dir work/splits --seed 13
python -m pytest -q PhongLu/tests/test_data_pipeline.py
```

## Evaluation design

| Protocol | Split | Positive pairs | Purpose |
| --- | --- | ---: | --- |
| P1 original strict | Peptide-unseen | 12,741 | Test peptides are absent from training. |
| P2 original uniform | Peptide-seen | 72,803 | Peptide identities can appear in training and test. |
| P3 matched strict | Peptide-unseen | 11,387 | Strict split on the matched positive pool. |
| P4 matched uniform | Peptide-seen | 11,387 | Uniform split on the same positive pool as P3. |

Each uses five folds: one test fold, the next fold for validation, and three for training. One synthetic negative is generated per positive within each fold. These negatives are unobserved pairs, not experimentally confirmed non-binders.

**Known issue requiring review before a paper:** the current negative generator excludes positives in the same fold, but it does not check candidates against every positive in the full source pool. This can label a known positive from another fold as negative. Audit and correct this, rebuild the splits and rerun evaluations before treating results as final. Matched strict and uniform have the same positive pool, but their fold-local generated negatives need not be identical.

## Model evaluation and results

See [Modeleval/README.md](Modeleval/README.md) for model inputs, commands, sources, and result availability. Current NetTCR-style development summaries (mean AUROC across folds) are 0.511 original strict, 0.896 original uniform, 0.506 matched strict and 0.667 matched uniform. These are subject to the issue above. The source paper's reported paired-chain AUC and these CDR3β-only results use different datasets and evaluation designs, so they cannot be compared directly.

## Data and licensing

The positive dataset retains source identifiers and references. Review the database's redistribution and attribution terms and the lab's sharing policy before distributing it. NetTCR-2.0 and STAG are third-party software: consult their repositories for their respective licenses. Do not upload personal tokens, MODELLER keys, downloaded raw archives, or large model weights.
