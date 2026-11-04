# Model evaluations

All commands are run from the repository root. The model requirements differ; `PhongLu/requirements.txt` covers the base data pipeline and is not a complete model environment.

## NetTCR-2.0-style CDR3β + peptide baseline

Original paper: https://doi.org/10.1038/s42003-021-02610-3. Authors' repository and academic-use terms: https://github.com/mnielLab/NetTCR-2.0. This project implements the one-chain architecture locally in `models/nettcr2.py`; `train_nettcr2.py` evaluates the four project splits. The original paper's paired-chain model reported AUROC 0.89 under its own data and partitioning; the project results are not directly comparable.

With a compatible TensorFlow installation:

```bash
python -m PhongLu.Modeleval.train_nettcr2 \
  --splits PhongLu/data/preprocessing/splits/matched_strict_five_fold.csv \
  --outdir work/results/nettcr_matched_strict --seed 13 --epochs 30
```

Repeat with the other three split CSVs. The script writes predictions, `fold_metrics.csv`, and `summary.json`. Development summaries and fold metrics are in `results/nettcr2_*/`; see the known negative-sampling issue in the parent README.

## ESM-2 and TCR-ESM-style classifier

`generate_esm2_embeddings.py` creates sequence embeddings; `train_esm2_classifier.py` supports the simple and `--architecture tcr-esm` classifier settings. The encoder supports ESM-2 and ESM-1v; the TCR-ESM-style setup uses ESM-1v. This is an adapted classifier, not a claim to have reproduced the complete original published method. Run `python -m PhongLu.Modeleval.generate_esm2_embeddings --help` and `python -m PhongLu.Modeleval.train_esm2_classifier --help` for input and output options. These require PyTorch and `fair-esm`. `results/tcr_esm_three_seed_summary.json` is the available three-seed development summary; complete historical environment and model provenance need confirmation.

## STAG structural evaluation

Paper: https://doi.org/10.1109/TCBBIO.2024.3504235. Original code and structural dataset instructions: https://github.com/KavrakiLab/STAG_public. `evaluate_stag_strict.py` uses the authors' STAG repository and its graph data, but changes the evaluation to peptide-disjoint folds. Run `python -m PhongLu.Modeleval.evaluate_stag_strict --help` to specify the repository, input table, structures and output directory. It requires the STAG dependencies and public structural data; neither is included here. The report previously recorded AUROC 0.526 ± 0.014 across three seeds on that separate dataset; underlying STAG fold artifacts were not found in this workspace, so the result requires recovery and verification before publication. A VDJdb pilot structure was converted to a graph; this does not establish VDJdb classification performance.
