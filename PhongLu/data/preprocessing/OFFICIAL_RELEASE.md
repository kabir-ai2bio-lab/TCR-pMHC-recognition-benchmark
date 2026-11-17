# Official VDJdb release benchmark: 2026-06-03

This is a separate rebuild from the earlier chunk-based experiment. The source is the official `vdjdb-2026-06-03.zip` release ([Zenodo record](https://zenodo.org/records/22104776)), specifically `vdjdb-2026-06-03/vdjdb_full.txt`. The ZIP used locally has SHA-256 `9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1` and MD5 `b5b56a492de41d1bbad7558942ed6faa`, matching the Zenodo-listed MD5. The extracted `vdjdb_full.txt` has SHA-256 `55ee89fa1a0109f52f60fe30fd47c2f6842f94a0476d034b2d98b2511748bddc`. Verify the archive and extracted file before running. `source_record_id` identifies the file and its physical 1-based line number, including the header.

## Filtering audit

| Stage | Rows remaining |
| --- | ---: |
| Official release `vdjdb_full.txt` records | 192,753 |
| Human | 176,130 |
| MHC class I | 170,284 |
| Valid CDR3β and peptide amino acids | 145,192 |
| Nine-residue peptides | 119,424 |
| CDR3β length 10–18 | 117,267 |
| HLA-A*02 family | 90,532 |
| Unique CDR3β–peptide pairs | 73,654 |

The final pool has 71,061 unique CDR3β sequences, 915 peptides and eight observed MHC labels. These counts describe the present release and implemented filters; the filtering and pair-level deduplication choices remain the same as the earlier benchmark.

The generated positive table, audit, four split files, and split audits are saved under `PhongLu/data/preprocessing/official_2026_06_03/`. These are a new dataset and do not overwrite the previous chunk-based outputs.

For structural feasibility, the same official release was separately filtered for valid paired CDR3α/β, alpha and beta V/J fields, human class I, HLA-A*02, and 9-mer peptides. This produced 49,586 unique alpha–beta–peptide–MHC candidate rows spanning 849 peptides. These are candidate records, not generated 3D structures or STAG predictions. The structural filter differs from the beta-only Version-1 filter, so the two pools should not be compared as identical. Results and audit are in `official_2026_06_03/structural/`.

## Reproduction

From the repository root, with the verified `vdjdb_full.txt` extracted to `data/raw/vdjdb/vdjdb-2026-06-03/` and base dependencies installed:

```bash
python -m PhongLu.data.preprocessing.extract_vdjdb_v1 \
  --full-file data/raw/vdjdb/vdjdb-2026-06-03/vdjdb_full.txt \
  --output work/official_2026_06_03/processed/positive.csv \
  --audit work/official_2026_06_03/processed/audit.json
python -m PhongLu.data.preprocessing.build_splits \
  --input work/official_2026_06_03/processed/positive.csv \
  --output-dir work/official_2026_06_03/splits --seed 13
python -m PhongLu.data.preprocessing.build_matched_splits \
  --input work/official_2026_06_03/processed/positive.csv \
  --output-dir work/official_2026_06_03/splits --seed 13
python -m pytest -q PhongLu/tests
```

| Dataset | Positive pairs | Synthetic negatives |
| --- | ---: | ---: |
| Original strict | 12,586 | 12,586 |
| Original uniform | 72,364 | 72,364 |
| Matched strict | 11,296 | 11,296 |
| Matched uniform | 11,296 | 11,296 |

The strict splits are peptide-disjoint across folds. The synthetic negative generator samples TCRs within each fold and rejects pairs known positive in the full **filtered Version-1 positive pool**, including those removed by downsampling, as well as negative pairs generated in earlier folds. It does not establish that a generated pair is experimentally non-binding, and the exclusion set does not contain external databases or positives excluded by the benchmark filters. Across all four generated files, checks found zero full-pool positive collisions and zero duplicate pairs.

Earlier NetTCR and TCR-ESM results outside the `official_2026_06_03/` results folder use the older chunk snapshot and earlier negative-generation code. New NetTCR runs and matched ESM evaluations are recorded separately in `PhongLu/Modeleval/results/official_2026_06_03/`. STAG has not yet been evaluated on this release.
