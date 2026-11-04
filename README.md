# TCR–pMHC recognition benchmark

This repository contains Phong Lu's work on predicting whether a T-cell receptor (TCR) recognizes a peptide presented by MHC. The central question is how well models perform on peptides they have never seen during training.

```text
PhongLu/
├── data/preprocessing/       VDJdb extraction, filters, processed data, and folds
├── Modeleval/                NetTCR-2, ESM-2, TCR-ESM-style and STAG evaluation
│   ├── models/               local model and peptide-balancing components
│   └── results/              saved development result summaries
├── tests/                    data and embedding-classifier tests
└── requirements.txt          base data-processing dependencies
```

Start with [PhongLu/README.md](PhongLu/README.md) for data access, evaluation protocols, Colab notes, and commands. The uploaded Version-1 results are development results; their negative-sampling limitation is documented there.
