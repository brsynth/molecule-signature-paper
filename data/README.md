# Data organization

```text
.
├── emolecules
│   ├── analysis    < graphical comparitive distributions of chemical features >
│   ├── dataset     < step by step intermediary datasets >
│   ├── download    < source data (automatically downloaded) >
│   ├── sampling    < sampled subsets >
│   ├── splitting   < train / valid / test subsets from sampling >
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   └── valid.tsv
│   └── tokens      < Sentence Piece tokenizer models and statistics >
├── metanetx
│   ├── analysis
│   ├── dataset
│   │   ├── 1_reshaped.tsv
│   │   ├── 2_filtered.tsv
│   │   ├── 3_fingerprints.tsv
│   │   └── 4_deduped.tsv
│   ├── download
│   ├── sampling
│   ├── splitting   < 5-fold split subsets >
│   │   ├── test.tsv
│   │   ├── train_fold0.tsv
│   │   ├── train_fold1.tsv
│   │   ├── train_fold2.tsv
│   │   ├── train_fold3.tsv
│   │   ├── train_fold4.tsv
│   │   ├── valid_fold0.tsv
│   │   ├── valid_fold1.tsv
│   │   ├── valid_fold2.tsv
│   │   ├── valid_fold3.tsv
│   │   └── valid_fold4.tsv
│   └── tokens
├── models          < pre-trained and fine-tuned models for SMILES and ECFP tokenization >
└── tokens          < Sentence Piece tokenizer models (cp from emolecules) >
    ├── ECFP.model
    ├── ECFP.vocab
    ├── SMILES.model
    └── SMILES.vocab
```