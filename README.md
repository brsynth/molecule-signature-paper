# Retrosig

## Install

```bash
conda env create -f recipes/environment.yaml
conda activate signature-paper
pip install --no-deps -e .
```

Since the signature is not yet publicly available, the signature code has to be
installed from the source code:

```bash
conda activate signature-paper
git clone git@github.com:brsynth/signature.git lib/signature  # Credentials required
pushd lib/signature
pip install --no-deps -e .
popd
```

## Code architecture

```text
❯ tree -L 3 src
.
└── paper
    ├── dataset  # code for preparing datasets for learning
    │   ├── prepare.py
    │   ├── tokens.py
    │   └── utils.py
    └── learning  # code for generative model
        ├── config.py
        ├── data.py
        ├── model.py
        ├── predict.py
        ├── train.py
        └── utils.py
```

## Data organization

```text
❯ tree -L 3 data
data
├── emolecules          # data about the emolecules database
│   ├── download        # source data
│   │   └── emol_raw_2024-07-01.tsv.gz
│   ├── sampling        # sampled subset
│   │   ├── 0-sample_1k.tsv
│   │   ├── 1-sample_10k.tsv
│   │   └── 2-sample_5000k.tsv
│   ├── splitting       # train / valid / test subsets from sampling
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   └── valid.tsv
│   └── tokens         # alphabets file from / for Sentence Piece tokenizer
│       ...
│       ├── SMILES.log
│       ├── SMILES.model
│       ├── SMILES.stats
│       ├── SMILES.stats_fp
│       └── SMILES.vocab
└── metanetx            # data about the metanetx database
    ├── analysis        # graphical distributions of some chemical features accross subsets
    │   ├── 0-sample_1k
    │   ├── 1-sample_10k
    │   ├── 2-sample_5000k
    │   ├── 4_deduped
    │   ├── comparative
    │   ├── test
    │   ├── train_fold0
    │   ...
    ├── dataset         # step by step datasets
    │   ├── 1_reshaped.tsv
    │   ├── 2_filtered.tsv
    │   ├── 3_fingerprints.tsv
    │   └── 4_deduped.tsv
    ├── download        # source data
    │   └── mnx_raw_4_4.tsv
    ├── sampling        # sampled subset (from dataset/4_deduped.tsv)
    │   ├── 0-sample_1k.tsv
    │   ├── 1-sample_10k.tsv
    │   └── 2-sample_5000k.tsv
    ├── splitting       # 5-fold split subsets
    │   ├── test.tsv
    │   ├── train_fold0.tsv
    │   ├── ...
    │   ├── valid_fold0.tsv
    │   └── ...
    └── tokens          # alphabets file from / for Sentence Piece tokenizer
        ...
        ├── SMILES.log
        ├── SMILES.model
        ├── SMILES.stats
        ├── SMILES.stats_fp
        └── SMILES.vocab
```

## Run

### Prepare datasets (download, reshape, filter, sample, analyze)

```bash
❯ python src/paper/dataset/prepare.py --help
usage: prepare.py [-h] [--db DB] [--show_progress] [--workers WORKERS] [--download_again]
                  [--reshape_again] [--filter_again] [--filter_max_mw MW] [--fingerprints_redo]
                  [--fingerprints_resume] [--dedupe_again] [--dedupe_from_db EXTERNAL DB]
                  [--dedupe_from_db_again] [--sample_again] [--sample_sizes SIZE [SIZE ...]]
                  [--sample_seed SEED] [--split_test_proportion PROPORTION]
                  [--split_valid_proportion PROPORTION] [--split_method METHOD]
                  [--split_num_folds FOLDS] [--split_seed SEED] [--split_again] [--analyze_redo]
                  [--analyze_chunksize CHUNKSIZE] [--compare_redo]
                  {all,download,reshape,filter,fingerprints,dedupe,sample,split,analyze,compare}
                  [{all,download,reshape,filter,fingerprints,dedupe,sample,split,analyze,compare} ...]

Prepare dataset

positional arguments:
  {all,download,reshape,filter,fingerprints,dedupe,sample,split,analyze,compare}
                        Actions to perform (default: ['all'])

options:
  -h, --help            show this help message and exit

General options:
  --db DB               Database to use (default: metanetx)
  --show_progress       Show progress bar (default: False)
  --workers WORKERS     Number of workers (default: 5)

Download options:
  --download_again      Download again database (default: False)

Reshape options:
  --reshape_again       Reshape again (default: False)

Filter options:
  --filter_again        Filter again (default: False)
  --filter_max_mw MW    Maximum molecular weight to filter (default: 500)

Fingerprints options:
  --fingerprints_redo   Recompute fingerprints (default: False)
  --fingerprints_resume
                        Resume computation of fingerprints (default: False)

Deduplicate options:
  --dedupe_again        Deduplicate again (default: False)
  --dedupe_from_db EXTERNAL DB
                        Deduplicate from external dataset (only used when main DB is emolcules,
                        default: metanetx)
  --dedupe_from_db_again
                        Deduplicate from external dataset again (default: False)

Sample options:
  --sample_again        Sample again (default: False)
  --sample_sizes SIZE [SIZE ...]
                        Size of samples (default: [1000, 10000, 5000000])
  --sample_seed SEED    Random seed for sampling (default: 42)

Split options:
  --split_test_proportion PROPORTION
                        Proportion of test set (default: 0.1)
  --split_valid_proportion PROPORTION
                        Proportion of validation set (only for 'train_valid_split' method)
                        (default: 0.1)
  --split_method METHOD
                        Split method (default: kfold; choices: train_valid_split, kfold)
  --split_num_folds FOLDS
                        Number of folds (only for 'kfold' method) (default: 5)
  --split_seed SEED     Random seed for splitting (default: 42)
  --split_again         Split again (default: False)

Analyze options:
  --analyze_redo        Redo analysis (default: False)
  --analyze_chunksize CHUNKSIZE
                        Chunksize for analysis (default: 10000)

Compare options:
  --compare_redo        Redo comparisons (default: False)
```

### Build tokenizer models

```bash
❯ python src/paper/dataset/tokens.py --help
usage: tokens.py [-h] [--db DB] [--token_fingerprints_types FINGERPRINT [FINGERPRINT ...]]
                 [--token_vocab_sizes SIZE [SIZE ...]] [--token_min_presence PRESENCE]
                 [--verbosity LEVEL]

Tokenize dataset

options:
  -h, --help            show this help message and exit
  --db DB               Dataset to tokenize (default: metanetx; choices: metanetx, emolecules)
  --token_fingerprints_types FINGERPRINT [FINGERPRINT ...]
                        Fingerprints to tokenize (default: ['SMILES', 'SIGNATURE', 'ECFP'])
  --token_vocab_sizes SIZE [SIZE ...]
                        Vocabulary size for each fingerprint (default: {'SMILES': 0, 'SIGNATURE':
                        0, 'ECFP': 0})
  --token_min_presence PRESENCE
                        Minimal presence of a token in the dataset (default: 0.0001)
  --verbosity LEVEL     Verbosity level (default: INFO)
  ```

### Train models

### Install chemigen

See: [https://github.com/brsynth/chemigen]
