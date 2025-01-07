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

## Architecture


In the `src` folder:

- `paper/dataset`
  - `download.py`: download, sanitize, split, create dataset files
    - `metanetx.4_4.tsv`: raw file downloaded from Metanetx
    - `metanetx.4_4.sanitize.csv`: metanetx smiles sanitized
    - `dataset.csv`: whole dataset. Columns: SMILES, SIG, SIG-NEIGH, SIG-NBIT,
      SIG-NEIGH-NBIT, ECFP4 (int, lenght 2048)
    - `dataset.train.csv`: subset of `dataset.csv` for training
    - `dataset.valid.csv`: subset of `dataset.csv` for validation
    - `dataset.test.csv`: subset of `dataset.csv` for testing
  - `tokenizer.py`: build the molecule description vocabularies, create
    target-source file pairs to be used for training the models
    - `spm`: tokenizer vocabularies and models
    - `pairs`: target-source file pairs
- `library`: directory managed by JL
- `retrosig`: translate jupyter notebook utilities to command line

## Data organization

```text
❯ tree -L 3 data
data
├── emolecules
└── metanetx
    ├── dataset ..................... # Prepared datasets
    │   ├── db_descriptors.tsv.gz ... # Ready-to-use file (step 3/3)
    │   ├── db_filtered.tsv.gz ...... # Filtered file (step 2/3)
    │   ├── db_reshape.tsv.gz ....... # Reshaped file (step 1/3)
    │   ├── db_sampled_10k.tsv.gz ... # Samples generated from the filtered file
    │   ├── db_sampled_1k.tsv.gz .... #
    │   └── db_sampled_50k.tsv.gz ... #
    ├── download .................... # Raw source
    └── spm ......................... # Sentence Piece trained models
```

## Download, reshape, filter, sample datasets

```bash
❯ python src/paper/dataset/prepare.py --help
usage: prepare.py [-h] [--db DB] [--show_progress] [--workers WORKERS] [--download_again] [--reshape_again]
                  [--filter_max_mw MW] [--filter_again] [--fingerprints_redo] [--fingerprints_resume] [--dedupe_again]
                  [--dedupe_from_db EXTERNAL DB] [--dedupe_from_db_again] [--sample_sizes SIZE [SIZE ...]]
                  [--sample_seed SEED] [--sample_again] [--split_test_proportion PROPORTION]
                  [--split_valid_proportion PROPORTION] [--split_method METHOD] [--split_num_folds FOLDS]
                  [--split_seed SEED] [--split_again] [--analyze_redo] [--analyze_chunksize CHUNKSIZE]
                  [--compare_redo]
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
  --filter_max_mw MW    Maximum molecular weight to filter (default: 500)
  --filter_again        Filter again (default: False)

Fingerprints options:
  --fingerprints_redo   Recompute fingerprints (default: False)
  --fingerprints_resume
                        Resume computation of fingerprints (default: False)

Deduplicate options:
  --dedupe_again        Deduplicate again (default: False)
  --dedupe_from_db EXTERNAL DB
                        Deduplicate from external dataset (only used when main DB is emolcules, default: metanetx)
  --dedupe_from_db_again
                        Deduplicate from external dataset again (default: False)

Sample options:
  --sample_sizes SIZE [SIZE ...]
                        Size of samples (default: [1000, 10000, 5000000])
  --sample_seed SEED    Random seed for sampling (default: 42)
  --sample_again        Sample again (default: False)

Split options:
  --split_test_proportion PROPORTION
                        Proportion of test set (default: 0.1)
  --split_valid_proportion PROPORTION
                        Proportion of validation set (only for 'train_valid_split' method) (default: 0.1)
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

### Prepare datasets for learning

```bash
❯ python src/paper/learning/prepare.py --help
usage: tokens.py [-h] [--db DB] [--token_fingerprints_types FINGERPRINT [FINGERPRINT ...]] [--token_vocab_sizes SIZE [SIZE ...]] [--token_min_presence PRESENCE] [--verbosity LEVEL]

Tokenize dataset

options:
  -h, --help            show this help message and exit
  --db DB               Dataset to tokenize (default: metanetx; choices: metanetx, emolecules)
  --token_fingerprints_types FINGERPRINT [FINGERPRINT ...]
                        Fingerprints to tokenize (default: ['SMILES', 'SIGNATURE', 'ECFP'])
  --token_vocab_sizes SIZE [SIZE ...]
                        Vocabulary size for each fingerprint (default: {'SMILES': 0, 'SIGNATURE': 0, 'ECFP': 0})
  --token_min_presence PRESENCE
                        Minimal presence of a token in the dataset (default: 0.0001)
  --verbosity LEVEL     Verbosity level (default: INFO)
```

## Build models

### Install chemigen

See: [https://github.com/brsynth/chemigen]
