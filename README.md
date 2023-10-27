# Retrosig

## Install

```bash
conda env create -f recipes/workflow.yaml
conda activate retrosig
pip install --no-deps -e .
```

## Architecture

In the `src` folder:

- `paper/dataset`
  - `download.py`: download, sanitize, split, create dataset files
    - `metanetx.4_4.tsv`: raw file downloaded from Metanetx
    - `metanetx.4_4.sanitize.csv`: metanetx smiles sanitized
    - `dataset.csv`: whole dataset. Columns: SMILES, SIG, SIG-NEIGH, SIG-NBIT, SIG-NEIGH-NBIT, ECFP4 (int, lenght 2048)
    - `dataset.train.csv`: subset of `dataset.csv` for training
    - `dataset.valid.csv`: subset of `dataset.csv` for validation
    - `dataset.test.csv`: subset of `dataset.csv` for testing
  - `tokenizer.py`: build the molecule description vocabularies, create target-source file pairs to be used for training the models
    - `spm`: tokenizer vocabularies and models
    - `pairs`: target-source file pairs
- `library`: directory managed by JL
- `retrosig`: translate jupyter notebook utilities to command line

## Data

Donwload link (as of 2023/8/4): see `data` folder from the `RetroSig` shared folder of JL (Google Drive)

## Paper

### `download.py`

See embedded `--help`:

```bash
python src/paper/dataset/download_metanetx.py --help
```

Example :

```bash
python src/paper/dataset/download_metanetx.py \
    --output-directory-str datasets/metanetx/dataset \
    --parameters-seed-int 0 \
    --parameters-max-molecular-weight-int 500 \
    --parameters-max-dataset-size-int inf \
    --parameters-radius-int 2 \
    --parameters-valid-percent-float 10 \
    --parameters-test-percent-float 10 \
    1> download.log \
    2> download.err

```

### `tokenizer.py`

Usage

```bash
$ python src/paper/dataset/tokenizer.py --help

usage: tokenizer.py [-h] --input-directory-str INPUT_DIRECTORY_STR [--output-directory-str OUTPUT_DIRECTORY_STR] [--model-type-str {word,unigram}]
                    [--depictions-list DEPICTIONS_LIST [DEPICTIONS_LIST ...]] [--build-pairs-list BUILD_PAIRS_LIST [BUILD_PAIRS_LIST ...]]

Tokenize datasets.

options:
  -h, --help            show this help message and exit
  --input-directory-str INPUT_DIRECTORY_STR
                        Path of the input directory where to find train, test and valid datasets (CSV files). Files are expected to be named
                        "dataset.train.csv", "dataset.test.csv" and "dataset.valid.csv".
  --output-directory-str OUTPUT_DIRECTORY_STR
                        Path of the output directory. Default: current directory
  --model-type-str {word,unigram}
                        Model type for the tokenizer. Default: word.
  --depictions-list DEPICTIONS_LIST [DEPICTIONS_LIST ...]
                        List of depictions to tokenize. Default: ['SMILES', 'SIG', 'SIG-NBIT', 'SIG-NEIGH-NBIT', 'ECFP4']. Note: the depictions must be present
                        in the input files.
  --build-pairs-list BUILD_PAIRS_LIST [BUILD_PAIRS_LIST ...]
                        List of pairs of depictions to write. Default: ['ECFP4.SMILES', 'ECFP4.SIG-NEIGH-NBIT', 'SIG-NEIGH-NBIT.SMILES']. Note: the depictions
                        must be present in the input files.
```

Example

```bash
python src/paper/dataset/tokenizer.py \
    --input-directory-str data/metanetx/dataset \
    --output-directory-str data/metanetx/data \
    --model-type-str word \
    --depictions-list ECFP4 SIG-NEIGH-NBIT SMILES \
    --build-pairs-list ECFP4.SMILES ECFP4.SIG-NEIGH-NBIT SIG-NEIGH-NBIT.SMILES \
    &> tokenizer.log
```

## Build models

### Install chemigen

See: [https://github.com/brsynth/chemigen]
