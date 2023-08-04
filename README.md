# Retrosig

## Install

```bash
conda env create -f recipes/worfklow.yaml
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

## Paper

### `download.py`

```bash
conda activate retrosig

python src/paper/dataset/download.py \
    --output-directory-str <outdir>
```

### `tokenizer.py`

```bash
python src/paper/dataset/tokenizer.py \
    --input-directory-str <indir containing dataset.*.csv files> \
    --output-directory-str <outdir>
```
