# Retrosig

## Install

```bash
conda create -f recipes/worfklow.yaml
conda activate retrosig
pip install --no-deps -e .
```

## Architecture

In the `src` folder:
* `paper`
    * `dataset`
        * `download.py`: create dataset files
            1 `metanetx.4_4.tsv`: raw file downloaded from Metanetx
            2 `metanetx.4_4.sanitize.csv`: metanetx smiles sanitized
            3 `dataset.csv`: whole dataset. Columns: SMILES, SIG, SIG-NEIGH, SIG-NBIT, SIG-NEIGH-NBIT, ECFP4 (int, lenght 2048)
            4 `dataset.train.csv`: subset of `dataset.csv` for training
            5 `dataset.valid.csv`: subset of `dataset.csv` for validate
            6 `dataset.test.csv`: subset of `dataset.csv` for testing
        * `tokenizer.py`: create 3 files (smiles, signature, ecfp4) given a dataset file
* `library`: directory managed by JL
* `retrosig`: translate jupyter notebook utilities to command line

## Paper

### `download.py`

```bash
conda activate retrosig
# cd retrosig

python src/paper/dataset/download.py \
    --output-directory-str <outdir>
```

### `tokenizer.py`

```bash
python src/paper/dataset/tokenizer.py \
    --input-dataset-csv <dataset created by download.py, csv> \
    --output-directory-str <outdir>
```
