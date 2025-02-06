# Supporting content for the Molecule Signature paper

[![Github Version](https://img.shields.io/github/v/release/brsynth/molecule-signature?display_name=tag&sort=semver&logo=github)](version)
[![Github Licence](https://img.shields.io/github/license/brsynth/molecule-signature?logo=github)](LICENSE.md)

This repository contains code to support the Molecule Signature publication. See citation for details.

## Table of Contents
- [1. Repository structure](#1-repository-structure)
  - [1.1. Datasets](#11-datasets)
  - [1.2. Supporting Notebooks](#12-supporting-notebooks)
  - [1.3. Source code](#13-source-code)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
  - [3.1. Preparing datasets](#31-preparing-datasets)
  - [3.2 Deterministic enumeration](#32-deterministic-enumeration)
  - [3.3. Train generative models](#33-train-generative-models)
  - [3.4. Predict molecules with generative models](#34-predict-molecules-with-generative-models)
- [4. Citation](#4-citation)

## 1. Repository structure

```text
.
├── data       < placeholder for data files >
│   └── ..
├── notebooks  < supporting jupyter notebooks >
│   ├── 1.enumeration_create_alphabets.ipynb
│   ├── 2.enumeration_results.ipynb
│   ├── 3.analysis_alphabets.ipynb
│   ├── 4.generation_evaluation.ipynb
│   ├── 5.generation_recovery.ipynb
│   └── handy.py
└── src        < source code for data preparation and modeling >
    └── paper
        ├── dataset
        └── learning

```

### 1.1. Datasets

The `data` directory is the place where put required data files to be used by the code. `emolecules` and `metanetx` subdirectories are created at execution time. See [data organization README](data/README.md) for details.

### 1.2. Supporting Notebooks

The `notebooks` directory contains Jupyter notebooks that support figures and tables from the paper. The `handy.py` file contains utility functions that are used in some of the notebooks.

### 1.3. Source code

The `src` directory contains the source code for the paper. The code is organized in two main directories: `dataset` for preparing datasets and `learning` for training and using the generative model. See [Usage](#3-usage) for details on how to run the code.

## 2. Installation

The following steps will set up a `signature-paper` conda environment.

0. **Install Conda:**

    The conda package manager is required. If you do not have it installed, you
    can download it from [here](https://docs.conda.io/en/latest/miniconda.html).
    Follow the instructions on the page to install Conda. For example, on
    Windows, you would download the installer and run it. On macOS and Linux,
    you might use a command like:

    ```bash
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ```

    Follow the prompts on the installer to complete the installation.

1. **Install dependencies:**

    ```bash
    conda env create -f recipes/environment.yaml
    conda activate signature-paper
    pip install --no-deps -e .
    ```

2. **Download data:**

    Precomputed alphabets, trained generative models and most important datasets are available as a Zenodo archive: <https://doi.org/10.5281/zenodo.5528831>. Extract the files and place them in the `data` directory.

3. **Optionnaly (for dev): set the signature package from source**

    Installing the signature code from source is optional may be useful for development
    purposes. This will allow you to make changes to the signature code and see the
    effects in the paper code without having to reinstall the package.

    ```bash
    conda activate signature-paper
    
    # Remove the packaged version
    conda remove molecule-signature
    
    # Set up the source version
    git clone git@github.com:brsynth/molecule-signature.git lib/molecule-signature
    pushd lib/molecule-signature
    pip install --no-deps -e .
    popd
    ```

## 3. Usage

### 3.1. Preparing datasets

The `src/paper/dataset` module contains code to prepare datasets for training and evaluation. The `prepare` command will create the datasets in the `data` directory.

- **Getting help**

    ```bash
    python -u src/paper/dataset/prepare.py --help
    python -u src/paper/dataset/tokens.py --help
    ```

- **eMolecules dataset**

    Prepare the emolecules dataset. **Cautions:** the emolecules dataset is large and may take a long time to download and process (as well as a substantial disk space and RAM consumption).

    ```bash
    # Datasets
    python src/paper/dataset/prepare.py all --db emolecules --workers 10  --show_progress --split_method train_valid_split

    # Tokenizers
    python src/paper/dataset/tokens.py --db emolecules --token_min_presence 0.0001
    ```

- **MetaNetX dataset**

    ```bash
    # Datasets
    python src/paper/dataset/prepare.py all --db metanetx --workers 10 --show_progress

    # Tokenizers
    python src/paper/dataset/tokens.py --db metanetx --token_min_presence 0.0001
    ```

### 3.2 Deterministic enumeration

Users will find explanations and examples on how to use the deterministic enumeration code in the notebooks folder, in particular:

- [1.enumeration_create_alphabets.ipynb](notebooks/1.enumeration_create_alphabets.ipynb): Create alphabets for deterministic enumeration.
- [2.enumeration_results.ipynb](notebooks/2.enumeration_results.ipynb): Get deterministic enumeration results.
- [3.analysis_alphabets.ipynb](notebooks/3.analysis_alphabets.ipynb): Analyze alphabets.

### 3.3. Train generative models

**Cautions**: settings may need to be adjusted depending on the available resources (e.g., GPU memory, disk space, etc.) and the HPC environment.

- **Getting help**

    ```bash
    python -u src/paper/learning/train.py --help
    ```

- **Pre-train model (from eMolecules datasets)**

    ```bash
    python src/paper/learning/train.py \
        --db emolecules \
        --source ECFP \
        --target SMILES \
        --dataloader_num_workers 3 \
        --enable_mixed_precision True \
        --ddp_num_nodes 1 \
        --split_method train_valid_split \
        --train_fold_index 0 \
        --data_max_rows -1 \
        --out_dir ${WORK}/hpc \
        --out_sub_dir NULL \
        --model_dim 512 \
        --model_num_encoder_layers 3 \
        --model_num_decoder_layers 3 \
        --scheduler_method plateau \
        --scheduler_lr 0.0001 \
        --scheduler_plateau_patience 1 \
        --scheduler_plateau_factor 0.1 \
        --early_stopping_patience 5 \
        --early_stopping_min_delta 0.0001 \
        --train_max_epochs 200 \
        --train_batch_size 128 \
        --train_accumulate_grad_batches 1 \
        --train_val_check_interval 1 \
        --train_seed 42 \
        --finetune false \
        --finetune_lr 0.0001 \
        --finetune_freeze_encoder false \
        --finetune_checkpoint None
    ```

- **Fine-tune model (from MetaNetX datasets, fold 0)**

    ```bash
    python src/paper/learning/train.py \
        --db metanetx \
        --source ECFP \
        --target SMILES \
        --dataloader_num_workers 3 \
        --enable_mixed_precision True \
        --ddp_num_nodes 1 \
        --split_method kfold \
        --train_fold_index 0 \
        --data_max_rows -1 \
        --out_dir ${WORK}/hpc/llogs \
        --out_sub_dir FOLD_0 \
        --model_dim 512 \
        --model_num_encoder_layers 3 \
        --model_num_decoder_layers 3 \
        --scheduler_method plateau \
        --scheduler_lr 0.0001 \
        --scheduler_plateau_patience 2 \
        --scheduler_plateau_factor 0.1 \
        --early_stopping_patience 6 \
        --early_stopping_min_delta 0.0001 \
        --train_max_epochs 200 \
        --train_batch_size 128 \
        --train_accumulate_grad_batches 1 \
        --train_val_check_interval 1 \
        --train_seed 42 \
        --finetune true \
        --finetune_lr 0.0001 \
        --finetune_freeze_encoder true \
        --finetune_checkpoint <path_to_pretrain_model_checkpoint>
    ```

### 3.4. Predict molecules with generative models

The `src/paper/learning/predict.py` script can be used to generate molecules from the trained models. The script requires a trained model checkpoint and a tokenizer, which can be downloaded from the Zenodo archive(see [Installation](#2-installation)).

- **Generate molecules from the fine-tuned model**

    ```bash
    python src/paper/learning/predict.py \
        --model_path "data/models/finetuned.ckpt" \
        --model_source_tokenizer "data/tokens/ECFP.model" \
        --model_target_tokenizer "data/tokens/SMILES.model" \
        --pred_mode "beam"
    ```

- **Getting help**

    ```bash
    python -u src/paper/learning/predict.py --help
    ```

- **Other examples within notebooks**

    The [4.generation_evaluation.ipynb](notebooks/4.generation_evaluation.ipynb) and [5.generation_recovery.ipynb](notebooks/5.generation_recovery.ipynb) notebooks bring examples of how to use the `predict.py` script as a library to generate molecules within a master script.

## 4. Citation

Meyer, P., Duigou, T., Gricourt, G., & Faulon, J.-L. Reverse Engineering Molecules from Fingerprints through Deterministic Enumeration and Generative Models. In preparation.
