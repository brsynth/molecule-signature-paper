#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
import random
from pandarallel import pandarallel
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import RDLogger
from rdkit.Chem import AddHs
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms, CalcNumRings
from sklearn.model_selection import KFold

from paper.dataset.utils import (
    setup_logger,
    log_config,
    open_file,
    mol_from_smiles,
    mol_filter,
    assign_stereo,
    get_smiles,
    get_signature,
    get_ecfp,
    CustomDataset,
    url_download,
)
from paper.learning.config import Config


# Utilities functions -----------------------------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def _get_nb_lines(file):
    with open_file(file, "r") as reader:
        return sum(1 for _ in reader)


def _get_nb_chunks(input_file: str, chunksize: int) -> int:
    return _get_nb_lines(input_file) // chunksize + 1


def _fingerprints(smiles):
    mol = mol_from_smiles(smiles)
    mol = assign_stereo(mol)
    smi = get_smiles(mol)
    sig = get_signature(mol, radius=2, nbits=2048)
    ecfp = get_ecfp(mol, radius=2, nbits=2048)
    return (
        smi,
        sig.to_string(morgans=False),
        ecfp,
        sig.to_string(morgans=True),
    )


def _get_mol_features(smiles: str) -> Dict[str, Any]:
    # List of features to compute
    composition = {"MW": 0, "HV": 0, "R": 0, "C": 0, "H": 0, "O": 0, "N": 0, "P": 0, "S": 0}

    mol = mol_from_smiles(smiles)

    # Return Nones if molecule is invalid
    if mol is None:
        return {feat: None for feat in composition}

    # Compute features
    mol = AddHs(mol)

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in composition:
            composition[symbol] += 1

    composition["MW"] = ExactMolWt(mol)
    composition["HV"] = CalcNumHeavyAtoms(mol)
    composition["R"] = CalcNumRings(mol)

    return composition


def download_dataset(config: Config) -> int:
    logger.info("Downloading dataset ...")
    if CONFIG.download_file.exists() and not CONFIG.download_again:
        logger.info("  L skipped, already exists, use --download_again to download again")
    else:
        url_download(url=CONFIG.download_url, path=CONFIG.download_file)
        logger.info("  L done")

    return 0


def reshape_dataset(config: Config) -> int:
    logger.info("Reshaping dataset ...")
    if CONFIG.dataset_reshaped_file.exists() and not CONFIG.reshape_again:
        logger.info("  L skipped, already exists, use --reshape_again to overwrite")
    else:
        count_written = 0

        with open_file(
            CONFIG.download_file, "r"
        ) as reader, open_file(
            CONFIG.dataset_reshaped_file, "w"
        ) as writer:
            if CONFIG.show_progress:  # Show progress bar
                reader = tqdm(
                    reader,
                    total=_get_nb_lines(CONFIG.download_file),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                )
            writer.write("ID\tSMILES\n")  # Header
            to_write = False  # Skip heading notes
            for index, line in enumerate(reader, start=1):
                # ID is metanetx
                # isosmiles is emolecules
                if line.startswith("#ID"):
                    def _get_info(line):
                        _id, _, _, _, _, _, _, _, smiles = line.split("\t")
                        return _id, smiles.strip()
                    to_write = True
                    continue
                elif line.startswith("isosmiles"):
                    def _get_info(line):
                        smiles, _id, _ = line.split(" ")
                        return _id, smiles
                    to_write = True
                    continue
                if to_write:
                    _id, smiles = _get_info(line)
                    writer.write(f"{_id}\t{smiles}\n")
                    count_written += 1

        logger.info(f"  L done (written: {count_written:,})")

    return 0


def filter_dataset(CONFIG: Config) -> int:
    logger.info("Filtering dataset ...")
    if CONFIG.dataset_filtered_file.exists() and not CONFIG.filter_again:
        logger.info("  L skipped, already exists, use --filter_again to overwrite")

    else:
        count_written = 0
        count_skipped = 0

        with open_file(
            CONFIG.dataset_reshaped_file, "r"
        ) as reader, open_file(
            CONFIG.dataset_filtered_file, "w"
        ) as writer:
            if CONFIG.show_progress:  # Show progress bar
                reader = tqdm(
                    reader,
                    total=_get_nb_lines(CONFIG.dataset_reshaped_file),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                )
            writer.write("ID\tSMILES\n")  # Header
            for index, line in enumerate(reader, start=1):
                if index == 1:  # Skip header
                    continue
                _ = line.split("\t")
                if len(_) != 2:  # Skip missing values
                    count_skipped += 1
                    continue
                else:
                    _id, smiles = _
                try:
                    mol = mol_from_smiles(smiles)
                    if mol is None:
                        count_skipped += 1
                        continue
                    mol = mol_filter(mol, max_mw=CONFIG.filter_max_mw)
                    if mol is None:
                        count_skipped += 1
                        continue
                    writer.write(line)
                    count_written += 1
                except Exception as err:
                    logger.error(f"  L {index, _id, err}")
                    count_skipped += 1

        logger.info(f"  L done (written: {count_written:,}, skipped: {count_skipped:,})")

    return 0


def compute_fingerprints(CONFIG: Config) -> int:
    logger.info("Computing fingerprints ...")

    CHUNKSIZE = 10000
    compute = True
    resume = False

    if CONFIG.dataset_fingerprints_file.exists():
        if CONFIG.fingerprints_resume:
            resume = True
            msg = "  L resuming computation of descriptors..."
            logger.info(msg)

        elif CONFIG.fingerprints_redo:
            msg = "  L recomputing fingerprints..."
            logger.info(msg)

        else:
            compute = False
            msg = (
                "  L skipped, already exists, use --fingerprints_redo to "
                "recompute descriptors or --fingerprints_resume to resume "
                "computation"
            )
            logger.info(msg)

    if compute:
        if resume:
            count_written = _get_nb_lines(CONFIG.dataset_fingerprints_file) - 1  # Skip header
            chunk_written = count_written // CHUNKSIZE
            open_mode = "a"

            msg = f"  L already computed: {count_written:,}"
            logger.info(msg)

        else:
            count_written = 0
            chunk_written = 0
            open_mode = "w"

        # Setting up parallel processing
        pandarallel.initialize(nb_workers=CONFIG.workers, verbose=1)

        with open_file(
            CONFIG.dataset_filtered_file, "r"
        ) as reader, open_file(
            CONFIG.dataset_fingerprints_file, open_mode
        ) as writer:
            # Read file in chunks
            reader = pd.read_csv(reader, sep="\t", iterator=True, chunksize=CHUNKSIZE)

            # Write header if needed
            if compute and not resume:
                writer.write("ID\tSMILES_0\tSMILES\tSIGNATURE\tECFP\tSIGNATURE_MORGANS\n")

            # Show progress bar
            if CONFIG.show_progress:
                reader = tqdm(
                    reader,
                    total=_get_nb_chunks(CONFIG.dataset_filtered_file, chunksize=CHUNKSIZE),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    unit=f" chunk(s) of {CHUNKSIZE:,} lines",
                )

            # Process each chunk
            for chunk_idx, df_chunk in enumerate(reader):
                if chunk_idx < chunk_written:  # Skip if chunk has already been processed
                    continue

                # Rename the SMILES column to SMILES_0 to avoid the error of having the same column
                # name when parallelizing the apply function
                df_chunk.rename(columns={"SMILES": "SMILES_0"}, inplace=True)

                # Compute descriptors in parallel
                df_chunk[["SMILES", "SIGNATURE", "ECFP", "SIGNATURE_MORGANS"]] = df_chunk.parallel_apply(  # noqa E501
                    lambda x: _fingerprints(x["SMILES_0"]), axis=1, result_type="expand"
                )

                # Write chunk to file
                df_chunk.to_csv(writer, sep="\t", index=False, header=False)  # Append chunk to file

                # Update counters
                count_written += len(df_chunk)

        logger.info(f"  L done (written: {count_written:,})")

    return 0


def dedupe_dataset(CONFIG: Config) -> int:
    logger.info("Deduplicating dataset ...")
    if CONFIG.dataset_deduped_file.exists() and not CONFIG.dedupe_again:
        logger.info("  L Skipped, already exists, use --dedupe_again to overwrite")

    else:
        count_written = 0
        count_skipped = 0

        dataset = CustomDataset(CONFIG.dataset_fingerprints_file)
        all_smiles = set()

        with open(CONFIG.dataset_deduped_file, "w") as writer:
            # Header
            writer.write(dataset.header)

            # Show progress
            if CONFIG.show_progress:
                dataset = tqdm(
                    dataset,
                    total=len(dataset),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    desc="Deduplicating dataset",
                )

            # Deduplicate and write
            for item in dataset:
                parts = item.split("\t")
                smiles = parts[CONFIG.data_col_indexes["SMILES"]]
                if smiles in all_smiles:
                    count_skipped += 1
                    continue
                else:
                    all_smiles.add(smiles)
                    writer.write(item)
                    count_written += 1

        logger.info(f"  L done (written: {count_written:,}, skipped: {count_skipped:,})")

    return 0


def dedupe_from_external_dataset(CONFIG: Config, CONFIG_EXT: Config) -> int:
    logger.info("Deduplicating dataset from external dataset ...")
    if CONFIG.dataset_deduped_ext_file.exists() and not CONFIG.dedupe_from_db_again:
        logger.info("  L Skipped, already exists, use --dedupe_from_db_again to overwrite")

    else:
        count_written = 0
        count_skipped = 0

        dataset_main = CustomDataset(CONFIG.dataset_deduped_file)
        dataset_ext = CustomDataset(CONFIG_EXT.dataset_deduped_file)

        # Load all SMILES from the external dataset (expected to be the smallest)
        ext_smiles = set(
            item.rstrip().split("\t")[CONFIG_EXT.data_col_indexes["SMILES"]]
            for item in dataset_ext
        )

        # Dedupe the main dataset
        with open(CONFIG.dataset_deduped_ext_file, "w") as writer:
            # Header
            writer.write(dataset_main.header)

            # Show progress
            if CONFIG.show_progress:
                dataset_main = tqdm(
                    dataset_main,
                    total=len(dataset_main),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    desc="Deduplicating dataset",
                )

            # Dedupe and write
            for item in dataset_main:
                parts = item.rstrip().split("\t")
                smiles = parts[CONFIG.data_col_indexes["SMILES"]]
                if smiles in ext_smiles:
                    count_skipped += 1
                    continue
                else:
                    writer.write(item)
                    count_written += 1

        logger.info(f"  L done (written: {count_written:,}, skipped: {count_skipped:,})")

    return 0


def sample_dataset(CONFIG: Config) -> int:
    logger.info("Sampling dataset ...")
    for sample_size, sample_file in zip(CONFIG.sample_sizes, CONFIG.sample_files):

        sample_file = Path(sample_file)
        logger.info(f"  L Sampling {sample_size:,} lines ...")

        if sample_file.exists() and not CONFIG.sample_again:
            msg = "    L skipped, already exists, use --sample_again to resample"
            logger.info(msg)

        else:
            dataset = CustomDataset(CONFIG.dataset_ready_file)  # May take some time to index

            # Check if sample size is valid
            if sample_size > len(dataset):
                sample_size = len(dataset)
                logger.warning(
                    f"    L sample size is larger than the dataset, "
                    f"reducing to {sample_size:,} which is the dataset size"
                )

            # Set seed within the loop to keep the same starting items
            random.seed(CONFIG.sample_seed)

            # Draw random indexes
            sample_indexes = random.sample(range(len(dataset)), sample_size)

            count_written = 0

            # Read and sample lines
            with open_file(sample_file, "w") as writer:

                # Show progress
                if CONFIG.show_progress:
                    sample_indexes = tqdm(
                        list(sample_indexes),
                        total=len(sample_indexes),
                        mininterval=1,
                        leave=False,
                        unit_scale=True,
                        desc=f"Writing {sample_size:,} items",
                    )

                # Write header
                writer.write(dataset.header)

                # Write sampled lines
                for index in sample_indexes:
                    writer.write(dataset[index])
                    count_written += 1

            logger.info(f"    L done (written: {count_written:,})")

    return 0


def _split_test(dataset: CustomDataset, CONFIG: Config) -> int:
    logger.info("  L Writing test file ...")
    logger.info(f"    L Test file: {CONFIG.split_test_file}")

    test_num_lines = int(len(dataset) * CONFIG.split_test_proportion)
    test_indexes = range(0, test_num_lines)

    if CONFIG.split_test_file.exists() and not CONFIG.split_again:
        logger.info("    L Skipped, already exists, use --split_again to overwrite")

    else:
        # Log
        logger.info(f"    L Building test from {test_num_lines:,} items")

        # Subset
        test_dataset = CustomDataset.subset(dataset, test_indexes)

        with open_file(CONFIG.split_test_file, "w") as writer:

            # Write
            writer.write(test_dataset.header)

            # Show progress
            if CONFIG.show_progress:
                test_dataset = tqdm(
                    test_dataset,
                    total=len(test_dataset),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    desc="Writing test file",
                )

            # Write
            writer.writelines([item for item in test_dataset])

    return 0


def _split_train_valid(dataset: CustomDataset, CONFIG: Config) -> int:
    logger.info("  L Writing train and valid files using 'train-valid-split' method ...")
    logger.info(f"    L Train files: {CONFIG.split_train_files}")
    logger.info(f"    L Valid files: {CONFIG.split_valid_files}")

    test_num_lines = int(len(dataset) * CONFIG.split_test_proportion)
    valid_num_lines = int(len(dataset) * CONFIG.split_valid_proportion)
    train_num_lines = len(dataset) - test_num_lines - valid_num_lines

    valid_indexes = range(test_num_lines, test_num_lines + valid_num_lines)
    train_indexes = range(test_num_lines + valid_num_lines, len(dataset))

    if any([f.exists() for f in CONFIG.split_train_files + CONFIG.split_valid_files]) and not CONFIG.split_again:  # noqa E501
        logger.info("    L Skipped, already exists, use --split_again to overwrite")

    else:
        # Log
        logger.info(f"    L Building train from {train_num_lines:,} items")

        # Save train subset
        train_dataset = CustomDataset.subset(dataset, train_indexes)
        train_file = CONFIG.split_train_files[0]
        with open_file(train_file, "w") as f:

            # Header
            f.write(train_dataset.header)

            # Show progress
            if CONFIG.show_progress:
                train_dataset = tqdm(
                    train_dataset,
                    total=len(train_dataset),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    desc="Writing train file",
                )

            # Write
            f.writelines([item for item in train_dataset])

        # Log
        logger.info(f"    L Building valid from {valid_num_lines:,} items")

        # Save valid subset
        valid_dataset = CustomDataset.subset(dataset, valid_indexes)
        valid_file = CONFIG.split_valid_files[0]
        with open_file(valid_file, "w") as f:

            # Header
            f.write(valid_dataset.header)

            # Show progress
            if CONFIG.show_progress:
                valid_dataset = tqdm(
                    valid_dataset,
                    total=len(valid_dataset),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    desc="Writing valid file",
                )

            # Write
            f.writelines([item for item in valid_dataset])

    logger.info("  L Done.")

    return 0


def _split_kfolds(dataset: CustomDataset, CONFIG: Config) -> int:
    logger.info("  L Writing train and valid files using 'k-folds' method ...")
    logger.info(f"    L Train files: {CONFIG.split_train_files}")
    logger.info(f"    L Valid files: {CONFIG.split_valid_files}")

    test_num_lines = int(len(dataset) * CONFIG.split_test_proportion)
    fold_num_lines = len(dataset) - test_num_lines

    fold_indexes = range(test_num_lines, len(dataset))

    if any([f.exists() for f in CONFIG.split_train_files + CONFIG.split_valid_files]) and not CONFIG.split_again:  # noqa E501
        logger.info("    L skipped, already exists, use --split_again to overwrite")

    else:
        # Log
        logger.info(f"    L Building folds from {fold_num_lines:,} items")

        # Subset
        fold_dataset = CustomDataset.subset(dataset, fold_indexes)

        # Split
        kfold = KFold(n_splits=CONFIG.split_num_folds, shuffle=True, random_state=CONFIG.split_seed)

        for fold_i, (train_indexes, valid_indexes) in enumerate(kfold.split(fold_dataset)):
            msg = (
                f"    L Fold {fold_i + 1:2}/{CONFIG.split_num_folds}"
                f" | Train: {len(train_indexes):,}"
                f" | Valid: {len(valid_indexes):,}"
            )
            logger.info(msg)

            # Save files
            with open_file(CONFIG.split_train_files[fold_i], "w") as f:

                # Show progress
                if CONFIG.show_progress:
                    train_indexes = tqdm(
                        train_indexes,
                        total=len(train_indexes),
                        mininterval=1,
                        leave=False,
                        unit_scale=True,
                        desc="Writing train file",
                    )

                # Write
                f.write(fold_dataset.header)
                f.writelines([fold_dataset[i] for i in train_indexes])

            with open_file(CONFIG.split_valid_files[fold_i], "w") as f:

                # Show progress
                if CONFIG.show_progress:
                    valid_indexes = tqdm(
                        valid_indexes,
                        total=len(valid_indexes),
                        mininterval=1,
                        leave=False,
                        unit_scale=True,
                        desc="Writing valid file",
                    )

                # Write
                f.write(fold_dataset.header)
                f.writelines([fold_dataset[i] for i in valid_indexes])

        logger.info("  L Done.")

    return 0


def split_dataset(CONFIG: Config) -> int:
    logger.info("Splitting dataset ...")

    # From file
    dataset = CustomDataset(CONFIG.split_from_file, has_header=True)

    # Test file
    _split_test(dataset, CONFIG)

    # Train and valid files (train_valid_split method)
    if CONFIG.split_method == "train_valid_split":
        _split_train_valid(dataset, CONFIG)

    # Train and valid files (k-folds)
    elif CONFIG.split_method == "kfold":
        _split_kfolds(dataset, CONFIG)

    else:
        msg = f"  L Unknown split method: {CONFIG.split_method}"
        logger.error(msg)
        raise ValueError(msg)

    return 0


def analyze_all_datasets(CONFIG: Config) -> int:
    logger.info("Analyzing datasets ...")

    # List files from directories
    datasets_files = [
        CONFIG.dataset_ready_file,
        *CONFIG.sample_dir.glob("*.tsv"),
        *CONFIG.split_dir.glob("*.tsv")
    ]

    # Check if some analysis have already been done
    if not CONFIG.analyze_redo and any([(CONFIG.analysis_dir / file.stem).exists() for file in datasets_files]):  # noqa E501
        msg = ("  L skipped, some datasets have already been analyzed. Use --analyze_redo to reanalyze.")  # noqa E501
        logger.info(msg)
        return 0

    # Tasks for one dataset
    def _func(file):
        out_dir = CONFIG.analysis_dir / file.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        analyze_one_dataset(file, out_dir, CONFIG, results)
        plot_one_dataset(results["distributions"], out_dir)
        return 0

    # Analyze each dataset
    for file in datasets_files:
        logger.info(f"  L Dataset: {file}")
        _func(file)

    return 0


def analyze_one_dataset(dataset_file: Path, out_dir: Path, CONFIG: Config, results: dict=None) -> int:  # noqa E501
    # Setting up parallel processing
    pandarallel.initialize(nb_workers=CONFIG.workers, verbose=1)

    # Prepare distributions
    distributions: Dict[str, List[Any]] = {
        "MW": [],  # Molecular weight
        "HV": [],  # Heavy atoms count
        "R": [],  # Rings count
        "C": [],  # CHONPS atoms count
        "H": [],
        "O": [],
        "N": [],
        "P": [],
        "S": [],
    }

    # Iterate over the dataset with chunks
    chunksize = CONFIG.analyze_chunksize
    use_cols = [CONFIG.data_col_indexes["SMILES"]]
    num_items = _get_nb_lines(dataset_file)

    with open_file(dataset_file, "r") as reader:
        df_iter = pd.read_csv(reader, sep="\t", usecols=use_cols, chunksize=chunksize, dtype=str)

        # Show progress bar
        if CONFIG.show_progress:
            df_iter = tqdm(
                df_iter,
                total=_get_nb_chunks(dataset_file, chunksize),
                mininterval=1,
                leave=False,
                unit_scale=True,
                unit=f" chunk(s) of {chunksize:,} lines",
                desc=f"Analyzing {num_items:,} items",
            )

        # Iterate over each chunk
        for chunk_idx, chunk in enumerate(df_iter, start=1):
            features_df = chunk['SMILES'].parallel_apply(_get_mol_features).apply(pd.Series)
            assert len(features_df) == len(chunk)

            # Only keep valid molecules
            features_df.dropna(subset=["MW", "HV", "R", "C", "H", "O", "N", "P", "S"], inplace=True)  # noqa E501

            num_invalids = len(chunk) - len(features_df)
            if num_invalids > 0:
                msg = f"  L Chunk {chunk_idx}: {num_invalids} invalid molecules ignored"
                logger.warning(msg)

            # Add valid molecules to distributions
            for feature in distributions:
                distributions[feature].extend(features_df[feature].astype(float).tolist())

    # Convert to DataFrame
    distributions_df = pd.DataFrame(distributions)

    # Save distributions
    tsv_file = out_dir / "distributions.tsv"
    pkl_file = out_dir / "distributions.pkl"
    distributions_df.to_csv(tsv_file, sep="\t", index=False)
    distributions_df.to_pickle(pkl_file)

    # Save summarized statistics
    summary_df = distributions_df.describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary_df.index = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    tsv_file = out_dir / "summary.tsv"
    summary_df.to_csv(tsv_file, sep="\t", index=True)

    # Log
    logger.info(f"    L Data saved in {out_dir}")

    # If results dict provided, store the results in it
    if results is not None:
        results["distributions"] = distributions_df
        results["summary"] = summary_df

    return 0


def plot_one_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # List features to plot
    features = ["MW", "HV", "R", "C", "H", "O", "N", "P", "S"]
    num_features = len(features)

    # Determine the number of columns and rows for the subplots
    cols = 3
    rows = (num_features + cols - 1) // cols  # Calculer le nombre de rangées nécessaires

    # Histogram settings
    histogram_bins = 10  # Small number of bins for discrete features
    histogram_kde_bw = 5  # Increase for smoother KDE

    # Gather distributions in a unique PNG
    plt.figure(figsize=(cols * 5, rows * 4))
    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(
            df[feature],
            kde=True,
            bins=histogram_bins,
            color='skyblue',
            kde_kws={'bw_adjust': histogram_kde_bw},
        )
        plt.title(f"{feature} distribution")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
    plt.tight_layout()
    hist_path = output_dir / "all_distributions.png"
    plt.savefig(hist_path)
    plt.close()

    # Gather boxplots in a unique PNG
    plt.figure(figsize=(cols * 5, rows * 4))
    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(y=df[feature], color='lightgreen')
        plt.title(f"{feature} boxplot")
        plt.ylabel(feature)
    plt.tight_layout()
    boxplot_path = output_dir / "all_boxplots.png"
    plt.savefig(boxplot_path)
    plt.close()

    # Build correlation matrix
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation matrix of features")
    plt.tight_layout()
    corr_path = output_dir / "correlation_matrix.png"
    plt.savefig(corr_path)
    plt.close()

    logger.info(f"    L Plots saved in {output_dir}")


def compare_all_datasets(CONFIG: Config) -> int:
    logger.info("Comparing datasets ...")

    # Explore analysis directories
    analysis_dirs = [
        path for path in CONFIG.analysis_dir.glob("*")
        if path.is_dir() and (path / "distributions.pkl").exists()
    ]
    analysis_dirs.sort()

    # Check if some comparisons have already been done
    if not CONFIG.compare_redo and CONFIG.comparative_dir.exists():  # noqa E501
        msg = "  L Skipped, some datasets have already been compared. Use --compare_redo to recompare."  # noqa E501
        logger.info(msg)
        return 0

    # Deduce dataset names from directories
    dataset_names = [path.stem for path in analysis_dirs]

    compare_datasets(analysis_dirs, output_dir=CONFIG.comparative_dir, dataset_names=dataset_names)

    return 0


def compare_datasets(analysis_dirs: List[Path], output_dir: Path, dataset_names: Optional[List[str]] = None) -> None:  # noqa E501
    """
    Compare features distributions across multiple datasets.

    Parameters
    ----------
    analysis_dirs : List[Path]
        List of paths to analysis directories containing 'distributions.pkl'.
    output_dir : Path
        Directory to save the comparative plots.
    dataset_names : Optional[List[str]], optional
        List of dataset names. If None, names will be derived from directories, by default None.

    Returns
    -------
    None
    """
    if dataset_names and len(dataset_names) != len(analysis_dirs):
        msg = ("  L Mismatch between number of dataset names and analysis directories.")
        logger.error(msg)
        raise ValueError(msg)

    logger.info("  L Loading distributions ...")

    combined_data = []
    for idx, analysis_dir in enumerate(analysis_dirs):
        try:
            distribution_path = analysis_dir / "distributions.pkl"
            if not distribution_path.exists():
                msg = f"  L {distribution_path} file not found, skipping."
                logger.warning(msg)
                continue
            df = pd.read_pickle(distribution_path)
            dataset_name = dataset_names[idx] if dataset_names else analysis_dir.stem
            df['Dataset'] = dataset_name
            combined_data.append(df)
            logger.info(f"    L Loaded distributions from '{dataset_name}'")
        except Exception as e:
            logger.error(f"    L Error loading {analysis_dir}: {e}")

    if not combined_data:
        logger.error("  L No valid datasets loaded.")
        return

    # Combine datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    logger.info(f"  L Total number of combined molecules: {len(combined_df)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  L Generating comparative distributions ...")

    # Get features to compare from the DataFrame
    # features = combined_df.columns.tolist()
    features = ["MW", "HV", "R", "C", "H", "O", "N", "P", "S"]

    # # Comparative distributions
    generate_comparative_kde_plots(combined_df, features, output_dir)

    # Comparative distributions with shift
    generate_comparative_kde_shifted_plots(combined_df, features, output_dir)

    # Comparative boxplots
    generate_comparative_boxplots(combined_df, features, output_dir)

    # Individual correlation matrices
    generate_individual_correlation_matrices(combined_df, features, output_dir)

    logger.info("  L Done.")


def generate_comparative_kde_shifted_plots(combined_df: pd.DataFrame, features: List[str], output_dir: Path) -> None:  # noqa E501
    """
    Generate and save comparative distribution plots (KDE) with shift for multiple datasets.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame containing distributions and dataset identifiers.
    features : List[str]
        List of features to visualize.
    output_dir : Path
        Directory to save the plots.

    Returns
    -------
    None
    """
    logger.info("    L Generating comparative distribution plots with shift...")

    import numpy as np
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 25))  # Adjust size according to the number of features
    cols = 3
    rows = (len(features) + cols - 1) // cols

    gaussian_kde_bw = 0.5

    datasets = combined_df["Dataset"].unique()
    palette = sns.color_palette("Set1", n_colors=len(datasets))
    dataset_to_color = dict(zip(datasets, palette))

    # Preparing handles and labels for the legend
    legend_elements = [Line2D([0], [0], color=dataset_to_color[dataset], lw=2, label=dataset) for dataset in datasets]  # noqa E501

    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        ax = plt.gca()

        for dataset_index, dataset in enumerate(datasets):
            data = combined_df[combined_df["Dataset"] == dataset][feature].dropna()
            if len(data) < 2:
                msg = f"    L Not enough data for KDE of '{feature}' in '{dataset}'."
                logger.warning(msg)
                continue
            try:
                density = gaussian_kde(data, bw_method=gaussian_kde_bw)
                x_min, x_max = data.min(), data.max()
                x = np.linspace(x_min, x_max, 1000)
                y = density(x)

                # Calculer les décalages
                shift_x = (x_max - x_min) * 0.01 * dataset_index  # 1% of the range x per index
                shift_y = y.max() * 0.05 * dataset_index          # 5% of the max y per index

                x_shifted = x + shift_x
                y_shifted = y + shift_y

                ax.plot(x_shifted, y_shifted, color=dataset_to_color[dataset], alpha=0.6)

            except Exception as e:
                msg = f"    L Error calculating KDE for '{feature}' in '{dataset}': {e}"
                logger.error(msg)

        plt.title(f"Distribution of {feature} accross datasets")
        plt.xlabel(feature)
        plt.ylabel("Density")

    # Build a global legend outside the subplots
    plt.legend(handles=legend_elements, title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    dist_comparative_path = output_dir / "comparative_distributions_kde_shifted.png"
    plt.savefig(dist_comparative_path, bbox_inches='tight')
    plt.close()

    logger.info(f"    L Comparative distribution plots with shift saved under {dist_comparative_path}")  # noqa E501


def generate_comparative_kde_plots(combined_df: pd.DataFrame, features: List[str], output_dir: Path) -> None:  # noqa E501
    """
    Generate and save comparative distribution plots (KDE) for multiple datasets.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame containing distributions and dataset identifiers.
    features : List[str]
        List of features to visualize.
    output_dir : Path
        Directory to save the plots.

    Returns
    -------
    None
    """
    logger.info("    L Generating comparative distribution plots...")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 25))  # Adjust size according to the number of features
    cols = 3
    rows = (len(features) + cols - 1) // cols

    histogram_kde_bw = 3  # Adjust for smoother KDE

    # Variables to collect handles and labels for the legend
    handles = []
    labels = []
    legend_added = False

    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.kdeplot(
            data=combined_df,
            x=feature,
            hue="Dataset",
            multiple="layer",  # layer or stack for multiple distributions
            bw_adjust=histogram_kde_bw,
            palette="Set1",
            fill=False,
            common_norm=False,
            alpha=0.6
        )
        plt.title(f"Distribution of {feature} accross datasets")
        plt.xlabel(feature)
        plt.ylabel("Density")

        # Collect handles and labels for the legend
        if not legend_added:
            handles, labels = plt.gca().get_legend_handles_labels()
            legend_added = True

    # Build a global legend outside the subplots
    plt.legend(handles, labels, title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    dist_comparative_path = output_dir / "comparative_distributions.png"
    plt.savefig(dist_comparative_path, bbox_inches='tight')
    plt.close()

    logger.info(f"    L Comparative distribution plots saved under {dist_comparative_path}")


def generate_comparative_boxplots(combined_df: pd.DataFrame, features: List[str], output_dir: Path) -> None:  # noqa E501
    """

    Generate and save comparative boxplots for multiple datasets.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame containing distributions and dataset identifiers.
    features : List[str]
        List of features to visualize.
    output_dir : Path
        Directory to save the plots.
    """
    logger.info("    L Generating comparative boxplots...")

    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 25))  # Adjust size according to the number of features
    cols = 3
    rows = (len(features) + cols - 1) // cols

    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(
            data=combined_df,
            x="Dataset",
            y=feature,
            palette="Set1",
            hue="Dataset",
            legend=False,
        )
        plt.title(f"Boxplot of {feature} accross datasets")
        plt.xlabel("Dataset")
        plt.ylabel(feature)

    plt.tight_layout()
    boxplot_comparative_path = output_dir / "comparative_boxplots.png"
    plt.savefig(boxplot_comparative_path)
    plt.close()

    logger.info(f"    L Comparative boxplots saved under {boxplot_comparative_path}")


def generate_individual_correlation_matrices(combined_df: pd.DataFrame, features: List[str], output_dir: Path) -> None:  # noqa E501
    """
    Generate and save individual correlation matrices for each dataset.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame containing distributions and dataset identifiers.
    features : List[str]
        List of features to visualize.
    output_dir : Path
        Directory to save the plots.
    """
    logger.info("    L Generating individual correlation matrices...")

    datasets = combined_df["Dataset"].unique()
    for dataset in datasets:
        plt.figure(figsize=(10, 8))
        subset_df = combined_df[combined_df["Dataset"] == dataset]
        if subset_df.shape[0] < 2:
            msg = f"    L Not enough data to generate a correlation matrix for '{dataset}'."
            logger.warning(msg)
            plt.close()
            continue
        correlation = subset_df[features].corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation matrix of features - {dataset}")
        plt.tight_layout()
        # Remplacer les espaces par des underscores pour les noms de fichiers
        safe_dataset_name = dataset.replace(' ', '_')
        corr_path = output_dir / f"correlation_matrix_{safe_dataset_name}.png"
        plt.savefig(corr_path)
        plt.close()

        logger.info(f"    L Correlation matrix for '{dataset}' saved under {corr_path}")


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
setup_logger(logger, logging.INFO)

# Disable RDKIT warnings
RDLogger.DisableLog("rdApp.*")


# Arg parsing -------------------------------------------------------------------------------------
def parse_args():
    # Get default config
    CONFIG = Config(db="metanetx", mode="prepare")

    parser = argparse.ArgumentParser(prog=Path(__file__).name, description="Prepare dataset")
    parser.add_argument(
        "actions",
        nargs="+",
        choices=["all", "download", "reshape", "filter", "fingerprints", "dedupe", "sample", "split", "analyze", "compare"],  # noqa E501
        default=["all"],
        help="Actions to perform (default: %(default)s)",
    )

    parser_general = parser.add_argument_group("General options")
    parser_general.add_argument(
        "--db",
        metavar="DB",
        type=str,
        default=CONFIG.db,
        help="Database to use (default: %(default)s)",
    )
    parser_general.add_argument(
        "--show_progress",
        action="store_true",
        help="Show progress bar (default: %(default)s)",
    )
    parser_general.add_argument(
        "--workers",
        metavar="WORKERS",
        type=int,
        default=os.cpu_count() // 2,
        help="Number of workers (default: %(default)s)",
    )

    parser_download = parser.add_argument_group("Download options")
    parser_download.add_argument(
        "--download_again",
        action="store_true",
        help="Download again database (default: %(default)s)",
    )

    parser_reshape = parser.add_argument_group("Reshape options")
    parser_reshape.add_argument(
        "--reshape_again",
        action="store_true",
        help="Reshape again (default: %(default)s)",
    )

    parser_filter = parser.add_argument_group("Filter options")
    parser_filter.add_argument(
        "--filter_again",
        action="store_true",
        help="Filter again (default: %(default)s)",
    )
    parser_filter.add_argument(
        "--filter_max_mw",
        metavar="MW",
        type=float,
        default=500,
        help="Maximum molecular weight to filter (default: %(default)s)",
    )

    parser_fingerprints = parser.add_argument_group("Fingerprints options")
    parser_fingerprints.add_argument(
        "--fingerprints_redo",
        action="store_true",
        help="Recompute fingerprints (default: %(default)s)",
    )
    parser_fingerprints.add_argument(
        "--fingerprints_resume",
        action="store_true",
        help="Resume computation of fingerprints (default: %(default)s)",
    )

    parser_dedupe = parser.add_argument_group("Deduplicate options")
    parser_dedupe.add_argument(
        "--dedupe_again",
        action="store_true",
        help="Deduplicate again (default: %(default)s)",
    )
    parser_dedupe.add_argument(
        "--dedupe_from_db",
        metavar="EXTERNAL DB",
        type=str,
        default="metanetx",
        help="Deduplicate from external dataset (only used when main DB is emolcules, default: %(default)s)",  # noqa E501
    )
    parser_dedupe.add_argument(
        "--dedupe_from_db_again",
        action="store_true",
        help="Deduplicate from external dataset again (default: %(default)s)",
    )

    parser_sample = parser.add_argument_group("Sample options")
    parser_sample.add_argument(
        "--sample_again",
        action="store_true",
        help="Sample again (default: %(default)s)",
    )
    parser_sample.add_argument(
        "--sample_sizes",
        metavar="SIZE",
        nargs="+",
        type=int,
        default=CONFIG.sample_sizes,
        help="Size of samples (default: %(default)s)",
    )
    parser_sample.add_argument(
        "--sample_seed",
        metavar="SEED",
        type=int,
        default=CONFIG.sample_seed,
        help="Random seed for sampling (default: %(default)s)",
    )

    parser_split = parser.add_argument_group("Split options")
    parser_split.add_argument(
        "--split_test_proportion",
        metavar="PROPORTION",
        type=float,
        default=CONFIG.split_test_proportion,
        help="Proportion of test set (default: %(default)s)",
    )
    parser_split.add_argument(
        "--split_valid_proportion",
        metavar="PROPORTION",
        type=float,
        default=CONFIG.split_valid_proportion,
        help="Proportion of validation set (only for 'train_valid_split' method) (default: %(default)s)",  # noqa E501
    )
    parser_split.add_argument(
        "--split_method",
        metavar="METHOD",
        choices=["train_valid_split", "kfold"],
        default=CONFIG.split_method,
        help="Split method (default: %(default)s; choices: %(choices)s)",
    )
    parser_split.add_argument(
        "--split_num_folds",
        metavar="FOLDS",
        type=int,
        default=CONFIG.split_num_folds,
        help="Number of folds (only for 'kfold' method) (default: %(default)s)",
    )
    parser_split.add_argument(
        "--split_seed",
        metavar="SEED",
        type=int,
        default=CONFIG.split_seed,
        help="Random seed for splitting (default: %(default)s)",
    )
    parser_split.add_argument(
        "--split_again",
        action="store_true",
        help="Split again (default: %(default)s)",
    )

    parser_analyze = parser.add_argument_group("Analyze options")
    parser_analyze.add_argument(
        "--analyze_redo",
        action="store_true",
        help="Redo analysis (default: %(default)s)",
    )
    parser_analyze.add_argument(
        "--analyze_chunksize",
        metavar="CHUNKSIZE",
        type=int,
        default=10000,
        help="Chunksize for analysis (default: %(default)s)",
    )

    parser_compare = parser.add_argument_group("Compare options")
    parser_compare.add_argument(
        "--compare_redo",
        action="store_true",
        help="Redo comparisons (default: %(default)s)",
    )

    # Trick to use the command pickargs from VSCODE for debugging
    if os.getenv("USED_VSCODE_COMMAND_PICKARGS", "0") == "1":
        parser.set_defaults(db="emolecules")
        parser.set_defaults(descriptors_resume=True)
        parser.set_defaults(show_progress=True)

    args = parser.parse_args()

    # Update config according to CLI args
    CONFIG = CONFIG.with_db(args.db).with_split_method(args.split_method)
    for key, value in vars(args).items():
        if value is not None:
            setattr(CONFIG, key, value)

    log_config(CONFIG, logger)

    return CONFIG


# Main --------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse arguments and load configuration
    CONFIG = parse_args()

    # Init directories
    CONFIG.download_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.dataset_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.sample_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.split_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.analysis_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    if CONFIG.actions == ["all"] or "download" in CONFIG.actions:
        download_dataset(CONFIG)

    # Reshape dataset
    if CONFIG.actions == ["all"] or "reshape" in CONFIG.actions:
        reshape_dataset(CONFIG)

    # Filter dataset
    if CONFIG.actions == ["all"] or "filter" in CONFIG.actions:
        filter_dataset(CONFIG)

    # Compute descriptors
    if CONFIG.actions == ["all"] or "fingerprints" in CONFIG.actions:
        compute_fingerprints(CONFIG)

    # Dedupe dataset
    if CONFIG.actions == ["all"] or "dedupe" in CONFIG.actions:
        dedupe_dataset(CONFIG)
        if CONFIG.db == "emolecules":
            CONFIG_EXT = Config(db=CONFIG.dedupe_from_db, mode="prepare")
            dedupe_from_external_dataset(CONFIG, CONFIG_EXT)
        else:
            logger.info("Deduplicating dataset from external dataset ...")
            logger.warning("  L Only available with 'emolecules' as main database, skipped")

    # Sample dataset
    if CONFIG.actions == ["all"] or "sample" in CONFIG.actions:
        sample_dataset(CONFIG)

    # Split dataset for training
    if CONFIG.actions == ["all"] or "split" in CONFIG.actions:
        split_dataset(CONFIG)

    # Analyze datasets
    if CONFIG.actions == ["all"] or "analyze" in CONFIG.actions:
        analyze_all_datasets(CONFIG)

    # Compare datasets
    if CONFIG.actions == ["all"] or "compare" in CONFIG.actions:
        compare_all_datasets(CONFIG)

    logger.info("All done.")
