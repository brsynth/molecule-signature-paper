#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
from pandarallel import pandarallel
from pathlib import Path
import random
from tqdm import tqdm
from types import SimpleNamespace

from rdkit import RDLogger
from rdkit.Chem import MolToSmiles
import coloredlogs

from signature.utils import mol_from_smiles, mol_filter
from retrosig.utils import cmd
from paper.dataset.utils import log_config, open_file, get_signature, get_ecfp


# Settings ----------------------------------------------------------------------------------------
_mnx = SimpleNamespace()  # metanetx
_mnx.url = "https://www.metanetx.org/ftp/4.4/chem_prop.tsv"
_mnx.dir_base = Path(".") / "data" / "metanetx"
_mnx.dir_download = _mnx.dir_base / "download"
_mnx.dir_dataset = _mnx.dir_base / "dataset"
_mnx.file_db_raw = _mnx.dir_download / "mnx_raw_4_4.tsv"
_mnx.file_db_reshaped = _mnx.dir_dataset / "db_reshape.tsv.gz"
_mnx.file_db_filtered = _mnx.dir_dataset / "db_filtered.tsv.gz"
_mnx.file_db_sampled = _mnx.dir_dataset / "db_sampled_{size}.tsv.gz"
_mnx.file_db_descriptors = _mnx.dir_dataset / "db_descriptors.tsv"

_emol = SimpleNamespace()  # emolecules
_emol.url = "https://downloads.emolecules.com/free/2024-07-01//version.smi.gz"
_emol.dir_base = Path(".") / "data" / "emolecules"
_emol.dir_download = _emol.dir_base / "download"
_emol.dir_dataset = _emol.dir_base / "dataset"
_emol.file_db_raw = _emol.dir_download / "emol_raw_2024-07-01.tsv.gz"
_emol.file_db_reshaped = _emol.dir_dataset / "db_reshaped.tsv.gz"
_emol.file_db_filtered = _emol.dir_dataset / "db_filtered.tsv.gz"
_emol.file_db_sampled = _emol.dir_dataset / "db_sampled_{size}.tsv.gz"
_emol.file_db_descriptors = _emol.dir_dataset / "db_descriptors.tsv.gz"


ALL_CONFIGS = {
    "metanetx": _mnx,
    "emolecules": _emol,
}


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


def _get_descriptors(smiles):
    mol = mol_from_smiles(smiles)
    smiles = MolToSmiles(mol)
    signature = get_signature(mol, radius=2, nbits=2048)
    ecfp = get_ecfp(mol, radius=2, nbits=2048)
    return smiles, signature, ecfp


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add TQDM logging handler
coloredlogs.install(
    level='DEBUG',
    logger=logger,
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


RDLogger.DisableLog("rdApp.*")  # Disable RDKIT warnings


# Main --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Download dataset",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        type=str,
        default="metanetx",
        choices=ALL_CONFIGS.keys(),
        help="Dataset to use (default: %(default)s; choices: %(choices)s)",
    )
    parser.add_argument(
        "--download_again",
        action="store_true",
        help="Download the dataset again if it already exists",
    )
    parser.add_argument(
        "--reshape_again",
        action="store_true",
        help="Reshape the dataset again if it already exists",
    )
    parser.add_argument(
        "--filter_again",
        action="store_true",
        help="Filter the dataset again if it already exists",
    )
    parser.add_argument(
        "--filter_max_mw",
        metavar="MW",
        type=float,
        default=500,
        help="Maximum molecular weight to filter (default: %(default)s)",
    )
    parser.add_argument(
        "--sample_again",
        action="store_true",
        help="Sample the dataset again if it already exists",
    )
    parser.add_argument(
        "--sample_sizes",
        metavar="SIZE",
        type=int,
        nargs="+",
        default=[1000, 10000, 50000],
        help="Number of lines to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--sample_seed",
        metavar="SEED",
        type=int,
        default=42,
        help="Random seed for sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--descriptors_again",
        action="store_true",
        help="Generate again signature and ECFP if it already exists",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show progress bar",
    )
    # Add argument for setting the number of workers to run in parallel
    parser.add_argument(
        "--workers",
        metavar="WORKERS",
        type=int,
        default=os.cpu_count() // 2,
        help="Number of workers to run in parallel (default: %(default)s)",
    )
    args = parser.parse_args()

    # Settings ----------------------------------
    CONFIG = ALL_CONFIGS[args.dataset]
    for key, value in vars(args).items():
        setattr(CONFIG, key, value)
    log_config(CONFIG, logger=logger)

    # Init directories --------------------------
    CONFIG.dir_download.mkdir(parents=True, exist_ok=True)
    CONFIG.dir_dataset.mkdir(parents=True, exist_ok=True)

    # Download dataset --------------------------
    logger.info("Downloading dataset...")
    if CONFIG.file_db_raw.exists() and not CONFIG.download_again:
        logger.info("  L skipped, already exists, use --download_again to "
                    "download again")
    else:
        cmd.url_download(url=CONFIG.url, path=CONFIG.file_db_raw)
        logger.info("  L done")

    # Reshape dataset ---------------------------
    logger.info("Reshaping dataset...")
    if CONFIG.file_db_reshaped.exists() and not CONFIG.reshape_again:
        logger.info("  L skipped, already exists, use --reshape_again to "
                    "reshape again")
    else:
        cnt_written = 0

        with open_file(
            CONFIG.file_db_raw, "r"
        ) as reader, open_file(
            CONFIG.file_db_reshaped, "w"
        ) as writer:
            if CONFIG.show_progress:  # Show progress bar
                reader = tqdm(
                    reader,
                    total=_get_nb_lines(CONFIG.file_db_raw),
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
                    cnt_written += 1

        logger.info(f"  L done (written: {cnt_written:,})")

    # Filter dataset ----------------------------
    logger.info("Filtering dataset...")
    if CONFIG.file_db_filtered.exists() and not CONFIG.filter_again:
        logger.info("  L skipped, already exists, use --filter_again to "
                    "filter again")
    else:
        cnt_written = 0
        cnt_skipped = 0

        with open_file(
            CONFIG.file_db_reshaped, "r"
        ) as reader, open_file(
            CONFIG.file_db_filtered, "w"
        ) as writer:
            if CONFIG.show_progress:  # Show progress bar
                reader = tqdm(
                    reader,
                    total=_get_nb_lines(CONFIG.file_db_reshaped),
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
                    cnt_skipped += 1
                    continue
                else:
                    _id, smiles = _
                try:
                    mol = mol_from_smiles(smiles)
                    if mol is None:
                        cnt_skipped += 1
                        continue
                    mol = mol_filter(mol, max_mw=CONFIG.filter_max_mw)
                    if mol is None:
                        cnt_skipped += 1
                        continue
                    writer.write(line)
                    cnt_written += 1
                except Exception as err:
                    logger.error(f"  L {index, _id, err}")
                    cnt_skipped += 1

        logger.info(f"  L done (written: {cnt_written:,}, skipped: {cnt_skipped:,})")

    # Compute descriptors -----------------------
    logger.info("Computing descriptors...")
    if CONFIG.file_db_descriptors.exists() and not CONFIG.descriptors_again:
        logger.info("  L skipped, already exists, use --descriptor_again to "
                    "compute again descriptors")
    else:
        cnt_written = 0

        CHUNKSIZE = 10000

        # Setting up parallel processing
        pandarallel.initialize(nb_workers=CONFIG.workers, verbose=1)

        with open_file(
            CONFIG.file_db_filtered, "r"
        ) as reader, open_file(
            CONFIG.file_db_descriptors, "w"
        ) as writer:

            reader = pd.read_csv(reader, sep="\t", iterator=True, chunksize=CHUNKSIZE)
            writer.write("ID\tSMILES_0\tSMILES\tSignature\tECFP\n")  # Header

            if CONFIG.show_progress:
                reader = tqdm(
                    reader,
                    total=_get_nb_chunks(CONFIG.file_db_filtered, chunksize=CHUNKSIZE),
                    mininterval=1,
                    leave=False,
                    unit_scale=True,
                    unit=f" chunk(s) of {CHUNKSIZE:,} lines",
                )

            for df_chunk in reader:
                df_chunk.rename(columns={"SMILES": "SMILES_0"}, inplace=True)
                # Rename the SMILES column to SMILES_0 to avoid the error of having the same column
                # name when parallelizing the apply function
                df_chunk[["SMILES", "Signature", "ECFP"]] = df_chunk.parallel_apply(
                    lambda x: _get_descriptors(x["SMILES_0"]), axis=1, result_type="expand"
                )
                df_chunk.to_csv(writer, sep="\t", index=False, header=False)  # Append chunk to file
                cnt_written += len(df_chunk)

        logger.info(f"  L done (written: {cnt_written:,})")

    # Sample dataset ----------------------------
    logger.info("Sampling dataset...")
    for size in CONFIG.sample_sizes:
        _sample_file = str(CONFIG.file_db_sampled)
        _sample_file = _sample_file.format(size=f"{size/1000:.0f}k")
        _sample_file = Path(_sample_file)
        logger.info(f"  L Sampling {size:,} lines into {_sample_file}...")

        if _sample_file.exists() and not CONFIG.sample_again:
            logger.info("    L skipped, already exists, use --sample_again to "
                        "sample again")
        else:
            cnt_written = 0
            nb_lines = _get_nb_lines(CONFIG.file_db_filtered)

            random.seed(CONFIG.sample_seed)
            sample_indexes = random.sample(range(1, nb_lines), size)  # 1 to skip header

            sampled_lines = []
            _indexes = []

            with open_file(CONFIG.file_db_filtered, "r") as reader:

                if CONFIG.show_progress:
                    reader = tqdm(
                        reader,
                        total=nb_lines,
                        mininterval=1,
                        leave=False,
                        desc="Collecting sampled lines",
                        unit_scale=True,
                    )

                for index, line in enumerate(reader):
                    if index in sample_indexes:
                        sampled_lines.append(line)
                        _indexes.append(index)  # store the index of the sampled row

            # Reorder the sampled lines to match the order of sampled indexes
            _indexes = [_indexes.index(idx) for idx in sample_indexes]
            sampled_lines = [sampled_lines[_] for _ in _indexes]

            # Write sampled lines
            with open_file(_sample_file, "w") as writer:

                if CONFIG.show_progress:
                    sampled_lines = tqdm(
                        sampled_lines,
                        total=len(sampled_lines),
                        mininterval=1,
                        leave=False,
                        unit_scale=True,
                        desc=f"Writing sampled {size:,} lines",
                    )

                writer.write("ID\tSMILES\n")  # Header

                for line in sampled_lines:
                    writer.write(line)
                    cnt_written += 1

            logger.info(f"    L done (written: {cnt_written:,})")
