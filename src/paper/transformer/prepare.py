#!/usr/bin/env python
import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import yaml

import coloredlogs
import sentencepiece as spm

from paper.dataset.utils import log_config, open_file


# Utilities functions -----------------------------------------------------------------------------
def dict_to_simplenamespace(d: dict) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_simplenamespace(v)
        else:
            d[k] = v
    return SimpleNamespace(**d)


def tokenize(
        input_file: str | Path,
        descriptor: str,
        col_idx: int,
        vocab_size: int,
        algorithm: str = "bpe",
        out_dir: str | Path = Path(".")
) -> None:

    # Switch to absolute paths
    input_file = Path(input_file).resolve()
    out_dir = Path(out_dir).resolve()

    # Remember the current directory
    cwd = Path.cwd()

    # Create and swith to output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)

    # Temporary file
    tmp_file = "tokenize.tmp"

    with open_file(input_file, 'r') as file, open_file(tmp_file, 'w') as tmp:
        next(file)  # Skip the header
        for row in file:
            desc = row.split('\t')[col_idx]
            desc = desc.replace(' ', '')  # Remove spaces
            tmp.write(desc + '\n')

    # Train the SentencePiece model
    spm.SentencePieceTrainer.Train(
        input=tmp_file,
        model_prefix=descriptor,
        vocab_size=vocab_size,
        model_type=algorithm,
        character_coverage=1.0,
        hard_vocab_limit=True,  # Avoid error about vocab size: https://github.com/google/sentencepiece/issues/605  # noqa E501
        pad_id=3,
        # use_all_vocab=_use_all_vocab,  # important to avoid error about vocab size
    )

    # Remove temporary file
    os.remove(tmp_file)

    # Switch back to the original directory
    os.chdir(cwd)


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

# Main --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Prepare datasets to be used with DL models"
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default=Path(__file__).parent / "config.yaml",
        help="Configuration file (default: %(default)s)"
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        type=str,
        default="metanetx",
        choices=("metanetx", "emolecules"),
        help="Dataset to tokenize (default: %(default)s; choices: %(choices)s)"
    )
    args = parser.parse_args()

    # Load configuration ------------------------
    with open(args.config, "r") as file:
        CONFIG = dict_to_simplenamespace(yaml.safe_load(file))
    log_config(CONFIG, logger=logger)

    # Tokenize ----------------------------------
    logger.info("Tokenizing molecules...")

    # Source file
    base_path = vars(CONFIG.data.base_path)[args.dataset]
    input_file = Path(base_path) / CONFIG.data.dataset_file
    out_dir = Path(base_path) / CONFIG.data.spm_dir

    # Iterate over the 3 descriptors
    for descriptor in ["SMILES", "SIGNATURE", "ECFP"]:

        token_conf = getattr(CONFIG.tokenizer, descriptor)
        logger.debug(f"  L Tokenizing {descriptor} from {input_file} with {token_conf.vocab_size} tokens using {token_conf.algorithm}")

        tokenize(
            input_file=input_file,
            descriptor=descriptor,
            col_idx=token_conf.col_idx,
            vocab_size=token_conf.vocab_size,
            algorithm=token_conf.algorithm,
            out_dir=out_dir,
        )

    logger.info("  L Done.")
