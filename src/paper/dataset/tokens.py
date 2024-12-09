#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
import sys
from contextlib import contextmanager
from pathlib import Path
from collections import Counter
from typing import List, Optional

import sentencepiece as spm

from paper.learning.config import Config
from paper.dataset.utils import (
    setup_logger,
    log_config,
    open_file,
    pretokenizer_for_smiles,
    pretokenizer_for_signature,
    pretokenizer_for_ecfp,
)


def len_fingerprint(fp: str, sep: str = " ") -> int:
    """Count the number of tokens in a fingerprint."""
    return len(fp.split(sep))


def nb_fingerprints(file_path: Path) -> int:
    """Count the number of fingerprints in a file."""
    with open(file_path, "r") as file:
        return sum(1 for _ in file)


def count_tokens(file_path: Path, sep: str = " ") -> Counter:
    """Count the number of tokens in a pre-tokenized file."""
    counter = Counter()
    with open(file_path, "r") as file:
        for line in file:
            counter.update(line.strip().split(sep))

    return counter


def count_fp_by_tokens(file_path: Path, sep: str = " ") -> Counter:
    """Count the number of fingerprints in which each token appears."""
    counter = Counter()
    with open(file_path, "r") as file:
        for line in file:
            tokens = set(line.strip().split(sep))
            counter.update(tokens)

    return counter


def tokenize_all_fingerprints(CONFIG: Config) -> None:
    """Tokenize a dataset containing one or more fingerprints."""
    logger = logging.getLogger(__name__)
    logger.info("Tokenizing dataset...")

    # IOs
    data_file = CONFIG.token_from_file
    output_dir = CONFIG.token_dir

    # General parameters
    algorithm = CONFIG.token_method
    min_presence = CONFIG.token_min_presence

    # Pretoknizers
    pretokenizers = {
        "SMILES": pretokenizer_for_smiles,
        "SIGNATURE": pretokenizer_for_signature,
        "ECFP": pretokenizer_for_ecfp,
    }

    # for fingerprint in ["SMILES"]:  # DEBUG
    for fingerprint in CONFIG.token_fingerprints_types:
        logger.info(f"  L Working on {fingerprint} from '{data_file}'")

        # Working file
        tmp_file = CONFIG.token_dir / f"{fingerprint}.tmp"

        # Get fingerprint-specific parameters
        col_idx = CONFIG.data_col_indexes[fingerprint]
        vocab_size = CONFIG.token_vocabulary_sizes[fingerprint]

        # Get the pretokenizer
        if algorithm == "word":
            _pretok_func = pretokenizers.get(fingerprint, None)
            if _pretok_func is None:
                raise ValueError(f"No pretokenizer for '{fingerprint}'")
            logger.info(f"    L Using pretokenizer: '{_pretok_func.__name__}'")

        else:
            def _pretok_func(fp):
                return fp.replace(" ", "")

            logger.warning("    L Default pretokenization: spaces will be removed.")

        # Pre-tokenize
        with open_file(data_file, "r") as reader, open_file(tmp_file, "w") as writer:
            next(reader)  # Skip header
            for line in reader:
                fp = line.rstrip().split("\t")[col_idx]
                fp = _pretok_func(fp)
                if fp is not None:
                    writer.write(fp + "\n")

        # Word algorithm: refine the vocabulary size
        if algorithm == "word" and vocab_size == 0:
            # Count tokens
            tokens_count: Counter = count_tokens(tmp_file)  # Count tokens
            total_tokens = tokens_count.total()  # Total tokens

            # Count fingerprints by tokens
            fingerprint_by_tokens_count: Counter = count_fp_by_tokens(tmp_file)
            total_fps = nb_fingerprints(tmp_file)  # Total fingerprints

            # Get tokens statitics
            def _stats_func(token, token_count, total_tokens, fingerprint_by_tokens_count, total_fps) -> List:  # noqa E501
                return (
                    token,
                    token_count,
                    token_count / total_tokens,
                    fingerprint_by_tokens_count[token],
                    fingerprint_by_tokens_count[token] / total_fps,
                )

            # Build the dataframe
            df_stats = pd.DataFrame(
                data=[
                    _stats_func(token, count, total_tokens, fingerprint_by_tokens_count, total_fps)
                    for token, count in tokens_count.items()
                ],
                columns=[
                    "token",
                    "token_count",
                    "token_freq",
                    "fingerprint_count",
                    "fingerprint_freq",
                ],
            )

            # Cumulative token frequency
            df_stats = df_stats.sort_values(["token_freq", "token"], ascending=[False, True])
            df_stats["token_cum_freq"] = df_stats["token_freq"].cumsum()
            df_stats = df_stats[[  # Reorder columns
                "token",
                "token_count",
                "token_freq",
                "token_cum_freq",
                "fingerprint_count",
                "fingerprint_freq"
            ]]

            # Filter tokens using min presence threshold
            df_subset = df_stats[df_stats["fingerprint_freq"] >= min_presence]

            # Check for remaining tokens
            if len(df_stats) == 0:
                raise ValueError("No token left after filtering")

            # Update values
            tokens_subset = set(df_subset["token"])
            vocab_size = len(tokens_subset) + 4  # Add 4 for special tokens
            df_subset = df_subset.sort_values(["token_freq", "token"], ascending=[False, True])
            df_subset["token_cum_freq"] = df_subset["token_freq"].cumsum()
            vocab_cover = df_subset["token_cum_freq"].iloc[-1]

            # Check which proportion of fingerprint is covered by the refined vocabulary
            fp_covered = 0
            fp_total = 0
            with open_file(tmp_file, "r") as reader:
                for line in reader:
                    fp_total += 1
                    tokens = set(line.strip().split())
                    if tokens.issubset(tokens_subset):
                        fp_covered += 1

            # Update the all tokens stats by flagging the tokens that are kept
            df_stats["kept"] = df_stats["token"].apply(lambda x: "*" if x in tokens_subset else " ")  # noqa E501s
            df_stats = df_stats[[  # Reorder columns
                "kept",
                "token",
                "token_count",
                "token_freq",
                "token_cum_freq",
                "fingerprint_count",
                "fingerprint_freq"
            ]]

            # Export
            df_stats.to_csv(tmp_file.with_suffix(".stats"), sep="\t", index=False, float_format="%.2f")  # noqa E501

            # Compute few more statistics
            #   - number of tokens per fingerprint
            #   - min, max, median, mean, quartiles, ...
            len_fps = []
            with open_file(tmp_file, "r") as reader:
                for fp in reader:
                    len_fps.append(len_fingerprint(fp))

            # FP length stats as data frame
            df_len_fps = pd.DataFrame(len_fps, columns=["len"])

            # Append min, max, median, mean, quartiles, ...
            df_len_fps_stats = df_len_fps.describe(percentiles=[.01, .05, .10, .25, .5, .75, .90, .95, .99])  # noqa E501

            # Log
            logger.debug("    L Tokens per fingerprint:")
            for key, value in df_len_fps_stats.to_dict()["len"].items():  # noqa E501
                logger.debug(f"      L {key:<5}: {value:>10.2f}")

            # Write those stats
            df_len_fps_stats.to_csv(tmp_file.with_suffix(".stats_fp"), sep="\t", float_format="%.6f")  # noqa E501

            # Log
            logger.info(f"    L Total fingerprints       : {total_fps:>10,}")
            logger.info(f"    L Total tokens             : {total_tokens:>10,}")
            logger.info(f"    L Total unique tokens      : {len(tokens_count):>10,}")
            logger.info(f"    L Minimal token presence   : {min_presence:>10.5f}")
            logger.info("    L Vocabulary refinement    :")
            logger.info(f"      L Refined vocab. size    : {vocab_size:>10,}")
            logger.info(f"      L Refined vocab. coverage: {vocab_cover:>10.5f}")
            logger.info(f"      L Refined fp coverage    : {fp_covered / fp_total:>10.5f}")
            logger.debug("    L Tokens statistics:")
            logger.debug(f"    L {'kept':<4} {'token':<10} {'token_count':<10} {'token_freq':<10} {'token_cum_freq':<10} {'fp_count':<10} {'fp_freq':<10}")  # noqa E501
            for _, row in df_stats.iterrows():
                logger.debug(f"      {row['kept']:<4} {row['token']:<10} {row['token_count']:>10,d} {row['token_freq']:>10.5f} {row['token_cum_freq']:>10.5f} {row['fingerprint_count']:>10,d} {row['fingerprint_freq']:>10.5f}")  # noqa E501

            # Cheat sentencepiece with a fake input file: we force sentencepiece
            # to use the tokens we want.
            with open_file(tmp_file, "w") as writer:
                for token in tokens_subset:
                    writer.write(token + "\n")

        else:
            tokens_subset = None

        # Tokenize
        logger.info(f"    L Building tokenizer with {vocab_size} ({vocab_size-4} + 4) tokens using '{algorithm}' algorithm")  # noqa E501
        sentencepiece_call(
            input_file=tmp_file,
            fingerprint=fingerprint,
            vocab_size=vocab_size,
            algorithm=algorithm,
            output_dir=output_dir,
            user_defined_symbols=",".join(sorted(list(tokens_subset))) if tokens_subset is not None else "",  # noqa E501
        )

        # Clean up
        tmp_file.unlink()

    logger.info("  L Done.")


def sentencepiece_call(
    input_file: str | Path,
    fingerprint: str,
    vocab_size: int,
    algorithm: str = "bpe",
    output_dir: str | Path = Path("."),
    user_defined_symbols: Optional[str] = None,
) -> None:

    # Switch to absolute paths
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    # Remember the current directory
    original_cwd = Path.cwd()

    try:
        # Create the output directory if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Change to the output directory
        os.chdir(output_dir)

        # Define the path to the log file
        log_file_path = output_dir / f"{fingerprint}.log"

        # Redirect stdout and stderr to the log file
        with redirect_output(log_file_path):
            try:
                # Train SentencePiece model
                spm.SentencePieceTrainer.Train(
                    input=input_file,
                    model_prefix=fingerprint,
                    vocab_size=vocab_size,
                    model_type=algorithm,
                    pad_id=3,
                    max_sentence_length=6000,  # signature can be very long
                    hard_vocab_limit=False,  # Avoid error about vocab size: https://github.com/google/sentencepiece/issues/605  # noqa E501
                    normalization_rule_name="identity",  # Do not normalize the input
                    # user_defined_symbols=user_defined_symbols,
                    # add_dummy_prefix=False,  # Do not add dummy prefix
                    # use_all_vocab=True,  # Important to avoid error about vocab size
                )
            except Exception as err:
                print(f"Error tokenizing '{fingerprint}': {err}")
                raise

    finally:
        # Switch back to the original directory
        os.chdir(original_cwd)


@contextmanager
def redirect_output(log_file_path: Path):
    """
    Redirects stdout and stderr to a specified log file.

    Parameters
    ----------
    log_file_path : Path
        Path to the log file.
    """
    # Open the log file in write mode
    with open(log_file_path, 'w') as log_file:
        # Flush any existing output
        sys.stdout.flush()
        sys.stderr.flush()

        # Duplicate the original stdout and stderr file descriptors
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            # Redirect stdout and stderr to the log file
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
        finally:
            # Restore the original stdout and stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)

            # Close the duplicated file descriptors
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# Args --------------------------------------------------------------------------------------------
def parse_args():
    # Get default config
    CONFIG = Config(db="metanetx", mode="token")

    parser = argparse.ArgumentParser(prog=Path(__file__).name, description="Tokenize dataset")
    parser.add_argument(
        "--db",
        metavar="DB",
        type=str,
        default=CONFIG.db,
        choices=("metanetx", "emolecules"),
        help="Dataset to tokenize (default: %(default)s; choices: %(choices)s)",
    )
    parser.add_argument(
        "--token_fingerprints_types",
        metavar="FINGERPRINT",
        type=str,
        nargs="+",
        default=CONFIG.token_fingerprints_types,
        help="Fingerprints to tokenize (default: %(default)s)",
    )
    parser.add_argument(
        "--token_vocab_sizes",
        metavar="SIZE",
        type=int,
        nargs="+",
        default=CONFIG.token_vocabulary_sizes,
        help="Vocabulary size for each fingerprint (default: %(default)s)",
    )
    parser.add_argument(
        "--token_min_presence",
        metavar="PRESENCE",
        type=float,
        default=CONFIG.token_min_presence,
        help="Minimal presence of a token in the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--verbosity",
        metavar="LEVEL",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Verbosity level (default: %(default)s)",
    )

    # Trick to use the command pickargs from VSCODE for debugging
    if os.getenv("USED_VSCODE_COMMAND_PICKARGS", "0") == "1":
        parser.set_defaults(db="metanetx")
        parser.set_defaults(token_min_presence=0.0001)
        parser.set_defaults(verbosity="DEBUG")

    args = parser.parse_args()

    # Set up logger with right logger level
    setup_logger(logger, args.verbosity)

    # Update config according to CLI args
    CONFIG = CONFIG.with_db(args.db)
    for key, value in vars(args).items():
        if value is not None:
            setattr(CONFIG, key, value)

    log_config(CONFIG, logger)

    return CONFIG


# Main --------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse CLI arguments
    CONFIG = parse_args()

    # Init directories
    CONFIG.token_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize
    tokenize_all_fingerprints(CONFIG)
