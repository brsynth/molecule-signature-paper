#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

from signature.utils import mol_from_smiles
from paper.learning.config import Config
from paper.learning.data import ListDataset, collate_fn_simple
from paper.learning.utils import Tokenizer
from paper.learning.model import TransformerModel
from paper.dataset.utils import (
    setup_logger,
    log_config,
)

# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger()


# Utils -------------------------------------------------------------------------------------------
def tanimoto(fp1, fp2):
    # Load packages
    try:
        from rdkit.Chem import DataStructs
    except ImportError:
        raise ImportError("Please install the signature package to use this function.")

    # Get rid of None values
    if fp1 is None or fp2 is None:
        return None

    # Compute Tanimoto similarity
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def mol_to_ecfp(mol):
    # Load packages
    try:
        from rdkit import RDLogger
        from rdkit.Chem.AllChem import GetMorganGenerator
    except ImportError:
        raise ImportError("Please install the signature package to use this function.")

    # Disable RDKit warnings
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog('rdApp.warning')

    # Convert to ECFP
    if mol is None:
        ecfp = None
    else:
        try:
            ecfp = GetMorganGenerator(radius=2, fpSize=2048).GetCountFingerprint(mol)
            # ecfp = [[idx] * count for idx, count in ecfp.GetNonzeroElements().items()]
            # ecfp = [idx for sublist in ecfp for idx in sublist]
        except Exception:
            ecfp = None

    return ecfp


# Args --------------------------------------------------------------------------------------------
def parse_args():
    CONFIG = Config(db="emolecules", source="ECFP", target="SMILES", mode="predict")
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Predict on a dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model file path.",
    )
    parser.add_argument(
        "--db",
        type=str,
        choices=["emolecules", "metanetx"],
        default="emolecules",
        help=(
            "Database used to build the tokens. This will determine internally which "
            "tokenizers to use. Choices: %(choices)s. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["ECFP", "SIGNATURE", "SMILES"],
        default="ECFP",
        help="Source data type. Choices: %(choices)s. Default: %(default)s",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["ECFP", "SIGNATURE", "SMILES"],
        default="SMILES",
        help="Target data type. Choices: %(choices)s. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_test_file",
        type=str,
        default=CONFIG.pred_test_file,
        help="Data file path. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_source_index",
        type=int,
        default=4,
        help="Source column index. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_target_index",
        type=int,
        default=2,
        help="Target column index. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_max_rows",
        type=int,
        default=10,
        help="Max number of rows to read from the data file. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_batch_size",
        type=int,
        default=16,
        help="Prediction batch size. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_max_length",
        type=int,
        default=128,
        help="Prediction max length. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_mode",
        type=str,
        choices=["greedy", "beam"],
        default="greedy",
        help="Prediction mode. Default: %(default)s",
    )
    parser.add_argument(
        "--pred_beam_size",
        type=int,
        default=10,
        help="Prediction beam size. Default: %(default)s",
    )
    parser.add_argument(
        "--verbosity",
        metavar="LEVEL",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Verbosity level (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use for training. Default: %(default)s",
    )

    args = parser.parse_args()

    # Set up logger with right logger level
    setup_logger(logger, args.verbosity)

    # Update config with args
    CONFIG = Config(db=args.db, source=args.source, target=args.target, mode="predict")
    for key, value in vars(args).items():
        if value is not None:
            setattr(CONFIG, key, value)

    log_config(CONFIG, logger)

    return CONFIG


# Main --------------------------------------------------------------------------------------------
def main():
    CONFIG = parse_args()

    # Load tokenizers
    src_tokenizer = Tokenizer(model_path=CONFIG.source_token_model, fp_type=CONFIG.source)
    trg_tokenizer = Tokenizer(model_path=CONFIG.target_token_model, fp_type=CONFIG.target)

    # Load dataset
    df = pd.read_csv(CONFIG.pred_test_file, sep="\t", nrows=CONFIG.pred_max_rows)
    dataset = ListDataset(
        data=df.iloc[:, CONFIG.pred_source_index].to_list(),
        token_fn=src_tokenizer,
        max_length=CONFIG.pred_max_length,
        max_rows=CONFIG.pred_max_rows,
    )

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=CONFIG.pred_batch_size, collate_fn=collate_fn_simple)  # noqa E501

    # Load model
    model = TransformerModel.load_from_checkpoint(CONFIG.model_path)
    model.set_decoding_strategy(strategy=CONFIG.pred_mode, max_length=CONFIG.pred_max_length, beam_size=CONFIG.pred_beam_size)  # noqa E501

    # Predict
    trainer = L.Trainer(accelerator="mps")
    results = trainer.predict(model, dataloader)

    # Store results in a dataframe
    results = [r for batch in results for r in batch]  # remove batch dimension
    _tmp = []
    for idx, result in enumerate(results):
        for tokens, score in result:
            _tmp += [
                pd.Series({
                    "Seq": idx,
                    "Tokens": tokens,
                    "Score": score,
                    "Target": df.iloc[idx, CONFIG.pred_target_index],
                })
            ]
    results = pd.DataFrame(_tmp)

    # Decode
    results["Prediction"] = results["Tokens"].apply(trg_tokenizer.decode)

    # Check target equality
    results["Target match"] = results["Prediction"] == results["Target"]

    # Compute mol and ECFP of the prediction
    results["Prediction Mol"] = results["Prediction"].apply(mol_from_smiles)
    results["Prediction ECFP"] = results["Prediction Mol"].apply(mol_to_ecfp)

    # Compute mol and ECFP of the target
    results["Target Mol"] = results["Target"].apply(mol_from_smiles)
    results["Target ECFP"] = results["Target Mol"].apply(mol_to_ecfp)

    # Compute the Tanimoto similarity
    results["Tanimoto"] = results.apply(lambda x: tanimoto(x["Prediction ECFP"], x["Target ECFP"]), axis=1)  # noqa E501

    # Save results as pickle
    results.to_pickle("results.pkl")

    # Print results if verbosity is DEBUG
    if logger.level == logging.DEBUG:
        print(results)


if __name__ == "__main__":
    main()
