#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

from rdkit import RDLogger  # for disabling RDKit warnings
from rdkit.Chem import DataStructs, MolToSmiles
from rdkit.Chem.AllChem import GetMorganGenerator

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

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog('rdApp.warning')


# Utils -------------------------------------------------------------------------------------------
def tanimoto(fp1, fp2):
    # Get rid of None values
    if fp1 is None or fp2 is None:
        return None

    # Compute Tanimoto similarity
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def mol_to_ecfp(mol, include_stereo=True):
    # Get rid of None values
    if mol is None:
        ecfp = None

    else:
        try:
            ecfp = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=include_stereo).GetCountFingerprint(mol)  # noqa E501
            # ecfp = [[idx] * count for idx, count in ecfp.GetNonzeroElements().items()]
            # ecfp = [idx for sublist in ecfp for idx in sublist]

        except Exception:
            ecfp = None

    return ecfp


def mol_to_smiles(mol):
    # Get rid of None values
    if mol is None:
        smiles = None

    else:
        try:
            smiles = MolToSmiles(mol)

        except Exception:
            smiles = None

    return smiles


# Args --------------------------------------------------------------------------------------------
def parse_args():
    CONFIG = Config(db="emolecules", source="ECFP", target="SMILES", mode="predict")
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Evaluate a model on a dataset.",
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
        "--source_col_idx",
        type=int,
        default=4,
        help="Source column index. Default: %(default)s",
    )
    parser.add_argument(
        "--target_col_index",
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
        metavar="STR",
        type=str,
        default="mps",
        help="Device to use for training. Default: %(default)s",
    )
    parser.add_argument(
        "--output_file",
        metavar="FILE",
        type=str,
        default=None,
        help="Output file path. Use None for no output file. Default: %(default)s",
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
def _run():
    CONFIG = parse_args()
    run(CONFIG)


def run(CONFIG):
    """Predict on a dataset.

    Args:
        CONFIG (Config): Configuration object.

    Returns:

    """
    # Load tokenizers
    src_tokenizer = Tokenizer(model_path=CONFIG.source_token_model, fp_type=CONFIG.source)
    trg_tokenizer = Tokenizer(model_path=CONFIG.target_token_model, fp_type=CONFIG.target)

    # Load dataset
    _nrows = None if CONFIG.pred_max_rows == -1 else CONFIG.pred_max_rows
    df = pd.read_csv(CONFIG.pred_test_file, sep="\t", nrows=_nrows)
    dataset = ListDataset(
        data=df.iloc[:, CONFIG.source_col_idx].to_list(),
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
    trainer = L.Trainer(accelerator=CONFIG.device)
    results = trainer.predict(model, dataloader)

    # Store results in a dataframe
    results = [r for batch in results for r in batch]  # remove batch dimension
    _tmp = []
    for idx, result in enumerate(results):
        for tokens, logit in result:
            _tmp += [
                pd.Series({
                    "Seq ID": idx,
                    "Prediction Tokens": tokens,
                    "Prediction Log Prob": logit,
                    "Target SMILES": df.iloc[idx, CONFIG.target_col_idx],
                })
            ]
    results = pd.DataFrame(_tmp)

    # Decode
    results["Prediction SMILES"] = results["Prediction Tokens"].apply(trg_tokenizer.decode)

    # Check if prediction matches target
    results["Target match"] = results["Prediction SMILES"] == results["Target SMILES"]

    # Compute mol, ECFP and canonic SMILES of the prediction
    results["Prediction Mol"] = results["Prediction SMILES"].apply(mol_from_smiles)
    results["Prediction ECFP"] = results["Prediction Mol"].apply(mol_to_ecfp)
    results["Prediction Canonic SMILES"] = results["Prediction Mol"].apply(mol_to_smiles)

    # Compute mol and ECFP of the target
    results["Target Mol"] = results["Target SMILES"].apply(mol_from_smiles)
    results["Target ECFP"] = results["Target Mol"].apply(mol_to_ecfp)

    # Compute Tanimoto similarity between prediction and target
    results["Tanimoto"] = results.apply(lambda x: tanimoto(x["Prediction ECFP"], x["Target ECFP"]), axis=1)  # noqa E501

    # Check if canocalized prediction matches target
    results["Canonic match"] = results["Prediction Canonic SMILES"] == results["Target SMILES"]

    # Print results if verbosity is DEBUG
    if logger.level == logging.DEBUG:
        print(results[["Seq ID", "Prediction Log Prob", "Target match", "Tanimoto", "Canonic match"]])  # noqa E501

    # Return results
    if "output_file" in CONFIG and CONFIG.output_file is not None:
        results.to_csv(CONFIG.output_file, sep="\t", index=False)

    return results


if __name__ == "__main__":
    _run()
