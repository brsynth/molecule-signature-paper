"""Predict on a dataset using a trained model."""
import logging
import argparse
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch.utils.data import DataLoader

from paper.dataset.utils import (
    setup_logger,
    log_config,
    mol_from_smiles,
    mol_to_smiles,
    mol_to_ecfp,
    ecfp_to_string,
)
from paper.learning.configure import Config
from paper.learning.data import ListDataset, collate_fn_simple
from paper.learning.model import TransformerModel
from paper.learning.utils import Tokenizer


# Logging -----------------------------------------------------------------------------------------

logger = logging.getLogger()

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog('rdApp.warning')

# Disable some Lightining warnings
warnings.filterwarnings("ignore", ".*The 'predict_dataloader' does not have many workers which may be a bottleneck.*")  # noqa


# Utils -------------------------------------------------------------------------------------------

def mol_from_smiles_with_exception(mol):
    """Convert SMILES to RDKit Mol object, handling exceptions."""
    try:
        return mol_from_smiles(mol)
    except Exception:
        return None


def mol_to_ecfp_string(mol) -> str:
    """Convert RDKit Mol object to ECFP string representation."""
    return ecfp_to_string(mol_to_ecfp(mol))


def refine_results(results: pd.DataFrame) -> pd.DataFrame:
    """Refine the results dataframe.

    Cautions: This function assumes that the results dataframe has the following columns:
        - Predicted SMILES
        - Predicted Log Prob

    Parameters
    ----------
    results : pd.DataFrame
        Results dataframe.

    Returns
    -------
    pd.DataFrame
        Refined results dataframe.
    """

    # Let's populate the prediction side
    results["Predicted Prob"] = results["Predicted Log Prob"].apply(np.exp)
    results["Predicted Mol"] = results["Predicted SMILES"].apply(mol_from_smiles_with_exception)
    results["Predicted ECFP Object"] = results["Predicted Mol"].apply(mol_to_ecfp)
    results["Predicted ECFP"] = results["Predicted ECFP Object"].apply(ecfp_to_string)
    results["Predicted Canonic SMILES"] = results["Predicted Mol"].apply(mol_to_smiles)

    # Now let's check for Mol validity
    results["SMILES Syntaxically Valid"] = results["Predicted Mol"].notnull()

    return results


# Args --------------------------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments and return a Config object."""

    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Predict on a dataset.",
    )
    parser.add_argument(
        "--model_path",
        metavar="FILE",
        type=str,
        help="Model file path.",
    )
    parser.add_argument(
        "--model_source_tokenizer",
        metavar="FILE",
        type=str,
        help="Path to the source tokenizer file.",
    )
    parser.add_argument(
        "--model_target_tokenizer",
        metavar="FILE",
        type=str,
        help="Path to the target tokenizer file.",
    )
    parser.add_argument(
        "--query_file",
        metavar="FILE",
        type=str,
        default=None,
        help="Path to the query file. One ECFP per line. If not provided, use query_string. Default: %(default)s",  # noqa
    )
    parser.add_argument(
        "--query_string",
        metavar="STR",
        type=str,
        default="80,80,135,135,199,199,334,378,437,605,640,650,652,656,656,684,693,807,855,881,881,915,915,976,1008,1057,1057,1057,1057,1057,1057,1057,1060,1060,1116,1116,1163,1163,1163,1203,1221,1274,1274,1274,1274,1380,1380,1452,1535,1594,1687,1702,1714,1750,1750,1821,1870,1873,1873,1873,1895,1917,1951,1951",  # noqa        help="ECFP query string. If not provided, query_file must be provided. Default: %(default)s.",  # noqa
        help="ECFP query string. If not provided, query_file must be provided. Default: %(default)s.",  # noqa
    )
    parser.add_argument(
        "--pred_max_rows",
        metavar="INT",
        type=int,
        default=-1,
        help="Maximum number of rows to predict. Use -1 for no limit. Default: %(default)s.",
    )
    parser.add_argument(
        "--pred_max_length",
        metavar="INT",
        type=int,
        default=128,
        help="Maximum length of the prediction. Default: %(default)s.",
    )
    parser.add_argument(
        "--pred_batch_size",
        metavar="INT",
        type=int,
        default=100,
        help="Batch size for prediction. Default: %(default)s.",
    )
    parser.add_argument(
        "--pred_mode",
        metavar="STR",
        type=str,
        choices=["greedy", "beam"],
        default="greedy",
        help="Prediction mode. Choices: %(choices)s. Default: %(default)s.",
    )
    parser.add_argument(
        "--pred_beam_size",
        metavar="INT",
        type=int,
        default=10,
        help="Beam size. Only used if pred_mode is 'beam'. Default: %(default)s.",
    )
    parser.add_argument(
        "--device",
        metavar="STR",
        type=str,
        default="auto",
        help="Device to use. Default: %(default)s.",
    )
    parser.add_argument(
        "--output_file",
        metavar="FILE",
        type=str,
        default=None,
        help="Output file path. Use None to print to stdout. Default: %(default)s.",
    )
    parser.add_argument(
        "--verbosity",
        metavar="STR",
        type=str,
        default="INFO",
        help="Verbosity level. Default: %(default)s.",
    )

    args = parser.parse_args()

    # Set up logger with right logger level
    setup_logger(logger, args.verbosity)

    # Update config with args
    CONFIG = Config()
    for key, value in vars(args).items():
        setattr(CONFIG, key, value)

    log_config(CONFIG, logger)

    return CONFIG


# Main --------------------------------------------------------------------------------------------

def _run():

    CONFIG = parse_args()  # Get config
    results = run(CONFIG)  # Run prediction
    results = refine_results(results)  # Refine results

    # Save results to file
    if CONFIG.output_file is not None:
        results.to_csv(CONFIG.output_file, sep="\t", index=False)

    # Print results to stdout..

    # Enable printing large dataframes
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    if CONFIG.verbosity == "DEBUG":
        print(results)

    else:
        _tmp = results[[
            "Query ID",
            "Predicted SMILES",
            "Predicted Prob",
            "SMILES Syntaxically Valid"
        ]]
        print(_tmp)


def run(CONFIG=None, query_data=None):
    """Run prediction on a dataset using a trained model.

    Parameters
    ----------
    CONFIG : Config, optional
        Configuration object with model and query settings. If None, will parse command line
        arguments. Default is None.
    query_data : pd.DataFrame, pd.Series, list, np.ndarray, str, optional
        Data to use for prediction. If None, will use CONFIG.query_file or CONFIG.query_string.
        Default is None. If provided, it can be a DataFrame with columns "Query ID" and "Query ECFP",
        or a Series, list, numpy array, or string containing ECFP representations.
    Returns
    -------
    pd.DataFrame
        DataFrame with prediction results, including predicted SMILES, log probabilities,
        and ECFP representations.
    """
    # Prepare query settings
    if query_data is not None:

        # Case 1: query_data is a dataframe
        if isinstance(query_data, pd.DataFrame):
            assert "Query ID" in query_data.columns
            assert "Query ECFP" in query_data.columns
            query_df = query_data

        # Case 2: query_data is a series / list / numpy array
        elif isinstance(query_data, (pd.Series, list, np.ndarray)):
            query_df = pd.DataFrame({
                "Query ID": range(1, len(query_data) + 1),  # 1-indexed
                "Query ECFP": query_data
            })

        # Case3: query_data is a string
        elif isinstance(query_data, str):
            query_df = pd.DataFrame({
                "Query ID": [1],  # 1-indexed
                "Query ECFP": [query_data]
            })

        # Case 4: Invalid query_data type
        else:
            raise ValueError("Invalid query_data type.")

    elif CONFIG.query_file is not None:
        # If query_file is provided, read it
        CONFIG.query_file = Path(CONFIG.query_file)

        query_df = pd.read_csv(
            CONFIG.query_file,
            sep="\t",
            nrows=None if CONFIG.pred_max_rows == -1 else CONFIG.pred_max_rows,
            header=None
        )
        query_df.rename(columns={0: "Query ID", 1: "Query ECFP"}, inplace=True)

    else:
        # CONFIG.query_string = CONFIG.query_string.replace(",", "-")
        query_df = pd.DataFrame({
            "Query ID": [1],  # 1-indexed
            "Query ECFP": [CONFIG.query_string]
        })

    # Load tokenizers
    src_tokenizer = Tokenizer(model_path=CONFIG.model_source_tokenizer, fp_type="ECFP")
    tgt_tokenizer = Tokenizer(model_path=CONFIG.model_target_tokenizer, fp_type="SMILES")

    # Load dataset
    dataset = ListDataset(
        data=query_df["Query ECFP"].to_list(),
        token_fn=src_tokenizer,
        max_length=CONFIG.pred_max_length,
        max_rows=CONFIG.pred_max_rows,
    )

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.pred_batch_size,
        collate_fn=collate_fn_simple,
    )

    # Load model
    model = TransformerModel.load_from_checkpoint(CONFIG.model_path)  # noqa
    model.set_decoding_strategy(
        strategy=CONFIG.pred_mode,
        max_length=CONFIG.pred_max_length,
        beam_size=CONFIG.pred_beam_size
    )

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
                    "Query ID": query_df["Query ID"].iloc[idx],
                    "Query ECFP": query_df["Query ECFP"].iloc[idx],
                    "Predicted Tokens": tokens.tolist(),
                    "Predicted Log Prob": logit,
                })
            ]
    results = pd.DataFrame(_tmp)

    # Decode
    results["Predicted SMILES"] = results["Predicted Tokens"].apply(tgt_tokenizer.decode)

    # Return data
    return results


if __name__ == "__main__":
    _run()
