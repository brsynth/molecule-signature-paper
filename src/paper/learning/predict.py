import argparse
import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch.utils.data import DataLoader

from paper.dataset.utils import setup_logger, log_config
from paper.learning.configure import Config
from paper.learning.data import ListDataset, collate_fn_simple
from paper.learning.model import TransformerModel
from paper.learning.utils import (
    Tokenizer,
    mol_from_smiles,
    mol_to_smiles,
    mol_to_ecfp,
    ecfp_to_string,
)


# Logging -----------------------------------------------------------------------------------------

logger = logging.getLogger()

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog('rdApp.warning')


# Utils -------------------------------------------------------------------------------------------

def mol_to_ecfp_string(mol) -> str:
    return ecfp_to_string(mol_to_ecfp(mol))


def refine_results(results: pd.DataFrame) -> pd.DataFrame:
    results["Prediction Prob"] = results["Prediction Log Prob"].apply(np.exp)
    results["Prediction Mol"] = results["Prediction SMILES"].apply(mol_from_smiles)
    results["Prediction ECFP"] = results["Prediction Mol"].apply(mol_to_ecfp_string)
    results["Prediction Canonic SMILES"] = results["Prediction Mol"].apply(mol_to_smiles)

    return results


# Args --------------------------------------------------------------------------------------------

def parse_args():
    # query-file: str or query-string: str
    # if query-file:
    #   one ECFP per line
    # else query-string
    # beam-size: int, opt, default=5
    # batch-size: int, opt, default=32
    # output-file: str, opt, default=None
    # scores: bool, opt, default=False
    # model: str, opt, default=CONFIG

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
        default="DEBUG",
        help="Verbosity level. Default: %(default)s.",
    )

    args = parser.parse_args()

    # Set up logger with right logger level
    setup_logger(logger, args.verbosity)

    # Update config with args
    CONFIG = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(CONFIG, key, value)

    print(CONFIG.to_dict())
    log_config(CONFIG, logger)

    return CONFIG


# Main --------------------------------------------------------------------------------------------

def _run():
    CONFIG = parse_args()
    results = run(CONFIG)

    if CONFIG.output_file is not None or CONFIG.verbosity == "DEBUG":
        # Comute additional columns
        results = refine_results(results)

        # Columns to display
        _tmp = results[[
            "Query ID",
            "Query",
            "Rank",
            "Prediction SMILES",
            "Prediction ECFP",
            "Prediction Canonic SMILES",
            "Prediction Prob"
        ]]

        if CONFIG.output_file is not None:
            _tmp.to_csv(CONFIG.output_file, sep="\t", index=False)

        if CONFIG.verbosity == "DEBUG":
            # Enable printing large dataframes
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", None)

            # Print
            print(_tmp)


def run(CONFIG=None, query_data=None):

    # Prepare query settings
    if query_data is not None:

        # Case 1: query_data is a dataframe
        if isinstance(query_data, pd.DataFrame):
            assert "Query" in query_data.columns
            query_df = query_data

        # Case 2: query_data is a series / list / numpy array
        elif isinstance(query_data, (pd.Series, list, np.ndarray)):
            query_df = pd.DataFrame({"Query": query_data})

        # Case3: query_data is a string
        elif isinstance(query_data, str):
            query_df = pd.DataFrame({"Query": [query_data]})

        # Case 4: Invalid query_data type
        else:
            raise ValueError("Invalid query_data type.")

    elif CONFIG.query_file is not None:
        CONFIG.query_file = Path(CONFIG.query_file)
        query_df = pd.read_csv(
            CONFIG.query_file,
            sep="\t",
            nrows=None if CONFIG.pred_max_rows == -1 else CONFIG.pred_max_rows,
            header=None
        )
        query_df.rename(columns={0: "Query"}, inplace=True)

    else:
        CONFIG.query_string = CONFIG.query_string.replace(",", "-")
        query_df = pd.DataFrame({"Query": [CONFIG.query_string]})

    # Load tokenizers
    src_tokenizer = Tokenizer(model_path=CONFIG.model_source_tokenizer, fp_type="ECFP")
    tgt_tokenizer = Tokenizer(model_path=CONFIG.model_target_tokenizer, fp_type="SMILES")

    # Load dataset
    dataset = ListDataset(
        data=query_df["Query"].to_list(),
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
                    "Query ID": idx,
                    "Query": query_df["Query"].iloc[idx],
                    "Prediction Tokens": tokens,
                    "Prediction Log Prob": logit,
                    # "Score": np.exp(logit),
                })
            ]
    results = pd.DataFrame(_tmp)

    # Decode
    results["Prediction SMILES"] = results["Prediction Tokens"].apply(tgt_tokenizer.decode)

    # Add a rank column based on the prediction score of each query
    results["Rank"] = results.groupby("Query ID")["Prediction Log Prob"].rank(ascending=False).astype(int)  # noqa

    # Return data
    return results


if __name__ == "__main__":
    _run()
