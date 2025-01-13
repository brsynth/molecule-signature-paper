import argparse
import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from paper.dataset.utils import setup_logger, log_config
from paper.learning.configure import Config
from paper.learning.data import ListDataset, collate_fn_simple
from paper.learning.model import TransformerModel
from paper.learning.utils import Tokenizer


# Logging -----------------------------------------------------------------------------------------

logger = logging.getLogger()


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
        "--pred_output_file",
        metavar="FILE",
        type=str,
        help="Output file path.",
    )
    parser.add_argument(
        "--pred_scores",
        metavar="BOOL",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Return scores. Default: %(default)s.",
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
    run(CONFIG)


def run(CONFIG=None):
    # Load tokenizers
    src_tokenizer = Tokenizer(model_path=CONFIG.model_source_tokenizer, fp_type="ECFP")
    tgt_tokenizer = Tokenizer(model_path=CONFIG.model_target_tokenizer, fp_type="SMILES")

    # Prepare query settings
    if CONFIG.query_file is not None:
        CONFIG.query_file = Path(CONFIG.query_file)
        _nrows = None if CONFIG.pred_max_rows == -1 else CONFIG.pred_max_rows
        df = pd.read_csv(CONFIG.query_file, sep="\t", nrows=_nrows, header=None)
        df.rename(columns={0: "ECFP"}, inplace=True)

    else:
        CONFIG.query_string = CONFIG.query_string.replace(",", "-")
        df = pd.DataFrame({"ECFP": [CONFIG.query_string]})

    # Load dataset
    dataset = ListDataset(
        data=df["ECFP"].to_list(),
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
                    "Prediction Tokens": tokens,
                    "Score": np.exp(logit),
                })
            ]
    results = pd.DataFrame(_tmp)

    # Decode
    results["Prediction SMILES"] = results["Prediction Tokens"].apply(tgt_tokenizer.decode)

    # Drop prediction tokens (not needed)
    results.drop(columns=["Prediction Tokens"], inplace=True)

    # Add a rank column based on the prediction score of each query
    results["Rank"] = results.groupby("Query ID")["Score"].rank(ascending=False).astype(int)

    # Reorder columns
    results = results[["Query ID", "Rank", "Prediction SMILES", "Score"]]

    # Output
    if CONFIG.output_file is not None:
        results.to_csv(CONFIG.output_file, sep="\t", index=False)

    else:
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
            "display.max_colwidth", None,
        ):
            print(results)


if __name__ == "__main__":
    _run()
