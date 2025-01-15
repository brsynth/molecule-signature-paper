import argparse
from pathlib import Path
from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)

from paper.dataset.utils import log_config


# Config class ------------------------------------------------------------------------------------

@dataclass
class Config:
    model_path: Path = field(default=None)
    model_source_tokenizer: Path = field(default=None)
    model_target_tokenizer: Path = field(default=None)

    query_file: Path = field(default=None)
    query_string: str = field(default=None)

    pred_max_rows: int = -1
    pred_max_length: int = 128
    pred_batch_size: int = 100
    pred_mode: str = "greedy"  # or "beam"
    pred_beam_size: int = 10

    def __dict__(self):
        def serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif is_dataclass(obj):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            else:
                return obj

        return serialize(self)

    def __str__(self):
        for key, value in self.__dict__().items():
            print(f"{key}: {value}")

    def to_dict(self):
        return self.__dict__()

    def save(self, output_file: Path):
        with open(output_file, "w") as f:
            for key, value in self.__dict__().items():
                f.write(f"{key}: {value}\n")

    def load(self, file_path: Path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            key, value = line.strip().split(": ")
            setattr(self, key, value)

    def log(self):
        log_config(self)

    @classmethod
    def from_dict(cls, dict_: dict):
        return cls(**dict_)

    @classmethod
    def from_file(cls, file_path: Path):
        config = cls()
        config.load(file_path)
        return config


# Main --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configuring the model, query and output files"
    )
    parser.add_argument(
        "--model_path",
        metavar="FILE",
        type=Path,
        help="Path to the model file.",
    )
    parser.add_argument(
        "--model_source_tokenizer",
        metavar="FILE",
        type=Path,
        help="Path to the source tokenizer file.",
    )
    parser.add_argument(
        "--model_target_tokenizer",
        metavar="FILE",
        type=Path,
        help="Path to the target tokenizer file.",
    )
    parser.add_argument(
        "--query_file",
        metavar="FILE",
        type=Path,
        default=None,
        help="Path to the query file. One ECFP per line. If not provided, use query_string. Default: %(default)s",  # noqa
    )
    parser.add_argument(
        "--query_string",
        metavar="STR",
        type=str,
        default=None,
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
        "--config_file",
        metavar="FILE",
        type=str,
        default="config.yml",
        help="File where to write the config. Default: %(default)s.",
    )

    args = parser.parse_args()
    config = Config(
        model_path=args.model_path,
        model_source_tokenizer=args.model_source_tokenizer,
        model_target_tokenizer=args.model_target_tokenizer,
        query_file=args.query_file,
        query_string=args.query_string,
        pred_max_rows=args.pred_max_rows,
        pred_max_length=args.pred_max_length,
        pred_batch_size=args.pred_batch_size,
        pred_mode=args.pred_mode,
        pred_beam_size=args.pred_beam_size,
    )

    log_config(config)
    config.save(Path(args.config_file))
