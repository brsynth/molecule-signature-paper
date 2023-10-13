import argparse
import os
import re
import tempfile
from typing import Generator

import pandas as pd
import sentencepiece as spm

# Define the default path for the produced files
OUTPUT_DIR = os.path.join(os.getcwd(), "data")
SPM_DIR = "spm"  # Tokenizer directory
TMP_DIR = "tmp"  # Temporary directory
PAIRS_DIR = "pairs"  # tgt-src pairs directory


class PreTokenizer(object):
    REGEX_ATOMS = re.compile(
        "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    REGEX_SMILES = re.compile(
        "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    @classmethod
    def build_signature(cls, signature: str) -> Generator:
        token_specification = [
            ("FINGERPRINT", r"(\d+,)*"),
            (
                "BOUND",
                r"UNSPECIFIED|SINGLE|DOUBLE|TRIPLEQUADRUPLE|QUINTUPLE|HEXTUPLE|ONEANDAHALF|TWOANDAHALF|THREEANDAHALF|FOURANDAHALF|FIVEANDAHALF|AROMATIC|IONIC|HYDROGEN|THREECENTER|DATIVEONE|DATIVE|DATIVEL|DATIVER|OTHER|ZERO",
            ),
            ("ATOMS", r"[\w=:\d\[\]\(\)]+"),
            ("SPACER", r"[\s\.\|]"),
        ]
        tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
        for mo in re.finditer(tok_regex, signature):
            kind = mo.lastgroup
            value = mo.group()
            tokens = []

            if value == "":
                continue
            if kind == "FINGERPRINT":
                tokens = [value.replace(",", "")] + [","]
            elif kind == "ATOMS":
                tokens = [token for token in cls.REGEX_ATOMS.findall(value)]
            elif kind == "BOUND":
                tokens = [value]  # list(mo.group(1, 2, 3))
            elif kind == "SPACER":
                if value == " ":
                    tokens = ["!"]
                else:
                    tokens = value
            yield tokens

    @classmethod
    def pretokenize_signature(cls, signature: str, sep: str = " ") -> str:
        res = []
        for token in cls.build_signature(signature=signature):
            res.extend(token)
        return sep.join(res)

    @classmethod
    def pretokenize_smiles(cls, smiles: str) -> str:
        tokens = [token for token in cls.REGEX_SMILES.findall(smiles)]
        assert smiles == "".join(tokens)
        return " ".join(tokens)

    @classmethod
    def pretokenize_ecfp4(cls, ecfp4: str) -> str:
        """Return indexes of the on-bits in the ECFP4."""
        on_bits = []
        lecfp4 = ecfp4.split("-")
        assert len(lecfp4) == 2048
        for ix, value in enumerate(lecfp4):
            if value != "0":
                on_bits.extend([ix] * int(value))
        return " ".join([str(i) for i in on_bits])


def count_words(filename: str) -> int:
    words = set()
    with open(filename, "r") as ifile:
        for line in ifile:
            for word in line.strip().split():
                words.add(word)
    return len(words)


def tokenize(
    src_file: str,
    model_prefix: str,
    vocab_size: int = -1,
    model_type: str = "word",
    user_defined_symbols: str = "",
):
    """Train a SentencePiece tokenizer.

    Parameters
    ----------
    src_file : str
        Path to the source file. Each line is a molecule.
    model_prefix : str
        Prefix of the model to be saved.
    vocab_size : int, optional
        Size of the vocabulary (default: -1). By default, the vocabulary is
        not limited in size, which means that all the tokens in the source
        file will be in the vocabulary.
    model_type : str, optional
        Type of the model (default: "word"). Possible values are "word" and
        "unigram".
    user_defined_symbols : str, optional
        User defined symbols (default: ""). This is useful to force the
        inclusion of some tokens in the vocabulary. For example, if you want
        to force the inclusion of a token "[C]", you can set
        `user_defined_symbols="[C]"`.
    """
    if vocab_size == -1:
        vocab_size = count_words(src_file) + 4  # +4 for the special tokens

    if model_type in ["word", "char"]:
        _use_all_vocab = True
        _hard_vocab_limit = True
    else:
        _use_all_vocab = False
        _hard_vocab_limit = False

    print("Vocab size:", vocab_size)
    spm.SentencePieceTrainer.Train(
        input=src_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        use_all_vocab=_use_all_vocab,  # important to avoid error about vocab size
        hard_vocab_limit=_hard_vocab_limit,  # Avoid error about vocab size: https://github.com/google/sentencepiece/issues/605  # noqa E501
        user_defined_symbols=user_defined_symbols,  # Force inclusion of meaningful tokens
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize datasets.")
    parser.add_argument(
        "--input-directory-str",
        required=True,
        help=(
            "Path of the input directory where to find train, test and valid datasets (CSV files). "
            'Files are expected to be named "dataset.train.csv", "dataset.test.csv" and "dataset.valid.csv".'
        ),
    )
    parser.add_argument(
        "--output-directory-str",
        default=OUTPUT_DIR,
        help=f"Path of the output directory. Default: {OUTPUT_DIR}",
    )

    parser.add_argument(
        "--model-type-str",
        default="word",
        choices=["word", "unigram"],
        help=("Model type for the tokenizer. Default: word."),
    )

    parser.add_argument(
        "--depictions-list",
        nargs="+",
        default=["SMILES", "SIG", "SIG-NBIT", "SIG-NEIGH-NBIT", "ECFP4"],
        help=(
            "List of depictions to tokenize. Default: ['SMILES', 'SIG', 'SIG-NBIT', 'SIG-NEIGH-NBIT', 'ECFP4']. "
            "Note: the depictions must be present in the input files."
        ),
    )

    parser.add_argument(
        "--build-pairs-list",
        nargs="+",
        default=["ECFP4.SMILES", "ECFP4.SIG-NEIGH-NBIT", "SIG-NEIGH-NBIT.SMILES"],
        help=(
            "List of pairs of depictions to write. Default: ['ECFP4.SMILES', 'ECFP4.SIG-NEIGH-NBIT', "
            "'SIG-NEIGH-NBIT.SMILES']. "
            "Note: the depictions must be present in the input files."
        ),
    )

    args = parser.parse_args()

    # Summarize arguments
    print("-" * 80)
    print("Arguments:")
    for arg in vars(args):
        print(f"    {arg: <22}: {getattr(args, arg)}")

    # Collect files
    input_files = []
    for file in os.listdir(args.input_directory_str):
        if re.match(r"dataset.(train|test|valid).csv", file):
            input_files.append(os.path.join(args.input_directory_str, file))
    print("-" * 80)
    print("Input files:")
    for file in input_files:
        print(f"    {file}")

    # Initialize output directories
    os.makedirs(args.output_directory_str, exist_ok=True)
    os.makedirs(os.path.join(args.output_directory_str, SPM_DIR), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory_str, PAIRS_DIR), exist_ok=True)

    # Temporary file
    tmpfile = tempfile.NamedTemporaryFile().name
    print("-" * 80)
    print(f"Temporary file: {tmpfile}")

    # Build vocabularies

    # For building the vocabularies, we concatenate all the datasets together.
    df = pd.DataFrame()
    for file in input_files:
        df = pd.concat([df, pd.read_csv(file, index_col=False)])
    df_pretokenized = pd.DataFrame()

    # SMILES
    df_pretokenized["SMILES"] = df["SMILES"].apply(PreTokenizer.pretokenize_smiles)
    df_pretokenized["SMILES"].to_csv(
        tmpfile,
        index=False,
        header=False,
    )
    tokenize(
        src_file=tmpfile,
        model_prefix=os.path.join(args.output_directory_str, SPM_DIR, "SMILES"),
    )
    # SIG
    df_pretokenized["SIG"] = df["SIG"].apply(PreTokenizer.pretokenize_signature)
    df_pretokenized["SIG"].to_csv(
        tmpfile,
        index=False,
        header=False,
    )
    tokenize(
        src_file=tmpfile,
        model_prefix=os.path.join(args.output_directory_str, SPM_DIR, "SIG"),
    )
    # SIG-NBIT
    df_pretokenized["SIG-NBIT"] = df["SIG-NBIT"].apply(
        PreTokenizer.pretokenize_signature
    )
    df_pretokenized["SIG-NBIT"].to_csv(
        tmpfile,
        index=False,
        header=False,
    )
    tokenize(
        src_file=tmpfile,
        model_prefix=os.path.join(args.output_directory_str, SPM_DIR, "SIG-NBIT"),
    )

    # SIG-NEIGH-NBIT
    df_pretokenized["SIG-NEIGH-NBIT"] = df["SIG-NEIGH-NBIT"].apply(
        PreTokenizer.pretokenize_signature
    )
    df_pretokenized["SIG-NEIGH-NBIT"].to_csv(
        tmpfile,
        index=False,
        header=False,
    )
    tokenize(
        src_file=tmpfile,
        model_prefix=os.path.join(args.output_directory_str, SPM_DIR, "SIG-NEIGH-NBIT"),
    )

    # ECFP4
    df_pretokenized["ECFP4"] = df["ECFP4"].apply(PreTokenizer.pretokenize_ecfp4)
    df_pretokenized["ECFP4"].to_csv(
        tmpfile,
        index=False,
        header=False,
    )
    tokenize(
        src_file=tmpfile,
        model_prefix=os.path.join(args.output_directory_str, SPM_DIR, "ECFP4"),
    )

    # Build target-source pairs
    #
    # This is useful for the training of the models.
    #
    # For each dataset (train, test, valid), we build files containing the (target, source) pairs.
    # Each row contains one couple {target format}\t{source format}, e.g.: {smiles}\t{signature}
    for file in input_files:
        pattern = re.compile(r"dataset\.(?P<type>train|test|valid)\.csv")
        type_ = pattern.search(file).group("type")
        df = pd.read_csv(file)
        df_pretokenized = pd.DataFrame()
        df_pretokenized["SMILES"] = df["SMILES"].apply(PreTokenizer.pretokenize_smiles)
        df_pretokenized["SIG"] = df["SIG"].apply(PreTokenizer.pretokenize_signature)
        df_pretokenized["SIG-NBIT"] = df["SIG-NBIT"].apply(
            PreTokenizer.pretokenize_signature
        )
        df_pretokenized["SIG-NEIGH-NBIT"] = df["SIG-NEIGH-NBIT"].apply(
            PreTokenizer.pretokenize_signature
        )
        df_pretokenized["ECFP4"] = df["ECFP4"].apply(PreTokenizer.pretokenize_ecfp4)
        # SMILES - SIG
        df_pretokenized[["SMILES", "SIG"]].to_csv(
            os.path.join(args.output_directory_str, PAIRS_DIR, f"SIG.SMILES.{type_}"),
            sep="\t",
            index=False,
            header=False,
        )
        # SMILES - SIG-NBIT
        df_pretokenized[["SMILES", "SIG-NBIT"]].to_csv(
            os.path.join(
                args.output_directory_str, PAIRS_DIR, f"SIG-NBIT.SMILES.{type_}"
            ),
            sep="\t",
            index=False,
            header=False,
        )
        # SMILES - SIG-NEIGH-NBIT
        df_pretokenized[["SMILES", "SIG-NEIGH-NBIT"]].to_csv(
            os.path.join(
                args.output_directory_str, PAIRS_DIR, f"SIG-NEIGH-NBIT.SMILES.{type_}"
            ),
            sep="\t",
            index=False,
            header=False,
        )
        # SIG - ECFP4
        df_pretokenized[["SIG", "ECFP4"]].to_csv(
            os.path.join(args.output_directory_str, PAIRS_DIR, f"ECFP4.SIG.{type_}"),
            sep="\t",
            index=False,
            header=False,
        )
        # SIG-NBIT - ECFP4
        df_pretokenized[["SIG-NBIT", "ECFP4"]].to_csv(
            os.path.join(
                args.output_directory_str, PAIRS_DIR, f"ECFP4.SIG-NBIT.{type_}"
            ),
            sep="\t",
            index=False,
            header=False,
        )
        # SIG-NEIGH-NBIT - ECFP4
        df_pretokenized[["SIG-NEIGH-NBIT", "ECFP4"]].to_csv(
            os.path.join(
                args.output_directory_str, PAIRS_DIR, f"ECFP4.SIG-NEIGH-NBIT.{type_}"
            ),
            sep="\t",
            index=False,
            header=False,
        )
        # SMILES - ECFP4
        df_pretokenized[["SMILES", "ECFP4"]].to_csv(
            os.path.join(args.output_directory_str, PAIRS_DIR, f"ECFP4.SMILES.{type_}"),
            sep="\t",
            index=False,
            header=False,
        )
