import argparse
import os
import re
import tempfile
from typing import Generator

import numpy as np
import pandas as pd
import sentencepiece as spm

# Define the default path for the produced files
OUTPUT_DIR = os.path.join(os.getcwd(), "data")
SPM_DIR = "spm"  # Tokenizer directory
TMP_DIR = "tmp"  # Temporary directory
PAIRS_DIR = "pairs"  # tgt-src pairs directory


class PreTokenizer(object):
    REGEX_ATOMS = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    REGEX_SMILES = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    @classmethod
    def build_signature(cls, signature: str) -> Generator:
        token_specification = [
            ("FINGERPRINT", r"(\d+,)*"),
            (
                "BOUND",
                r"UNSPECIFIED|SINGLE|DOUBLE|TRIPLEQUADRUPLE|QUINTUPLE|HEXTUPLE|ONEANDAHALF|TWOANDAHALF|THREEANDAHALF|FOURANDAHALF|FIVEANDAHALF|AROMATIC|IONIC|HYDROGEN|THREECENTER|DATIVEONE|DATIVE|DATIVEL|DATIVER|OTHER|ZERO",  # noqa E501
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

    # For building vocabularies, we concatenate all the datasets together.
    df = pd.DataFrame()
    for file in input_files:
        df = pd.concat([df, pd.read_csv(file, index_col=False)])
    # df_pretokenized = pd.DataFrame()

    if args.model_type_str == "word":
        # Tokenize using word-based model
        # NOTICE: word-based model requires pre-tokenization of the input (i.e. splitting the input into words)
        for depic in args.depictions_list:
            # Print
            print("-" * 80)
            print(f"Tokenizing {depic}")

            # Case by case pre-tokenization
            if depic == "SMILES":
                df[depic] = df[depic].apply(PreTokenizer.pretokenize_smiles)
            elif depic == "ECFP4":
                df[depic] = df[depic].apply(PreTokenizer.pretokenize_ecfp4)
            elif depic in ["SIG", "SIG-NBIT", "SIG-NEIGH-NBIT"]:
                df[depic] = df[depic].apply(PreTokenizer.pretokenize_signature)
            else:
                raise ValueError(f"Unknown depic: {depic}")

            # Write to temporary txt file (one molecule per line)
            np.savetxt(
                tmpfile,
                df[depic].values,
                fmt="%s",
            )

            # Tokenize
            tokenize(
                src_file=tmpfile,
                model_prefix=os.path.join(args.output_directory_str, SPM_DIR, depic),
                model_type=args.model_type_str,
            )

    elif args.model_type_str == "unigram":
        # Tokenize using unigram-based model
        # NOTICE: unigram-based model does not require pre-tokenization of the input
        #   (i.e. the input is tokenized as a whole)
        # however it requires vocabulary size to be specified

        # Hard coded vocabulary sizes (DEBUG)
        _VOCAB_SIZE = {
            "SMILES": 1024,
            "ECFP4": 2048,
            "SIG": 4096,
            "SIG-NBIT": 8192,
            "SIG-NEIGH-NBIT": 8192,
        }

        # Hard coded custom tokens (DEBUG)
        _USER_DEFINED_SYMBOLS = {
            "SMILES": "",
            "ECFP4": "",
            "SIG": "",
            "SIG-NBIT": "",
            "SIG-NEIGH-NBIT": ",".join(
                [
                    "UNSPECIFIED",
                    "SINGLE",
                    "DOUBLE",
                    "TRIPLEQUADRUPLE",
                    "QUINTUPLE",
                    "HEXTUPLE",
                    "ONEANDAHALF",
                    "TWOANDAHALF",
                    "THREEANDAHALF",
                    "FOURANDAHALF",
                    "FIVEANDAHALF",
                    "AROMATIC",
                    "IONIC",
                    "HYDROGEN",
                    "THREECENTER",
                    "DATIVEONE",
                    "DATIVE",
                    "DATIVEL",
                    "DATIVER",
                    "OTHER",
                    "ZERO",
                ]
            ),
        }

        for depic in args.depictions_list:
            # Get vocabulary size and user defined symbols
            vocab_size = _VOCAB_SIZE[depic]
            user_defined_symbols = _USER_DEFINED_SYMBOLS[depic]

            # Print
            print("-" * 80)
            print(f"Tokenizing {depic} with vocab size {vocab_size}")

            # Special treatment for ECFP4
            # Only keep the on-bits, only them bring information
            if depic == "ECFP4":
                df[depic] = df[depic].apply(PreTokenizer.pretokenize_ecfp4)

            # Write to temporary txt file (one molecule per line)
            np.savetxt(
                tmpfile,
                df[depic].values,
                fmt="%s",
            )

            # Tokenize
            tokenize(
                src_file=tmpfile,
                model_prefix=os.path.join(args.output_directory_str, SPM_DIR, depic),
                vocab_size=vocab_size,
                model_type=args.model_type_str,
                user_defined_symbols=user_defined_symbols,
            )

    # Build pairs of depictions for training, testing and validation sets
    for src_depic, tgt_depic in [item.split(".") for item in args.build_pairs_list]:
        # Print
        print("-" * 80)
        print(f"Building tgt-src pairs: {src_depic}.{tgt_depic}")

        # Iterate over datasets
        pattern = re.compile(r"dataset\.(?P<type>train|test|valid)\.csv")
        for file in input_files:  # Remember: input_files are datasets of interests
            type_ = pattern.search(file).group("type")

            # Print dataset
            print(f"    {type_}")

            # Read
            df = pd.read_csv(file)

            # Pre-tokenize
            for depic in [src_depic, tgt_depic]:
                if depic == "ECFP4":
                    df[depic] = df[depic].apply(PreTokenizer.pretokenize_ecfp4)
                elif depic == "SMILES" and args.model_type_str == "word":
                    df[depic] = df[depic].apply(PreTokenizer.pretokenize_smiles)
                elif (
                    depic in ["SIG", "SIG-NBIT", "SIG-NEIGH-NBIT"]
                    and args.model_type_str == "word"
                ):
                    df[depic] = df[depic].apply(PreTokenizer.pretokenize_signature)

            # Write
            with open(
                os.path.join(
                    args.output_directory_str,
                    PAIRS_DIR,
                    f"{src_depic}.{tgt_depic}.{type_}",
                ),
                "w",
            ) as ofile:
                # Using numpy
                np.savetxt(
                    ofile,
                    df[[tgt_depic, src_depic]].values,
                    fmt="%s",
                    delimiter="\t",
                )
