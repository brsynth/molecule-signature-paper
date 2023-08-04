import argparse
import os
import re
import sys
from typing import Generator

import pandas as pd
import sentencepiece as spm



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
            ("ATOMS", r"[\w=:\d\[\]\(\)]+"),
            (
                "BOUND",
                r"UNSPECIFIED|SINGLE|DOUBLE|TRIPLEQUADRUPLE|QUINTUPLE|HEXTUPLE|ONEANDAHALF|TWOANDAHALF|THREEANDAHALF|FOURANDAHALF|FIVEANDAHALF|AROMATI|IONIC|HYDROGEN|THREECENTER|DATIVEONE|DATIVE|DATIVEL|DATIVER|OTHER|ZERO",
            ),
            ("SPACER", r"[\s\.\|]"),
        ]
        tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
        is_fingerprint = False
        for mo in re.finditer(tok_regex, signature):
            kind = mo.lastgroup
            value = mo.group()
            tokens = []

            if value == "":
                continue
            if kind == "FINGERPRINT":
                tokens = list(value)
            elif kind == "ATOMS":
                tokens = [token for token in cls.REGEX_ATOMS.findall(value)]
            elif kind == "BOUND":
                tokens = value  # list(mo.group(1, 2, 3))
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
        assert smiles == ''.join(tokens)
        return ' '.join(tokens)

    @classmethod
        return " ".join(list(ecfp4))
    def pretokenize_ecfp4(cls, ecfp4: str) -> str:

def count_words(filename: str) -> int:
    words = set()
    with open(filename, "r") as ifile:
        for line in ifile:
            for word in line.strip().split():
                words.add(word)

    return len(words)


def tokenize(src_file: str, model_prefix: str, vocab_size: int = -1):
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
    """
    if vocab_size == -1:
        vocab_size = count_words(src_file) + 4  # +4 for the special tokens

    spm.SentencePieceTrainer.Train(
        input=src_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="word",
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-csv", required=True, help="Dataset file csv")
    parser.add_argument("--output-directory-str", required=True, help="Path of the output directory")

    args = parser.parse_args()

    outdir = args.output_directory_str

    # Init
    os.makedirs(outdir, exist_ok=True)

    # Load
    df = pd.read_csv(args.input_dataset_csv)

    # Smiles
    tokens = []
    for smiles in df["SMILES"]:
        tokens.append(Tokenizer.tokenize_smiles(smiles=smiles))
    with open(os.path.join(args.output_directory_str, "smiles.txt"), "w") as fod:
        fod.write("\n".join(tokens) + "\n")

    # Signature
    tokens = []
    for signature in df["SIG"]:
        tokens.append(Tokenizer.tokenize_signature(signature=signature))
    with open(os.path.join(args.output_directory_str, "signature.txt"), "w") as fod:
        fod.write("\n".join(tokens) + "\n")
    # signature = "1,8,2040,C=[O:1].DOUBLE|2,6,1021,C[C:1](C)=O 2,6,1021,C[C:1](C)=O.DOUBLE|1,8,2040,C=[O:1].SINGLE|2,6,1439,C[CH:1](O)O.SINGLE|2,6,1928,C[CH:1](C)O"
    # print(Tokenizer.tokenize_signature(signature=signature))
    
    # ECFP4
    tokens = []
    for ecfp4 in df["ECFP4"]:
        tokens.append(Tokenizer.tokenize_ecfp4(ecfp4=str(ecfp4)))
    with open(os.path.join(args.output_directory_str, "ecfp4.txt"), "w") as fod:
        fod.write("\n".join(tokens) + "\n")

