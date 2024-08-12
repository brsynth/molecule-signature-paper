import re
import gzip
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Generator

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from signature.Signature import MoleculeSignature

# # Old sig import, to be deprecated
# try:
#     from library.signature import SanitizeMolecule
#     from library.signature_alphabet import SignatureAlphabet, SignatureFromSmiles
# except ImportError:
#     logging.error("library.signature and library.signature_alphabet not found (utils.py).")


# Logging ---------------------------------------------------------------------
logger = logging.getLogger(__name__)


# Sanitize section ------------------------------------------------------------
def sanitize(
        smiles: str,
        max_molecular_weight: int = 500,
        isomeric_smiles: bool = False
) -> str:
    """Sanitize smiles string.

    Parameters
    ----------
    smiles : str
        Smiles string to sanitize.
    max_molecular_weight : int, optional
        Maximum molecular weight, by default 500.
    isomeric_smiles : bool, optional
        Whether to keep isomeric smiles, by default False.

    Returns
    -------
    str
        Sanitized smiles string. If smiles is not valid, returns empty string.
    """
    # Skip if not a string
    if smiles == "nan" or smiles == "" or pd.isna(smiles):
        return ""
    # Skip if not in one piece
    if "." in smiles:
        return ""
    # Skip if R-group
    if "*" in smiles:
        return ""
    # Skip if too long
    if len(smiles) > int(max_molecular_weight / 5):  # Cheap skip
        return res
    mol, smiles = SanitizeMolecule(Chem.MolFromSmiles(smiles), formalCharge=True)
    if mol is None:
        return res
    if Chem.Descriptors.ExactMolWt(mol) > max_molecular_weight:
        return res
    return smiles


# Fingerprint section -----------------------------------------------------------------------------

def get_signature(mol: Chem.Mol, radius: int, nbits: int) -> str:
    try:
        mol_sig = MoleculeSignature(mol, radius=radius, nbits=nbits)
        return mol_sig.to_string()

    except Exception as err:
        logger.error(f"Error processing molecule: {Chem.MolToSmiles(mol)}")
        logger.error(err)
        raise err


def get_ecfp(mol: Chem.Mol, radius: int, nbits: int) -> str:
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = fp_gen.GetCountFingerprint(mol)
    non_zero = []
    for ix, count in fp.GetNonzeroElements().items():
        non_zero.extend([ix] * count)
    return "-".join([str(x) for x in non_zero])


# Tokenize section --------------------------------------------------------------------------------

class PreTokenizer(object):
    REGEX_ATOMS = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    REGEX_SMILES = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
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
        try:
            assert smiles == "".join(tokens)
        except AssertionError as err:
            print("ERROR: SMILES and tokens do not match")
            print("SMILES:", smiles)
            print("TOKENS JOINED:", "".join(tokens))
            print("TOKENS:", tokens)
            print("Exiting...")
            raise err

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
        input_sentence_size=1e7,  # Cutoff to avoid memory issues
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


# Configurations utilitary methods --------------------------------------------

def update_config_paths(namespace: SimpleNamespace, old_path: Path, new_path: Path) -> None:
    """Update config paths.

    This method updates the paths of a config namespace. It is
    using recursion to update all paths of the config namespace.
    """
    for key, value in vars(namespace).items():
        if isinstance(value, SimpleNamespace):
            update_config_paths(value, old_path, new_path)
        elif isinstance(value, Path):
            setattr(namespace, key, Path(str(value).replace(str(old_path), str(new_path))))
        else:
            pass


def log_config(
        namespace: SimpleNamespace,
        recursion_level: int = 0,
        logger: logging.Logger = logging.getLogger(),
) -> None:
    """Print config settings.

    This method prints the settings of a config namespace. It is
    using recursion to print all settings of the config namespace.
    """
    _COL_WIDTH = 32  # column width
    _PRE_WIDTH = 4 + (4 * recursion_level)  # indent width
    if recursion_level == 0:
        logger.info(f"{'Settings ':-<80}")
    for key, value in sorted(vars(namespace).items()):
        if isinstance(value, SimpleNamespace):
            logger.info(f"{' ' * _PRE_WIDTH}{key + '.':<}")
            log_config(value, recursion_level + 1)
        else:
            logger.info(f"{' ' * _PRE_WIDTH}{key:<{_COL_WIDTH - _PRE_WIDTH}}: {value}")
    logger.info("-" * 80) if recursion_level == 0 else None


# Handy section --------------------------------------------------------------

def open_file(filepath: str | Path, mode: str = "rt"):
    """Open file with gzip support if needed.

    Parameters
    ----------
    filepath : str | Path
        Path to the file.
    mode : str, optional
        Mode to open the file (default: "rt").
    """
    try:
        filepath = str(filepath)
    except AttributeError:
        pass
    if filepath.endswith(".gz"):
        if mode == "r":
            mode = "rt"
        elif mode == "w":
            mode = "wt"
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)

# Compute signature in various format
def filter_smi(df: pd.DataFrame):
    df = df[~df["SMILES"].str.contains("\.")]
    df = df[~df["SMILES"].str.contains("\*")]
    df = df[~pd.isna(df["SMILES"])]
    return df


def df_sig1(smi: str, radius: int):
    sig1, mol, smi = SignatureFromSmiles(
        smi, SignatureAlphabet(neighbors=False, radius=radius, nBits=0), verbose=False
    )
    return sig1


def df_sig2(smi: str, radius: int):
    sig2, mol, smi = SignatureFromSmiles(
        smi, SignatureAlphabet(neighbors=True, radius=radius, nBits=0), verbose=False
    )
    return sig2


def df_sig3(smi: str, radius: int):
    sig3, mol, smi = SignatureFromSmiles(
        smi,
        SignatureAlphabet(neighbors=False, radius=radius, nBits=2048),
        verbose=False,
    )
    return sig3


def df_sig4(smi: str, radius: int):
    sig4, mol, smi = SignatureFromSmiles(
        smi, SignatureAlphabet(neighbors=True, radius=radius, nBits=2048), verbose=False
    )
    return sig4


def df_ecfp4(smi: str, radius: int):
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=2048)
    # fp = fpgen.GetFingerprint(mol)  # returns a bit vector (value 1 or 0)
    fp = fpgen.GetCountFingerprint(AllChem.MolFromSmiles(smi))
    return "-".join([str(x) for x in fp.ToList()])
