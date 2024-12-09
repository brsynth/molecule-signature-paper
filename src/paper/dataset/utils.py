import argparse
import re
import gzip
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Generator, List, Optional, Iterator

import coloredlogs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions, EnumerateStereoisomers

from signature.Signature import MoleculeSignature


# Logging -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def setup_logger(logger: logging.Logger = logging.getLogger(), level: int = logging.INFO) -> None:
    """Setup logger."""
    logger.setLevel(level)
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Fingerprint section -----------------------------------------------------------------------------
def assign_stereo(mol: Chem.Mol) -> Chem.Mol:
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True)
    stereoisomers = EnumerateStereoisomers(mol, options=options)  # returns a generator
    return next(stereoisomers, None)  # None is a default if the generator is empty


def get_smiles(mol: Chem.Mol, clear_aam: bool = True, clear_isotope: bool = True) -> str:
    try:
        if clear_aam:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        if clear_isotope:
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
        return Chem.MolToSmiles(mol)
    except Exception as err:
        logger.error(f"Error processing molecule: {Chem.MolToSmiles(mol)}")
        logger.error(err)
        raise err


def get_signature(mol: Chem.Mol, radius: int, nbits: int) -> str:
    try:
        mol_sig = MoleculeSignature(mol, radius=radius, nbits=nbits)
        return mol_sig.to_string(morgans=False)

    except Exception as err:
        logger.error(f"Error processing molecule: {Chem.MolToSmiles(mol)}")
        logger.error(err)
        raise err


def get_ecfp(mol: Chem.Mol, radius: int, nbits: int) -> str:
    fp = AllChem.GetMorganGenerator(
        radius=radius,
        fpSize=nbits,
        includeChirality=True
    ).GetCountFingerprint(mol)
    non_zero = []
    for ix, count in fp.GetNonzeroElements().items():
        non_zero.extend([ix] * count)
    return "-".join([str(x) for x in non_zero])


# Tokenize section --------------------------------------------------------------------------------
def pretokenizer(fp: str, pattern: re.Pattern) -> str:
    """General pretokenizer for a string representation of a fingerprint."""
    # Extract tokens
    tokens = [match.group() for match in pattern.finditer(fp)]

    # Build pretokenized fingerprint
    pretokenized_fp = " ".join(tokens)

    # Check for incomplete extraction
    msg = f"Pre-tokenization failed: {fp} -> {pretokenized_fp}"
    try:
        assert fp == pretokenized_fp.replace(" ", "")
    except AssertionError:
        logger.warning(msg)
        return None

    return pretokenized_fp


def pretokenizer_for_smiles(smiles: str) -> str:
    """Pre-tokenize a SMILES string"""
    TOKEN_PATTERN = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"  # noqa E501
    )

    return pretokenizer(smiles, TOKEN_PATTERN)


def pretokenizer_for_signature(signature: str) -> str:
    """Pre-tokenize a string representation of a signature"""
    tokens_pattern = re.compile(
        r"(\[[^\[\]]+\])"      # Atoms : [content]
        r"|(\.\.)"             # Separator : ..
        # r"|(<\-|->)"           # Dative bonds: <-, ->
        r"|([-=:#\\/])"        # Bonds : -, =, :, #, /, \
        r"|([()])"             # Branching : ( or )
        r"|(\d)"               # Cycles closure : 1-9
    )
    signature = signature.replace(" ", "")  # Remove spaces
    return pretokenizer(signature, tokens_pattern)


def pretokenizer_for_ecfp(ecfp: str) -> str:
    """Pre-tokenize the ECFP fingerprints."""
    # Special case for ECFP
    ecfp = ecfp.replace("-", " ")  # Replace dashes with spaces

    return ecfp


class __OLD__PreTokenizer(object):
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


def __OLD__count_words(filename: str) -> int:
    words = set()
    with open(filename, "r") as ifile:
        for line in ifile:
            for word in line.strip().split():
                words.add(word)
    return len(words)


def __OLD__tokenize(
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


# Configurations utilitary methods ----------------------------------------------------------------

def log_config(
        config: argparse.Namespace,
        logger: logging.Logger = logging.getLogger(),
) -> None:
    """Print config settings."""
    yaml_str = yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False, indent=8)

    logger.info(f"{'Settings ':-<80}")
    for line in yaml_str.split("\n"):
        logger.info(line)
    logger.info("-" * 80)


def log_config_deprecated(
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
    for key, value in vars(namespace).items():
        if isinstance(value, SimpleNamespace):
            logger.info(f"{' ' * _PRE_WIDTH}{key + '.':<}")
            log_config(value, recursion_level + 1, logger)
        else:
            logger.info(f"{' ' * _PRE_WIDTH}{key:<{_COL_WIDTH - _PRE_WIDTH}}: {value}")
    logger.info("-" * 80) if recursion_level == 0 else None


# Files section -----------------------------------------------------------------------------------

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


# Data section ------------------------------------------------------------------------------------

class CustomDataset():
    def __init__(
            self,
            file_path: str | Path,
            has_header: bool = True,
            offsets: Optional[List[int]] = None,
            header_offset: Optional[int] = None,
    ):
        self.file_path = file_path
        self.has_header = has_header

        if offsets is not None:
            self.offsets = offsets
            if self.has_header:
                if header_offset is not None:
                    self.header_offset = header_offset
                else:
                    raise ValueError("header_offset must be provided if had_header is True")
        else:
            self.offsets = self._index_file()

    def _index_file(self) -> List[int]:
        """Return the offsets of the lines in the dataset."""
        offsets = []
        with open(self.file_path, 'rb') as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)

        if self.has_header:
            self.header_offset = offsets[0]
            offsets = offsets[1:]  # Skip the header

        return offsets

    def __len__(self):
        """Return the number of lines in the dataset."""
        return len(self.offsets)

    def __getitem__(self, idx: int) -> str:
        """Return a line from the dataset.

        Parameters
        ----------
        idx : int
            Index of the line to return.

        Returns
        -------
        str
            A line from the dataset.
        """
        offset = self.offsets[idx]
        with open(self.file_path, 'rb') as f:
            f.seek(offset)
            return f.readline().decode('utf-8')

    @property
    def header(self) -> str:
        """Return the header of the dataset."""
        with open(self.file_path, 'rb') as f:
            f.seek(self.header_offset)
            return f.readline().decode('utf-8')

    @classmethod
    def subset(cls, dataset: "CustomDataset", idxs: List[int]) -> "CustomDataset":
        """Return a subset of the dataset.

        Parameters
        ----------
        dataset : CustomDataset
            Dataset to subset.
        idxs : List[int]
            Indexes of the subset.

        Returns
        -------
        CustomDataset
            Subset of the dataset.
        """
        if any(idx < 0 or idx >= len(dataset) for idx in idxs):
            raise IndexError("One or more indexes are out of bounds")

        new_offsets = [dataset.offsets[i] for i in idxs]
        header_offset = dataset.header_offset if dataset.has_header else None

        return cls(
            file_path=dataset.file_path,
            has_header=dataset.has_header,
            offsets=new_offsets,
            header_offset=header_offset
        )

    def __iter__(self):
        """Iterate over the dataset.

        Yields
        ------
        str
            A line from the dataset.
        """
        with open(self.file_path, 'rb') as f:
            if self.has_header:
                f.seek(self.header_offset)
                f.readline()  # Ignore header

            for offset in self.offsets:
                f.seek(offset)
                line = f.readline()
                if not line:
                    break  # Fin du fichier
                yield line.decode('utf-8')

    def iter_chunks(self, chunk_size: int = 1000) -> Iterator[List[str]]:
        """Iterate over the dataset in chunks.

        Parameters
        ----------
        chunk_size : int, optional
            Size of the chunks (default: 1000).

        Yields
        ------
        List[str]
            A chunk of lines from the dataset.
        """
        chunk = []
        for idx, line in enumerate(self):
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
