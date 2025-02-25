import argparse
import re
import gzip
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Iterator
from urllib.request import Request, urlopen

import coloredlogs
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions, EnumerateStereoisomers

from molsig.Signature import MoleculeSignature


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


# Cheminformatics section -------------------------------------------------------------------------

def mol_from_smiles(
        smiles: str,
        clear_stereo: bool = False,
        clear_aam: bool = True,
        clear_isotope: bool = True,
        clear_hs: bool = True
) -> Chem.Mol:
    """Sanitize a molecule

    Parameters
    ----------
    smiles : str
        Smiles string to sanitize.
    max_mw : int, optional
        Maximum molecular weight, by default 500.
    clear_stereo : bool, optional
        Clear stereochemistry information, by default True.
    clear_aam : bool, optional
        Clear atom atom mapping, by default True.
    clear_isotope : bool, optional
        Clear isotope information, by default True.
    clear_hs : bool, optional
        Clear hydrogen atoms, by default True.

    Returns
    -------
    Chem.Mol
        Sanitized molecule. If smiles is not valid, returns None.
    """
    try:
        if smiles == "nan" or smiles == "" or pd.isna(smiles):
            return
        if "." in smiles:  # Reject molecules
            return
        if "*" in smiles:   # Reject generic molecules
            return

        if clear_stereo:  # Wild but effective
            smiles = smiles.replace("@", "").replace("/", "").replace("\\", "")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        if clear_aam:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

        if clear_isotope:
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

        if clear_stereo or clear_isotope or clear_hs:
            # Removing stereochemistry and isotope information might leave
            # the molecule with explicit hydrogens that does not carry any
            # useful information. We remove them.
            mol = Chem.RemoveHs(mol)

        return mol

    except Exception as err:
        raise err


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Get SMILES string for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to convert to SMILES.

    Returns
    -------
    str
        SMILES string. If mol is None, returns None.
    """
    if mol is None:
        smiles = None

    else:
        try:
            smiles = Chem.MolToSmiles(mol)

        except Exception:
            smiles = None

    return smiles


def mol_to_ecfp(mol: Chem.Mol, radius=2, nbits=2048, include_chirality: bool = True) -> DataStructs:
    """Get ECFP fingerprint for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    radius : int, optional
        Radius of the ECFP, by default 2.
    nbits : int, optional
        Number of bits for the fingerprint, by default 2048.
    include_chirality : bool, optional
        Include stereochemistry information, by default True.

    Returns
    -------
    DataStructs
        ECFP fingerprint.
    """
    try:
        return GetMorganGenerator(
            radius=radius,
            fpSize=nbits,
            includeChirality=include_chirality
        ).GetCountFingerprint(mol)

    except Exception:
        return None


def ecfp_to_string(ecfp: DataStructs, sep="-") -> str:
    """Convert ECFP to a string.

    Bit are duplicated based on their count.

    Parameters
    ----------
    ecfp : rdkit.DataStructs
        ECFP fingerprint.
    sep : str, optional
        Separator for the string, by default "-".

    Returns
    -------
    str
        ECFP as a string.
    """
    if ecfp is None:
        return None

    non_zero = []
    for ix, count in ecfp.GetNonzeroElements().items():
        non_zero.extend([ix] * count)

    return sep.join([str(x) for x in non_zero])


def tanimoto(fp1, fp2):
    # Get rid of None values
    if fp1 is None or fp2 is None:
        return None

    # Compute Tanimoto similarity
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# Fingerprint section -----------------------------------------------------------------------------

def assign_stereo(mol: Chem.Mol) -> Chem.Mol:
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True)
    stereoisomers = EnumerateStereoisomers(mol, options=options)  # returns a generator

    new_mol = next(stereoisomers, None)  # None is a default if the generator is empty
    if new_mol is None:
        logger.warning(f"No stereoisomer found for molecule {Chem.MolToSmiles(mol)}")
        return mol

    Chem.SanitizeMol(new_mol)
    return new_mol


def get_smiles(mol: Chem.Mol, clear_aam: bool = True, clear_isotope: bool = True) -> str:
    _mol = Chem.Mol(mol)  # Copy the molecule to avoid modifying the original
    try:
        if clear_aam:
            for atom in _mol.GetAtoms():
                atom.SetAtomMapNum(0)

        if clear_isotope:
            for atom in _mol.GetAtoms():
                atom.SetIsotope(0)

        if clear_aam or clear_isotope:
            # Get rid of eventual excplicit Hs that might be not useful anymore
            _mol = Chem.RemoveHs(_mol)

        return Chem.MolToSmiles(_mol)

    except Exception as err:
        logger.error(f"Error processing molecule: {Chem.MolToSmiles(mol)}")
        logger.error(err)
        raise err


def get_signature(mol: Chem.Mol, radius: int, nbits: int) -> str:
    try:
        return MoleculeSignature(mol, radius=radius, nbits=nbits)

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


# System section ----------------------------------------------------------------------------------

def url_download(url: str, path: str) -> None:
    try:
        with urlopen(Request(url)) as fod:
            with open(path, "wb") as dst:
                while True:
                    chunk = fod.read(2**10)
                    if chunk:
                        dst.write(chunk)
                    else:
                        break
    except Exception as e:
        print(str(e))
