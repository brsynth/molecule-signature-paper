from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from library.signature import SanitizeMolecule
from library.signature_alphabet import SignatureAlphabet, SignatureFromSmiles


def sanitize(smiles: str, max_molecular_weight: int = 500) -> Optional[str]:
    # Remove molecules with weight > max_molecular_weight
    # and with more than one piece. Make sure all molecules
    # are unique.

    res = pd.NA
    if smiles == "nan" or pd.isna(smiles):
        return res
    if smiles.find(".") != -1:
        return res  # not in one piece
    if len(smiles) > int(max_molecular_weight / 5):  # Cheap skip
        return res
    mol, smiles = SanitizeMolecule(Chem.MolFromSmiles(smiles), formalCharge=True)
    if mol is None:
        return res
    if Chem.Descriptors.ExactMolWt(mol) > max_molecular_weight:
        return res
    return smiles


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
