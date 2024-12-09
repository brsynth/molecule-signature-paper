# Load file and sanitize molecule

import argparse
import os

import pandas as pd
from openbabel import pybel
from retrosig.utils import cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory-str",
        required=True,
        help="Path of the output directory",
    )

    args = parser.parse_args()
    outdir = args.output_directory_str

    # Initialize random seed

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Define the path of the produced files
    fraw = os.path.join(outdir, "emolecules.2023-07-01.sdf.gz")
    fsource = os.path.join(outdir, "emolecules.2023-07-01")
    max_size = 1e6
    current_file = 0
    current_item = 0
    # Get metanetx
    if not os.path.isfile(fraw):
        print("Download emolecules compound")
        cmd.url_download(
            url="https://downloads.emolecules.com/free/2023-07-01/version.sdf.gz",
            path=fraw,
        )
    print("Format emolecules file")
    data = {}
    for mol in pybel.readfile("sdf", fraw):
        if current_item > max_size - 1:
            df = pd.DataFrame.from_dict(
                data, orient="index", columns=["SMILES", "INCHI"]
            )
            df.index.name = "#ID"
            df.to_csv(fsource + "." + str(current_file) + ".csv")
            current_item = 0
            current_file += 1
            data = {}

        ident = mol.data["EMOL_VERSION_ID"]
        smi = mol.write("smi").replace("\n", "").replace("\t", "")
        inchi = mol.write("inchi").replace("\n", "").replace("\t", "")

        data[ident] = [smi, inchi]
        current_item += 1

    df = pd.DataFrame.from_dict(data, orient="index", columns=["SMILES", "INCHI"])
    df.index.name = "#ID"
    df.to_csv(fsource + ".raw." + str(current_file) + ".csv")
