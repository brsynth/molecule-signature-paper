import argparse
import csv
import glob
import gzip
import os
import re

import numpy as np
import pandas as pd

from pandarallel import pandarallel
from library.signature_alphabet import SignatureAlphabet
from library.utils import read_csv, read_tsv, write_csv
from retrosig.utils import cmd
from paper.dataset.utils import (
    sanitize,
    filter_smi,
    df_sig1,
    df_sig2,
    df_sig3,
    df_sig4,
    df_ecfp4,
)


def get_number(path: str) -> int:
    m = re.search(r".(\d+).csv.gz", path)
    nb = int(m.group(1))
    assert nb > -1
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory-str",
        required=True,
        help="Path of the output directory",
    )
    parser.add_argument("--parameters-seed-int", default=0, type=int, help="Seed")
    parser.add_argument(
        "--parameters-max-molecular-weight-int",
        default=500,
        type=int,
        help="Max molecular weight",
    )
    parser.add_argument(
        "--parameters-max-dataset-size-int",
        default=2e5,
        type=float,
        help="Max dataset size",
    )
    parser.add_argument(
        "--parameters-radius-int", default=2, type=int, help="Radius of the signature"
    )
    parser.add_argument(
        "--parameters-valid-percent-float",
        default=10,
        type=float,
        help="Size of the validation dataset (%%)",
    )
    parser.add_argument(
        "--parameters-test-percent-float",
        default=10,
        type=float,
        help="Size of the test dataset (%%)",
    )
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Number of threads",
    )

    args = parser.parse_args()

    # Initialize random seed
    np.random.seed(seed=args.parameters_seed_int)
    rng = np.random.default_rng(seed=args.parameters_seed_int)
    pandarallel.initialize(nb_workers=args.threads, progress_bar=True)

    # Define the path of the produced files
    femolecules = os.path.join(args.output_directory_str, "emolecules.2023-07-01")

    femolecules_sanitize = os.path.join(
        args.output_directory_str, "emolecules.2023-07-01.sanitize"
    )
    femolecules_sig = os.path.join(
        args.output_directory_str, "emolecules.2023-07-01.sig"
    )
    fdataset_all = os.path.join(args.output_directory_str, "dataset.all.gz")
    fdataset_subset = os.path.join(args.output_directory_str, "dataset.subset")
    fdataset_train = os.path.join(args.output_directory_str, "dataset.train")
    fdataset_valid = os.path.join(args.output_directory_str, "dataset.valid")
    fdataset_test = os.path.join(args.output_directory_str, "dataset.test")
    fdataset_test_small = os.path.join(args.output_directory_str, "dataset.test.small")
    fsig = os.path.join(args.output_directory_str, "sig.alphabet.npz")
    fsig_nbit = os.path.join(args.output_directory_str, "sig_nbit.alphabet.npz")

    # Create output directory
    if not os.path.isdir(args.output_directory_str):
        os.makedirs(args.output_directory_str)

    # Sanitize
    for filename in glob.glob(
        os.path.join(args.output_directory_str, "emolecules.2023-07-01.raw.*.csv.gz")
    ):
        nb = get_number(path=filename)
        foutput = femolecules_sanitize + "." + str(nb) + ".csv.gz"
        if os.path.isfile(foutput):
            continue
        print("Start sanitize:", filename)
        df_chunk = pd.read_csv(filename)
        df_chunk["SMILES"] = df_chunk["SMILES"].apply(
            sanitize,
            args=(args.parameters_max_molecular_weight_int,),
        )
        df_chunk = df_chunk[["#ID", "SMILES"]]
        df_chunk = df_chunk[~pd.isna(df_chunk["SMILES"])]
        df_chunk.to_csv(foutput, index=False)

    # Signature
    for filename in glob.glob(
        os.path.join(args.output_directory_str, "emolecules.2023-07-01.sanitize.*.csv.gz")
    ):
        nb = get_number(path=filename)
        foutput = femolecules_sig + "." + str(nb) + ".csv.gz"
        if os.path.isfile(foutput):
            continue
        print("Start signature:", filename)

        df_chunk = pd.read_csv(filename)
        df_chunk = filter_smi(df=df_chunk)

        print("Step", "SIG", df_chunk.shape[0])
        df_chunk["SIG"] = df_chunk["SMILES"].parallel_apply(
            df_sig1, args=(args.parameters_radius_int,)
        )
        df_chunk = df_chunk[df_chunk["SIG"] != ""]

        print("Step", "SIG-NEIGH", df_chunk.shape[0])
        df_chunk["SIG-NEIGH"] = df_chunk["SMILES"].parallel_apply(
            df_sig2, args=(args.parameters_radius_int,)
        )
        df_chunk = df_chunk[df_chunk["SIG-NEIGH"] != ""]

        print("Step", "SIG-NBIT", df_chunk.shape[0])
        df_chunk["SIG-NBIT"] = df_chunk["SMILES"].parallel_apply(
            df_sig3, args=(args.parameters_radius_int,)
        )
        df_chunk = df_chunk[df_chunk["SIG-NBIT"] != ""]

        print("Step", "SIG-NEIGH-NBIT", df_chunk.shape[0])
        df_chunk["SIG-NEIGH-NBIT"] = df_chunk["SMILES"].parallel_apply(
            df_sig4, args=(args.parameters_radius_int,)
        )
        df_chunk = df_chunk[df_chunk["SIG-NEIGH-NBIT"] != ""]

        print("Step", "ECFP4", df_chunk.shape[0])
        df_chunk["ECFP4"] = df_chunk["SMILES"].parallel_apply(
            df_ecfp4, args=(args.parameters_radius_int,)
        )

        df_chunk.to_csv(foutput, index=False)

    # Create Dataset
    if not os.path.isfile(fdataset_all + ".csv.gz"):
        print("Create dataset")
        with gzip.open(fdataset_all, "at", compresslevel=4) as fd:
            for ix, filename in enumerate(
                glob.glob(
                    os.path.join(
                        args.output_directory_str,
                        "emolecules.2023-07-01.sig.*.csv.gz",
                    )
                )
            ):
                with gzip.open(filename, "rt") as fid:
                    for ij, line in enumerate(fid):
                        if ix > 0 and ij < 1:
                            continue
                        fd.write(line)

    # Subset
    if not os.path.isfile(fdataset_subset + ".csv"):
        if args.parameters_max_dataset_size_int < 0:
            with gzip.open(fdataset_all, "rt", newline="") as fid, open(fdataset_subset + ".csv", "w") as fod:
                for line in fid:
                    fod.write(line)
        else:
            # Count total
            total = 0
            with gzip.open(fdataset_all, "rt", newline="") as fd:
                for line in fd:
                    total += 1

            # Sort
            keep = rng.choice(a=np.arange(2, total + 1), size=int(args.parameters_max_dataset_size_int), replace=False)
            keep.sort()
            cur_ix = 0
            total = 0
            with gzip.open(fdataset_all, "rt", newline="") as fid, open(fdataset_subset + ".csv", "w", newline="") as fod:
                csv_in = csv.DictReader(fid)
                csv_out = csv.DictWriter(fod, fieldnames=csv_in.fieldnames)
                csv_out.writeheader()
                for row in csv_in:
                    if csv_in.line_num == keep[cur_ix]:
                        csv_out.writerow(row)
                        total += 1
                        cur_ix += 1
                        if cur_ix > args.parameters_max_dataset_size_int:
                            break
            assert total == args.parameters_max_dataset_size_int

    # Split Dataset
    if (
        not os.path.isfile(fdataset_train + ".csv")
        or not os.path.isfile(fdataset_valid + ".csv")
        or not os.path.isfile(fdataset_test + ".csv")
        or not os.path.isfile(fdataset_test_small + ".csv")
    ):
        H, D = read_csv(fdataset_subset)

        Smiles = np.asarray(list(set(D[:, 1])))

        total_size = D.shape[0]
        valid_size = round(args.parameters_valid_percent_float * total_size / 100.0)
        test_size = round(args.parameters_test_percent_float * total_size / 100.0)
        train_size = total_size - valid_size - test_size
        print(
            "Total size:",
            total_size,
            "Train size:",
            train_size,
            "Valid size:",
            valid_size,
            "Test size:",
            test_size,
        )
        train_data = D[:train_size]
        valid_data = D[train_size : train_size + valid_size]
        test_data = D[train_size + valid_size :]
        test_small_data = D[train_size + valid_size : train_size + valid_size + 1000]
        print(D.shape[0], train_data.shape[0], valid_data.shape[0], test_data.shape[0])
        assert (
            train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] == D.shape[0]
        )
        assert train_data.shape[0] == train_size
        assert valid_data.shape[0] == valid_size
        assert test_data.shape[0] == test_size
        assert test_small_data.shape[0] == 1000

        df_train = pd.DataFrame(data=train_data, columns=H)
        df_train.to_csv(fdataset_train + ".csv", index=False)
        df_valid = pd.DataFrame(data=valid_data, columns=H)
        df_valid.to_csv(fdataset_valid + ".csv", index=False)
        df_test = pd.DataFrame(data=test_data, columns=H)
        df_test.to_csv(fdataset_test + ".csv", index=False)
        df_test_small = pd.DataFrame(data=test_small_data, columns=H)
        df_test_small.to_csv(fdataset_test_small + ".csv", index=False)

    # Alphabet Signature
    print("Build Signature alphabet")
    df = pd.read_csv(fdataset_subset + ".csv")
    Alphabet = SignatureAlphabet(
        radius=args.parameters_radius_int, nBits=0, neighbors=False, allHsExplicit=False
    )
    Alphabet.fill(df["SMILES"].tolist(), verbose=True)
    Alphabet.save(fsig)
    Alphabet.printout()

    Alphabet = SignatureAlphabet(
        radius=args.parameters_radius_int,
        nBits=2048,
        neighbors=False,
        allHsExplicit=False,
    )
    Alphabet.fill(df["SMILES"].tolist(), verbose=True)
    Alphabet.save(fsig_nbit)
    Alphabet.printout()
