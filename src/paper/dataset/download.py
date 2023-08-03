# Load file and sanitize molecule

import argparse
import os
import sys

import pandas as pd
from library.imports import *
from library.utils import read_tsv, read_txt, write_csv
from library.signature import SanitizeMolecule

from library.utils import read_tsv, read_txt, read_csv, write_csv
from library.signature_alphabet import SignatureAlphabet
from library.signature_alphabet import SignatureFromSmiles, GetSignatureInfo

from retrosig.utils import cmd
from rdkit.Chem import AllChem


def sanitize(data, MaxMolecularWeight: int = 500, size: float = float("inf")):
    # Remove molecules with weight > MaxMolecularWeight
    # and with more than one piece. Make sure all molecules
    # are unique.

    D, SMI = [], set()
    for i in range(data.shape[0]):
        ID, smi = data[i, 0], str(data[i, 1])
        if smi == "nan":
            continue
        if i % 100000 == 0:
            print(f"-------{i} {data[i,0]} {data[i,1]} {len(smi)}")
        if smi.find(".") != -1:
            continue  # not in one piece
        if smi in SMI:
            continue  # aready there
        if len(smi) > int(MaxMolecularWeight / 5):  # Cheap skip
            continue
        mol, smi = SanitizeMolecule(Chem.MolFromSmiles(smi))
        if mol == None:
            continue
        mw = Chem.Descriptors.ExactMolWt(mol)
        if mw > MaxMolecularWeight:
            continue
        if smi in SMI:
            continue  # canonical smi aready there
        SMI.add(smi)
        data[i, 1] = smi
        D.append(data[i])
        if len(D) >= size:
            break
    return np.asarray(D)


# Compute signature in various format
def filter(smi, radius, verbose=False):
    if "." in smi:  #
        return "", "", "", "", None, "", ""
    if "*" in smi:  #
        return "", "", "", "", None, "", ""
    if "[" in smi:  # cannot process [*] without kekularization
        if "@" not in smi:
            return "", "", "", "", None, "", ""
    smiles = smi
    Alphabet = SignatureAlphabet(neighbors=False, radius=radius, nBits=0)
    sig1, mol, smi = SignatureFromSmiles(smi, Alphabet, verbose=False)
    Alphabet = SignatureAlphabet(neighbors=True, radius=radius, nBits=0)
    sig2, mol, smi = SignatureFromSmiles(smi, Alphabet, verbose=False)
    Alphabet = SignatureAlphabet(neighbors=False, radius=radius, nBits=2048)
    sig3, mol, smi = SignatureFromSmiles(smi, Alphabet, verbose=False)
    Alphabet = SignatureAlphabet(neighbors=True, radius=radius, nBits=2048)
    sig4, mol, smi = SignatureFromSmiles(smi, Alphabet, verbose=False)
    if sig1 == "" or sig2 == "" or sig3 == "" or sig4 == "":
        return "", "", "", "", None, "", ""

    mol = AllChem.MolFromSmiles(smiles)
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=2048)
    fp = fpgen.GetFingerprint(mol)  # returns a bit vector (value 1 or 0)
    return sig1, sig2, sig3, sig4, mol, smi, "".join([str(x) for x in fp.ToList()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory-str", required=True, help="Path of the output directory")
    parser.add_argument("--parameters-seed-int", default=0, type=int, help="Seed")
    parser.add_argument(
        "--parameters-max-molecular-weight-int",
        default=500,
        type=int,
        help="Max molecular weight",
    )
    parser.add_argument(
        "--parameters-max-size-int",
        default=float("inf"),
        type=float,
        help="Max size dataset",
    )
    parser.add_argument(
        "--parameters-radius-int", default=2, type=int, help="Max size dataset"
    )
    parser.add_argument(
        "--parameters-valid-percent-float", default=5, type=float, help="Size valid dataset (%)"
    )
    parser.add_argument(
        "--parameters-test-percent-float", default=5, type=float, help="Size test dataset (%)"
    )
    args = parser.parse_args()

    # Init
    MaxMolecularWeight = args.parameters_max_molecular_weight_int
    size = args.parameters_max_size_int

    np.random.seed(seed=args.parameters_seed_int)

    fmetanetx_raw = os.path.join(args.output_directory_str, "metanetx.raw.4_4")
    fmetanetx = os.path.join(args.output_directory_str, "metanetx.4_4")
    fmetanetx_sanitize = os.path.join(
        args.output_directory_str, "metanetx.4_4.sanitize"
    )
    fdataset = os.path.join(args.output_directory_str, "dataset")
    fdataset_train = os.path.join(args.output_directory_str, "dataset.train")
    fdataset_valid = os.path.join(args.output_directory_str, "dataset.valid")
    fdataset_test = os.path.join(args.output_directory_str, "dataset.test")
    falphabet = os.path.join(args.output_directory_str, "alphabet.npz")

    # Get metanetx
    if not os.path.isfile(fmetanetx):
        print("Download metanetx compound")
        if not os.path.isfile(fmetanetx_raw + ".tsv"):
            cmd.url_download(
                url="https://www.metanetx.org/ftp/4.4/chem_prop.tsv",
                path=fmetanetx_raw + ".tsv",
            )
        with open(fmetanetx_raw + ".tsv") as fid, open(fmetanetx+".tsv", "w") as fod:
            towrite = False
            for line in fid:
                if line.startswith("#ID"):
                    towrite = True
                if towrite:
                    fod.write(line)
        os.remove(fmetanetx_raw+".tsv")

    # Sanitize
    if not os.path.isfile(fmetanetx_sanitize + ".csv"):
        print("Start sanitize")
        H, D = read_tsv(fmetanetx)
        H = ["ID", "SMILES"]
        D = D[:, [0, 8]]
        np.random.shuffle(D)
        print(f"size={D.shape[0]}")
        D = sanitize(D, MaxMolecularWeight, size)
        # f'{filename}_weight_{str(MaxMolecularWeight)}'
        # print(f'File={filename_metanetx_sanitize} Header={H} D={D.shape}')
        write_csv(fmetanetx_sanitize, H, D)

    # Create Dataset
    if not os.path.isfile(fdataset + ".csv"):
        H, D = read_csv(fmetanetx_sanitize)
        print(H, D.shape[0])
        Smiles = np.asarray(list(set(D[:, 1])))
        print(f"Number of smiles: {len(Smiles)}")
        #np.random.shuffle(Smiles)

        # Get to business
        H = ["SMILES", "SIG", "SIG-NEIGH", "SIG-NBIT", "SIG-NEIGH-NBIT", "ECFP4"]
        D, i= {}, 0
        for I in range(len(Smiles)):
            sig1, sig2, sig3, sig4, mol, smi, fp = filter(Smiles[i], radius=args.parameters_radius_int)
            if sig1 == "" or sig2 == "" or sig3 == "" or sig4 == "":
                print(Smiles[i])
                i += 1
                continue
            D[I] = [smi, sig1, sig2, sig3, sig4, fp]
            i, I = i + 1, I + 1
            if I == size:
                break
        D = np.asarray(list(D.values()))
        np.random.shuffle(D)
        print("Number of smiles", len(D))
        df = pd.DataFrame(data=D, columns=H)
        df.to_csv(fdataset+".csv", index=False)

    if not os.path.isfile(fdataset_train + ".csv") or not os.path.isfile(fdataset_valid + ".csv") or not os.path.isfile(fdataset_test + ".csv"):
        H, D = read_csv(fdataset)
        Smiles = np.asarray(list(set(D[:, 1])))

        total_size = D.shape[0]
        valid_size = round(args.parameters_valid_percent_float * total_size / 100.0)
        test_size = round(args.parameters_test_percent_float * total_size / 100.0)
        train_size = total_size - valid_size - test_size
        print("Total size:", total_size, "Train size:", train_size, "Valid size:", valid_size, "Test size:" , test_size)
        train_data = D[:train_size]
        valid_data = D[train_size: train_size+valid_size]
        test_data = D[train_size+valid_size:]
        print(D.shape[0], train_data.shape[0], valid_data.shape[0], test_data.shape[0])
        assert train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] == D.shape[0]
        assert train_data.shape[0] == train_size
        assert valid_data.shape[0] == valid_size
        assert test_data.shape[0] == test_size
        
        df_train = pd.DataFrame(data=train_data, columns=H)
        df_train.to_csv(fdataset_train+".csv", index=False)
        df_valid = pd.DataFrame(data=valid_data, columns=H)
        df_valid.to_csv(fdataset_valid+".csv", index=False)
        df_test = pd.DataFrame(data=test_data, columns=H)
        df_test.to_csv(fdataset_test+".csv", index=False)

    # Alphabet Signature
    print("Build Alphabet")
    df = pd.read_csv(fdataset+".csv")
    #Smiles = np.asarray(list(set(D[:, 0])))
    #print(Smiles)
    Alphabet = SignatureAlphabet(
        radius=args.parameters_radius_int, nBits=0, neighbors=False, allHsExplicit=False
    )
    Alphabet.fill(df["SMILES"].tolist(), verbose=True)
    Alphabet.save(falphabet)
    Alphabet.printout()
