# Load file and sanitize molecule
# Final file format is two columns ID + canonical SMILES
# Warning: this cell is slow

from library.imports import *
from library.utils import read_tsv, read_txt, write_csv
from library.signature import SanitizeMolecule

from library.imports import *
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

    fpgen = AllChem.GetMorganGenerator(radius=2)
    fp = fpgen.GetFingerprint(mol)
    return sig1, sig2, sig3, sig4, mol, smi, "".join(fp.ToList())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory-str", help="Path of the output directory")
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
    args = parser.parse_args()

    # Init
    MaxMolecularWeight = args.parameters_max_molecular_weight_int
    size = args.parameters_max_size_int

    np.random.seed(seed=args.seed_int)

    fmetanetx = os.path.join(args.output_directory_str, "metanetx.4_4")
    fmetanetx_sanitize = os.path.join(
        args.output_directory_str, "metanetx.4_4.sanitize"
    )
    fdataset = os.path.join(args.output_directory_str, "dataset.csv")

    # Get metanetx
    if not os.path.isfile(fmetanetx):
        cmd.url_download(
            url="https://www.metanetx.org/ftp/4.4/chem_prop.tsv",
            path=fmetanetx + ".tsv",
        )

    # Read metanetx
    H, D = read_tsv(fmetanetx)
    H = ["ID", "SMILES"]
    D = D[:, [0, 8]]
    np.random.shuffle(D)
    print(f"size={D.shape[0]}")
    D = sanitize(D, MaxMolecularWeight, size)
    # f'{filename}_weight_{str(MaxMolecularWeight)}'
    # print(f'File={filename_metanetx_sanitize} Header={H} D={D.shape}')
    write_csv(filename_metanetx_sanitize, H, D)

    # Load Smiles file
    H, D = read_csv(fmetanetx_sanitize)
    print(H, D.shape[0])
    Smiles = np.asarray(list(set(D[:, 1])))
    print(f"Number of smiles: {len(Smiles)}")
    np.random.shuffle(Smiles)

    # Get to business
    H = ["SMILES", "ECFP4", "SIG", "SIG-NEIGH", "SIG-NBIT", "SIG-NEIGH-NBIT"]
    D, i, I = {}, 0, 0
    fpgen = AllChem.GetMorganGenerator(radius=2)
    while True:
        sig1, sig2, sig3, sig4, mol, smi, fp = filter(Smiles[i], radius)
        if sig1 == "" or sig2 == "" or sig3 == "" or sig4 == "":
            print(Smiles[i])
            i += 1
            continue
        D[I] = [smi, fp, sig1, sig2, sig3, sig4]
        i, I = i + 1, I + 1
        if I == size:
            break
    D = np.asarray(list(D.values()))
    write_csv(fdataset, H, D)
