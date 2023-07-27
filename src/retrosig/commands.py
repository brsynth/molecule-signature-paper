import argparse
import logging
import os
import time

from retrosig.library.imports import *
from retrosig.library.utils import read_csv
from retrosig.library.signature_alphabet import AlphabetObject, LoadAlphabet
from retrosig._version import __app_name__, __version__
from retrosig.utils import cmdline

AP = argparse.ArgumentParser(
    description=__app_name__
    + " provides a cli interface to manipulate chemical signature",
    epilog="See online documentation: https://github.com/brsynth/" + __app_name__,
)
AP_subparsers = AP.add_subparsers(help="Sub-commnands (use with -h for more info)")


def _cmd_alpha_sig(args):
    logging.info("Start - Alphabet Signature")
    # Check arguments.
    if not os.path.isfile(args.input_smiles_csv):
        cmdline.abort(
            AP, "Input csv file does not exist: %s" % (args.input_smiles_csv,)
        )
    cmdline.check_output_dir(parser=AP, path=args.output_directory_str)
    radius = args.parameters_radius_int
    nBits = args.parameters_nbits_int
    neighbors = True
    if args.parameters_not_neighbors_bool:
        neighbors = args.parameters_not_neighbors_bool
    allHsExplicit = False
    if args.parameters_all_hs_explicit_bool:
        allHsExplicit = args.parameters_all_hs_explicit_bool

    # Ouput basename.
    ext = "_N" if neighbors else ""
    ext += "H" if allHsExplicit else ""
    ext = ext + "_" + str(radius) + "_" + str(nBits)
    # file_smiles = "./rules/retrorules_rr02_flat_all"
    output_basename = "retrosig_alphabet_signature" + ext

    # Load Smiles file
    logging.info("Load file")
    H, D = read_csv(args.input_smiles_csv)
    print(f"Header={H}\nD={D.shape}")
    Smiles = np.asarray(list(set(D[:, 7]).union(set(D[:, 9]))))
    print(f"Number of smiles: {len(Smiles)}")

    # Get save and load Alphabet
    logging.info("Build Alphabet")
    start_time = time.time()
    Alphabet = AlphabetObject(
        radius=radius, nBits=nBits, neighbors=neighbors, allHsExplicit=allHsExplicit
    )
    Alphabet.fill(Smiles, verbose=True)

    logging.info("Save Alphabet")
    Alphabet.save(output_basename)
    Alphabet = LoadAlphabet(file_alphabet)
    print(f"CPU time compute Alphabet: {time.time() - start_time:.2f}")
    Alphabet.printout()

    logging.info("End - Alphabet Signature")


P_alpha_sig = AP_subparsers.add_parser(
    "alphabet-signature", help=_cmd_alpha_sig.__doc__
)
# Input
P_alpha_sig_input = P_alpha_sig.add_argument_group("Input")
P_alpha_sig_input.add_argument(
    "--input-smiles-csv", required=True, help="CSV file with 2 columns"
)
# Output
P_alpha_sig_output = P_alpha_sig.add_argument_group("Output")
P_alpha_sig_output.add_argument(
    "--output-directory-str", required=True, help="Output directory"
)
# Parameters
P_alpha_sig_params = P_alpha_sig.add_argument_group("Parameters")
P_alpha_sig_params.add_argument(
    "--parameters-radius-int", type=int, default=2, help="Radius value"
)
P_alpha_sig_params.add_argument(
    "--parameters-nbits-int",
    type=int,
    default=2048,
    help="",
)
P_alpha_sig_params.add_argument(
    "--parameters-not-neighbors-bool", action="store_true", help="Compute neighbors"
)
P_alpha_sig_params.add_argument(
    "--parameters-all-hs-explicit-bool", action="store_true", help="Add all HsExplicit"
)


P_alpha_sig.set_defaults(func=_cmd_alpha_sig)


# Version.
def print_version(_args):
    """Display this program"s version"""
    print(__version__)


P_version = AP_subparsers.add_parser("version", help=print_version.__doc__)
P_version.set_defaults(func=print_version)


# Help.
def print_help():
    """Display this program"s help"""
    print(AP_subparsers.help)
    AP.exit()


# Main.
def parse_args(args=None):
    """Parse the command line"""
    return AP.parse_args(args=args)
