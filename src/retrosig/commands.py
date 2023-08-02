import argparse
import logging
import os
import time

from library.imports import *
from library.utils import read_csv
#from library.signature_alphabet import AlphabetObject, LoadAlphabet
#from library.one_step_retro import ReactionObject, LoadReaction
from retrosig._version import __app_name__, __version__
from retrosig.utils import cmdline

AP = argparse.ArgumentParser(
    description=__app_name__
    + " provides a cli interface to manipulate chemical signature",
    epilog="See online documentation: https://github.com/brsynth/" + __app_name__,
)
AP_subparsers = AP.add_subparsers(help="Sub-commnands (use with -h for more info)")


def _cmd_alphabet(args):
    logging.info("Start - Alphabet")
    # Check arguments
    if not os.path.isfile(args.input_smiles_csv):
        cmdline.abort(
            AP, "Input csv file does not exist: %s" % (args.input_smiles_csv,)
        )
    cmdline.check_output_file(parser=AP, path=args.output_signature_npz)
    cmdline.check_output_file(parser=AP, path=args.output_reaction_npz)

    # Init
    radius = args.parameters_radius_int
    nBits = args.parameters_nbits_int
    neighbors = True
    if args.parameters_not_neighbors_bool:
        neighbors = args.parameters_not_neighbors_bool
    allHsExplicit = False
    if args.parameters_all_hs_explicit_bool:
        allHsExplicit = args.parameters_all_hs_explicit_bool

    # Load Smiles file
    logging.info("Load file")
    H, D = read_csv(args.input_smiles_csv)
    logging.info(f"Header={H}\nD={D.shape}")
    Smiles = np.asarray(list(set(D[:, 7]).union(set(D[:, 9]))))
    logging.info(f"Number of smiles: {len(Smiles)}")

    # Alphabet Signature
    logging.info("Build Alphabet")
    start_time = time.time()
    Alphabet = AlphabetObject(
        radius=radius, nBits=nBits, neighbors=neighbors, allHsExplicit=allHsExplicit
    )
    Alphabet.fill(Smiles, verbose=True)

    logging.info("Save Alphabet Signature")
    Alphabet.save(args.output_signature_npz)
    logging.info(f"CPU time compute Alphabet: {time.time() - start_time:.2f}")
    Alphabet.printout()

    # Alphabet Reaction
    logging.info("Build Alphabet Reaction")
    start_time = time.time()
    D = D[np.transpose(np.argwhere(D[:, 3] == 2 * Alphabet.radius))[0]]
    logging.info(f"Reaction size: {D.shape[0]}")
    Reaction = ReactionObject(Alphabet)
    # Substrate, Product, ID, Diameter, Rule, Score, Direction
    Reaction.fill(D[:, 7], D[:, 9], D[:, 0], D[:, 3], D[:, 5], D[:, 13], D[:, 15])
    logging.info("Save Alphabet Reaction")
    Reaction.save(args.output_reaction_npz)
    logging.info(f"CPU time compute Alphabet: {time.time() - start_time:.2f}")
    Reaction.printout()

    logging.info("End - Alphabet")


P_alphabet = AP_subparsers.add_parser("alphabet", help=_cmd_alphabet.__doc__)
# Input
P_alphabet_input = P_alphabet.add_argument_group("Input")
P_alphabet_input.add_argument(
    "--input-smiles-csv", required=True, help="CSV file with 2 columns"
)
# Output
P_alphabet_output = P_alphabet.add_argument_group("Output")
P_alphabet_output.add_argument(
    "--output-signature-npz",
    required=True,
    help="Output file, Alphabet Signature, .npz",
)
P_alphabet_output.add_argument(
    "--output-reaction-npz", required=True, help="Output file, Alphabet Reaction, .npz"
)
#
# Parameters
P_alphabet_params = P_alphabet.add_argument_group("Parameters")
P_alphabet_params.add_argument(
    "--parameters-radius-int", type=int, default=2, help="Radius value"
)
P_alphabet_params.add_argument(
    "--parameters-nbits-int",
    type=int,
    default=2048,
    help="",
)
P_alphabet_params.add_argument(
    "--parameters-not-neighbors-bool", action="store_true", help="Compute neighbors"
)
P_alphabet_params.add_argument(
    "--parameters-all-hs-explicit-bool", action="store_true", help="Add all HsExplicit"
)


P_alphabet.set_defaults(func=_cmd_alphabet)


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
