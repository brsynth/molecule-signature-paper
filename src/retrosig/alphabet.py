from library.imports import *
from library.utils import read_csv
from library.signature_alphabet import AlphabetObject, LoadAlphabet

# Parameters
radius = 2
nBits = 2048
neighbors = True
allHsExplicit = False
ext = "_N" if neighbors else ""
ext += "H" if allHsExplicit else ""
ext = ext + "_" + str(radius) + "_" + str(nBits)
file_smiles = "./rules/retrorules_rr02_flat_all"
file_alphabet = file_smiles + "_Alphabet" + ext

# Load Smiles file
H, D = read_csv(file_smiles)
print(f"Header={H}\nD={D.shape}")
Smiles = np.asarray(list(set(D[:, 7]).union(set(D[:, 9]))))
print(f"Number of smiles: {len(Smiles)}")

# Get save and load Alphabet
start_time = time.time()
Alphabet = AlphabetObject(
    radius=radius, nBits=nBits, neighbors=neighbors, allHsExplicit=allHsExplicit
)
Alphabet.fill(Smiles, verbose=True)
Alphabet.save(file_alphabet)
Alphabet = LoadAlphabet(file_alphabet)
print(f"CPU time compute Alphabet: {time.time() - start_time:.2f}")
Alphabet.printout()

# Reactions

from library.imports import *
from library.utils import read_csv
from library.signature_alphabet import AlphabetObject, LoadAlphabet
from library.one_step_retro import ReactionObject, LoadReaction

# Parameters
file_smiles = "./rules/retrorules_rr02_flat_all"
file_alphabet = "./rules/retrorules_rr02_flat_all_Alphabet_N_2_2048"
file_reaction = file_alphabet + "_Reaction"

# Load files
Alphabet = LoadAlphabet(file_alphabet)
Alphabet.printout()
H, D = read_csv(file_smiles)
print(f"Header={H}\nD={D.shape}")

#  Compute Reaction to restricted diameter
start_time = time.time()
D = D[np.transpose(np.argwhere(D[:, 3] == 2 * Alphabet.radius))[0]]
print(f"Reaction size: {D.shape[0]}")
Reaction = ReactionObject(Alphabet)
# Substrate, Product, ID, Diameter, Rule, Score, Direction
Reaction.fill(D[:, 7], D[:, 9], D[:, 0], D[:, 3], D[:, 5], D[:, 13], D[:, 15])
Reaction.save(file_reaction)
Reaction = LoadReaction(file_reaction)
print(f"CPU time compute Alphabet: {time.time() - start_time:.2f}")
Reaction.printout()
