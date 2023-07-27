###############################################################################
# This library compute save and load Alphabet of atom signatures
# The library also provides functions to compute a molecule signature as a
# vector of occurence numbers over an Alphabet, and a Morgan Fingerprint
# from a molecule signature string.
# Note that when molecules have several connected components
# the individual molecular signatures string are separated by ' . '
# Authors: Jean-loup Faulon jfaulon@gmail.com
# March 2023
###############################################################################

from library.imports import *

###############################################################################
# Alphabet callable object
###############################################################################

class AlphabetObject:
    def __init__(self, 
                 radius=2,
                 nBits=0,
                 neighbors=False, 
                 split=False, 
                 useRange=False, 
                 isomericSmiles=False, 
                 formalCharge=True, 
                 atomMapping=False, 
                 kekuleSmiles= True, 
                 allHsExplicit=False, 
                 maxvalence=4,
                 Dict={}):
        self.filename=''
        self.radius=radius # radius signatures are computed
        self.nBits=nBits # the number of bits in Morgan vector (defaut 0 = no vector)
        self.neighbors=neighbors # when True signature is computed on neighbors with radius-1
        self.split=split # when True the signature is computed for each molecule within a smiles string
        self.useRange=useRange # when True the signature is computed from 0 to radius 
        self.isomericSmiles=isomericSmiles # include information about stereochemistry
        self.formalCharge=formalCharge # Remove charges on atom when False. Defaults to False
        self.atomMapping=atomMapping # atoRemove atom mapping when False
        self.kekuleSmiles=kekuleSmiles # if true, H are added
        self.allHsExplicit=allHsExplicit # if true, all H  will be explicitly in signatures
        self.maxvalence=maxvalence # for all atoms
        self.Dict=Dict # the alphabet dictionary keys = atom signature, values = index
        
    def fill(self, Smiles, verbose=False):
        # Fill signature dictionary
        # Smiles: an array of Smiles
        from library.utils import VectorToDic
        Dict = set()
        start_time = time.time()
        for i in range(len(Smiles)):  
            signature, _ = SignatureFromSmiles(Smiles[i], self, verbose=verbose)
            if len(signature) == 0:
                print(f'WARNING no signature for molecule {i} {Smiles[i]}')
                continue  
            for sig in signature.split(' . '): # separate molecules
                for s in sig.split(' '): # separate atom signatures
                    Dict.add(s)
            if verbose:
                if i % 1000 == 0:
                    print(f'... processing alphabet iteration: {i} \
                          size: {len(list(Dict))} \
                          time: {(time.time()-start_time)}')
                    start_time = time.time()
        self.Dict = VectorToDic(list(Dict))

    def save(self, filename):
        filename = filename+'.npz' if filename.find('.npz') == -1 else filename
        np.savez_compressed(filename, 
                            filename=filename,
                            radius=self.radius,
                            nBits=self.nBits,
                            neighbors=self.neighbors,
                            split=self.split,
                            useRange=self.useRange,
                            isomericSmiles=self.isomericSmiles,
                            formalCharge=self.formalCharge,
                            atomMapping=self.atomMapping,
                            kekuleSmiles=self.kekuleSmiles,
                            allHsExplicit=self.allHsExplicit,
                            maxvalence=self.maxvalence,
                            Dict=list(self.Dict.keys()))

    def printout(self):
        print(f'filename: {self.filename}')
        print(f'radius: {self.radius}')
        print(f'nBits: {self.nBits}')
        print(f'neighbors: {self.neighbors}')
        print(f'split: {self.split}')
        print(f'useRange:{self.useRange}')
        print(f'isomericSmiles: {self.isomericSmiles}')
        print(f'formalCharge: {self.formalCharge}')
        print(f'atomMapping: {self.atomMapping}')
        print(f'kekuleSmiles: {self.kekuleSmiles}')
        print(f'allHsExplicit: {self.allHsExplicit}')
        print(f'maxvalence: {self.maxvalence}')
        print(f'alphabet length: {len(self.Dict.keys())}')

def LoadAlphabet(filename, verbose=False):
    from library.utils import VectorToDic
    filename = filename+'.npz' if filename.find('.npz') == -1 else filename
    load = np.load(filename, allow_pickle=True)
    Alphabet=AlphabetObject()
    Alphabet.filename=filename
    Alphabet.Dict = VectorToDic(load['Dict'])
    # Flags to compute signatures
    Alphabet.radius=int(load['radius'])
    Alphabet.nBits=int(load['nBits'])
    Alphabet.maxvalence=int(load['maxvalence'])
    Alphabet.neighbors=bool(load['neighbors'])
    Alphabet.split=bool(load['split'])
    Alphabet.useRange=bool(load['useRange'])
    Alphabet.isomericSmiles=bool(load['isomericSmiles'])
    Alphabet.formalCharge=bool(load['formalCharge'])
    Alphabet.atomMapping=bool(load['atomMapping'])
    Alphabet.kekuleSmiles=bool(load['kekuleSmiles'])
    Alphabet.allHsExplicit=bool(load['allHsExplicit'])
    if verbose:
        Alphabet.printout()
    return Alphabet

###############################################################################
# Signature as a vector provided an alphabet
###############################################################################

def SignatureStringToVector(signature, Dict, verbose=False):
# Callable function
# ARGUMENTS:
# Signature: a string signature
# Dict: a dictionary of unique atom signatures
# RETURNS:
# SigV: an array of Dict size where
#       SigV[i] is the occurence number of the atom signatures 
#       in sig

    signatureV = np.zeros(len(Dict.keys()))
    for sig in signature.split(' . '): # separate molecules
        for s in sig.split(' '): # separate atom signatures
            try:
                index = Dict[s]
                signatureV[index] += 1
            except:
                print(f'Error atom signature not found in Alphabet {s}')
                continue # !!!
                sys.exit('Error') 
    return signatureV

def SignatureVectorToString(sigV, Dict, verbose=False):
# Callable function
# ARGUMENTS:
# sigV: a Vector signature
# sig[i] is the occurence number of the atom signatures 
# Dict a dictionary, list or array of atom signatures
# RETURNS:
# sig: a string signature (without split between molecule)

    I, sig = np.transpose(np.argwhere(sigV != 0))[0], ''
    if isinstance(Dict, (dict)):
        A = list(Dict.keys())
    else:
        A = list(Dict)
    for i in I:
        for k in range(int(sigV[i])):
            sig = A[int(i)] if sig == '' else sig + ' ' + A[i]
    return sig

###############################################################################
# Signature string or vector computed from smiles
###############################################################################

def SignatureFromSmiles(smiles, Alphabet, 
                        string=True, verbose=False):
# Callable function
# Get a sanatized signature vector for the provided smiles
# A local routine to make sure all signatures are standard
# ARGUMENTS:
# smiles: a smiles string (can contain several molecuel spareted by '.'
# string: return a sring when True else return a vector
# Alphabet: the Alphabet of atom signature used to compute signature vector
# RETURNS:
# molecule: an array of RDKit molecule 
# signature: the signature string (string=True) or vector

    from library.signature import SanatizeMolecule
    from library.signature import GetMoleculeSignature

    S = smiles.split('.') if Alphabet.split else [smiles]
    signature, temp, molecule = '', [], []
    for i in range(len(S)):
        mol = Chem.MolFromSmiles(S[i])
        mol = SanatizeMolecule(mol, 
                               isomericSmiles=Alphabet.isomericSmiles, 
                               formalCharge=Alphabet.formalCharge,
                               atomMapping=Alphabet.atomMapping,
                               verbose=verbose)
        sig = GetMoleculeSignature(mol,
                                   radius=Alphabet.radius,
                                   nBits=Alphabet.nBits,
                                   neighbors=Alphabet.neighbors,
                                   useRange=Alphabet.useRange,
                                   isomericSmiles=Alphabet.isomericSmiles,
                                   kekuleSmiles=Alphabet.kekuleSmiles,
                                   allHsExplicit=Alphabet.allHsExplicit,
                                   verbose=verbose)
        if sig != '':
            temp.append(sig)
            molecule.append(mol)
        
    if len(temp) < 1:
        if string == False and Alphabet.Dict != {}:
            return [], molecule
        else:
            return '', molecule
    
    temp = sorted(temp)
    signature = ' . '.join(sig for sig in temp) 
        
    if string == False and Alphabet.Dict != {}:
        signature = SignatureStringToVector(signature, Alphabet.Dict, 
                                            verbose=verbose)
    return signature, molecule


###############################################################################
# Morgan Vector of a signature 
###############################################################################

def MorganBitFromSignature(sa, Alphabet, verbose=False):
# Callable function
# Get the Morgan bit for an atom signature
# ARGUMENTS:
# sa : atom signature (string)
# Alphabet: the alphabet of atom signatures
# RETURNS:
# The Morgan bit

    if Alphabet.Dict == {}:
        print(f'Error Empty Alphabet')
        sys.exit('Error') 
    if Alphabet.nBits == 0:
        print(f'Error signature does not include Morgan bits')
        sys.exit('Error')       
    return int(sa.split(',')[2])

def MorganVectorFromSignature(signature, Alphabet, verbose=False):
# Callable function
# Get the Morgan vector for a signature
# ARGUMENTS:
# signature of molecule (string)
# Alphabet: the alphabet of atom signatures
# RETURNS:
# A Morgan vector of size nBit

    MorganVector = np.zeros(Alphabet.nBits)
    if Alphabet.Dict == {}:
        print(f'Error Empty Alphabet')
        sys.exit('Error') 
    if Alphabet.nBits == 0:
        print(f'Error signature does not include Morgan bits')
        sys.exit('Error')       
    for sa in signature.split(' '): # separate atom signatures
        mbit = MorganBitFromSignature(sa, Alphabet, verbose=verbose)
        MorganVector[mbit] += 1
            
    return MorganVector

def SignatureAlphabetFromMorganBit(MorganBit, Alphabet, verbose=False):
# Callable function
# Get all signatures in Alphabet having the provided Morgan bit
# ARGUMENTS:
# Morgan bit: an int in [0, nBits]
# Alphabet: the alphabet of atom signatures
# RETURNS:
# A list of signature having the provided Morgan bit

    from library.signature import SanatizeMolecule
    from library.signature import GetMoleculeSignature

    Signatures = []
    if Alphabet.Dict == {}:
        print(f'WARNING Empty Alphabet')
        return Signatures
    if MorganBit > Alphabet.nBits:
        print(f'WARNING MorganBit {MorganBit} exceeds nBits {Alphabet.nBits}')
        return Signatures
    for sig in Alphabet.Dict.keys():
        mbit = int(sig.split(',')[2])
        if mbit == MorganBit:
            Signatures.append(sig)
    
    return Signatures
