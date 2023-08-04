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

class SignatureAlphabet:
    def __init__(self, 
                 radius=2,
                 nBits=0,
                 neighbors=False, 
                 splitcomponent=False, 
                 useRange=False, 
                 isomericSmiles=False, 
                 formalCharge=True, 
                 atomMapping=False, 
                 kekuleSmiles=False, 
                 allHsExplicit=False, 
                 maxvalence=4,
                 Dict={}):
        self.filename=''
        self.radius=radius # radius signatures are computed
        # the number of bits in Morgan vector (defaut 0 = no vector)
        self.nBits=nBits 
        # when True signature is computed on neighbors with radius-1
        self.neighbors=neighbors
        # when True the signature is computed for each molecule 
        self.splitcomponent=splitcomponent 
        # when True the signature is computed from 0 to radius
        self.useRange=useRange 
        # include information about stereochemistry
        self.isomericSmiles=isomericSmiles 
        # Remove charges on atom when False. Defaults to False
        self.formalCharge=formalCharge
        # Remove atom mapping when False
        self.atomMapping=atomMapping 
        self.kekuleSmiles=kekuleSmiles
        # if true, all H  will be explicitly in signatures
        self.allHsExplicit=allHsExplicit
        self.maxvalence=maxvalence # for all atoms
        # the alphabet dictionary keys = atom signature, values = index
        self.Dict=Dict 
        
    def fill(self, Smiles, verbose=False):
        # Fill signature dictionary
        # Smiles: an array of Smiles
        from library.utils import VectorToDic
        Dict = set()
        start_time = time.time()
        for i in range(len(Smiles)):  
            signature, _, _ = SignatureFromSmiles(Smiles[i], self, verbose=verbose)
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
                            splitcomponent=self.splitcomponent,
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
        print(f'splitcomponent: {self.splitcomponent}')
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
    Alphabet=SignatureAlphabet()
    Alphabet.filename=filename
    Alphabet.Dict = VectorToDic(load['Dict'])
    # Flags to compute signatures
    Alphabet.radius=int(load['radius'])
    Alphabet.nBits=int(load['nBits'])
    Alphabet.maxvalence=int(load['maxvalence'])
    Alphabet.neighbors=bool(load['neighbors'])
    Alphabet.splitcomponent=bool(load['splitcomponent'])
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
# Signature utilities
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

def GetSignatureInfo(sig, Alphabet=None, unique=False, verbose=False):
# Callable function
# ARGUMENTS:
# sig: a signature cf. signature.py for signature format 
# Alphabet: cf. SignatureAlphabet object
# unique: a flag indicating if the atom signature list 
#         must contain only unique atom signatures
# RETURNS:
# AS: an array of atom signature
# NAS, Deg: the occurence nbr (degree) of each atom signature

    if Alphabet != None:
        sig = SignatureVectorToString(sigV, Alphabet.Dict, verbose=verbose)
    LAS = sig.split(' ')
    LAS.sort()
    AS = list(set(LAS)) if unique else LAS
    AS.sort()
    AS = np.asarray(AS)
    N = AS.shape[0] # nbr of atoms 
    NAS, Deg, M = {}, {}, 0
    for i in range(N):
        NAS[i] = LAS.count(AS[i]) if unique else 1
        Deg[i] = len(AS[i].split('.'))-1
        M = M + Deg[i]
    Ncycle = int(M/2-N+1)
    NAS = np.asarray(list(NAS.values()))    
    Deg = np.asarray(list(Deg.values()))

    if verbose:
        print(f'Nbr atoms, bonds, Cycle, {N}, {int(M/2)}, {Ncycle}')
        print(f'LAS, {len(AS)}')
        for i in range(len(LAS)):
            print(f'- {i}: {LAS[i]}')
        print(f'Deg {Deg}, {len(Deg)}')
        print(f'NAS, {NAS}, {len(NAS)}')

    return AS, NAS, Deg

###############################################################################
# Signature string or vector computed from smiles
###############################################################################

def SignatureFromSmiles(smiles, Alphabet, string=True, verbose=False):
# Callable function
# Get a sanitized signature vector for the provided smiles
# A local routine to make sure all signatures are standard
# ARGUMENTS:
# smiles: a smiles string (can contain several molecuel spareted by '.'
# string: return a sring when True else return a vector
# Alphabet: cf. SignatureAlphabet object
# RETURNS:
# molecule: an array of RDKit molecule 
# signature: the signature string or vector, 
#            the molecule and the corresponding smiles


    from library.signature import SanitizeMolecule, GetMoleculeSignature

    S = smiles.split('.') if Alphabet.splitcomponent else [smiles]
    signature, temp, molecule, smiles = '', [], [], ''
    for i in range(len(S)):
        mol = Chem.MolFromSmiles(S[i])
        mol, smi = SanitizeMolecule(mol, 
                                    kekuleSmiles=Alphabet.kekuleSmiles,
                                    allHsExplicit=Alphabet.allHsExplicit,
                                    isomericSmiles=Alphabet.isomericSmiles,
                                    formalCharge=Alphabet.formalCharge,
                                    atomMapping=Alphabet.atomMapping,
                                    verbose=verbose)
        if mol == None:
            continue
        sig = GetMoleculeSignature(mol,
                                   radius=Alphabet.radius,
                                   neighbors=Alphabet.neighbors,
                                   nBits=Alphabet.nBits,
                                   useRange=Alphabet.useRange,
                                   isomericSmiles=Alphabet.isomericSmiles,
                                   allHsExplicit=Alphabet.allHsExplicit,
                                   verbose=verbose)
        if sig != '':
            temp.append(sig)
            molecule.append(mol)
            smiles = f'{smiles}.{smi}' if len(smiles) else smi
        
    if len(temp) < 1:
        if string == False and Alphabet.Dict != {}:
            return [], molecule, ''
        else:
            return '', molecule, ''
    
    temp = sorted(temp)
    signature = ' . '.join(sig for sig in temp) 
        
    if string == False and Alphabet.Dict != {}:
        signature = SignatureStringToVector(signature, Alphabet.Dict, 
                                            verbose=verbose)

    return signature, molecule, smiles

###############################################################################
# Morgan Vector of a signature 
###############################################################################

def MorganVectorString(morgan):
    s = ''
    for i in range(len(morgan)):
        if morgan[i]:
            s = f'{i}:{morgan[i]}' if s == '' else s + f', {i}:{morgan[i]}'
    return s

def MorganBitFromSignature(sa, verbose=False):
# Callable function
# Get the Morgan bit for an atom signature
# ARGUMENTS:
# sa : atom signature (string)
# RETURNS:
# The Morgan bit

    if len(sa.split(',')) == 0:
        if verbose:
            print(f'Error signature does not include Morgan bits')
        return -1
    return int(sa.split(',')[0])

def MorganVectorFromSignature(signature, Alphabet, verbose=False):
# Callable function
# Get the Morgan vector for a signature
# ARGUMENTS:
# signature of molecule (string)
# Alphabet: the alphabet of atom signatures
# RETURNS:
# A Morgan vector of size nBits

    MorganVector = np.zeros(Alphabet.nBits) 
    for sa in signature.split(' '): # separate atom signatures
        mbit = MorganBitFromSignature(sa, verbose=verbose)
        if mbit < 0:
            if verbose:
                print(f'Error signature does not include Morgan bits')
            return MorganVector
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

    from library.signature import GetMoleculeSignature

    Signatures = []
    if Alphabet.Dict == {}:
        print(f'WARNING Empty Alphabet')
        return Signatures
    if MorganBit > Alphabet.nBits:
        print(f'WARNING MorganBit {MorganBit} exceeds nBits {Alphabet.nBits}')
        return Signatures
    for sig in Alphabet.Dict.keys():
        mbit = int(sig.split(',')[0])
        if mbit == MorganBit:
            Signatures.append(sig)
    
    return Signatures
