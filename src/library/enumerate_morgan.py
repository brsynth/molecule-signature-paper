###############################################################################
# This library enumerate signatures or molecules matching a morgan vector
#   than the provided atom signature
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format 
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Apr. 2023
###############################################################################

from library.imports import *

class OccurenceSignature:
    # A local object used to enumerate atom signature occurences
    def __init__(self, 
                 AS, # atom signature array
                 C, # constraint matrix
                 occidx, # the atom numbers
                 occmin, # minimum values for occurence vector
                 occmax, # maximum values for occurence vector
                 max_nbr_recursion, # maximum nbr recursion allowed
                 max_nbr_solution=float('inf') # to produce all solutions
                ):
        self.AS = AS
        self.C = C
        self.occidx = occidx
        self.occmin, self.occmax = occmin, occmax
        self.occ = np.copy(occmin)
        self.max_nbr_solution = max_nbr_solution
        self.nbr_recursion = 0
        self.max_nbr_recursion = max_nbr_recursion

    def printout(self, verbose=False):
        if verbose: 
            print(f'AS: {self.AS}')
            print(f'C: {self.C}')
            print(f'occ: {self.occ}')
            print(f'occidx: {self.occidx}')
            print(f'occmin: {self.occmin}')
            print(f'occmax: {self.occmax}')
            print(f'max_nbr_solution: {self.max_nbr_solution}')
            print(f'Estimated number of solutions: {np.prod(self.occmax):.0E}')
        
    def validocc(self, i, new=True, verbose=False):
        last = True if i == self.occ.shape[0]-1 else False
        if last == False:
            last = True if self.occidx[i+1] != self.occidx[i] else False
        if last == False:
            return True
        # i is last element
        I = np.transpose(np.argwhere(self.occidx == self.occidx[i]))[0]
        occsum = np.sum(self.occ[I])    
        if occsum != self.occmax[i]:
            return False 
        else: 
            return True
    
    def endocc(self, i, verbose=False):
        if verbose==2 and \
        self.nbr_recursion % int(self.max_nbr_recursion/10) == 0:
            print(f'...{self.nbr_recursion}')
        self.nbr_recursion += 1
        if self.nbr_recursion > self.max_nbr_recursion:
            if verbose==2: print(f'nbr recursion exceeded for occ')
        if i == self.occ.shape[0]:
            # end recursion check constraints (C) are respected
            # for occ i.e. C.occ = 0
            C = self.C[:,:self.occ.shape[0]]
            valid = True \
            if np.sum(np.absolute(C.dot(self.occ))) % 2 == 0 else False
            if verbose==2: print(f'{valid} occ found at {self.nbr_recursion}')
            return valid
        return False
    
    def nextocc(self, i, verbose=False):
    # Return first occ that can be increased
        j = i
        for j in range(i, self.occ.shape[0]):
            if self.occ[j] < self.occmax[j]:
                break
        return j
    
    def resetocc(self, i, verbose=False):
    # Reset all occ > i
        for j in range(i+1, self.occ.shape[0]):
            self.occ[j] = self.occmin[j]

def EnumerateOcc(OS, index, verbose=False):
# Local function
# Enumerate all possible occ vector within occmin and occmax

    if OS.endocc(index, verbose=verbose):
        if OS.nbr_recursion > OS.max_nbr_recursion:
            return np.asarray([[]])
        else:
            return np.asarray([OS.occ])
    elif index == OS.occ.shape[0]:
        return np.asarray([[]])
    
    # Increase occ and recursive call
    i = OS.nextocc(index)
    OCCi = np.asarray([[]])
    while OS.occ[i] <= OS.occmax[i]:
        if OS.nbr_recursion > OS.max_nbr_recursion:
            return OCCi
        OS.resetocc(i)
        if OS.validocc(i, verbose=verbose):
            occi = EnumerateOcc(OS, index+1, verbose=verbose)
            if occi.shape[1] > 0:
                OCCi  = np.concatenate((OCCi, occi), axis=0) \
                if OCCi.shape[1] > 0 else np.copy(occi)
                if OCCi.shape [0] >= OS.max_nbr_solution:
                    return OCCi
        OS.occ[i] += 1
            
    return OCCi

###############################################################################
# Callable functions
###############################################################################

def EnumerateSignatureFromMorgan(morgan, Alphabet, 
                                 max_nbr_recursion=1.0e5,
                                 max_nbr_solution=float('inf'),
                                 verbose=False):
# Callable function
# Compute all possible signature having a the same Morgan vector 
# than the provided one
# ARGUMENTS:
# The Morgan vector
# max_nbr_solution: maximum nbr of solutions returned 
# max_nbr_recursion constant used in signature_enumerate
# RETURNS:
# The list of signature matching the Morgan vector of the provided signature

    from library.enumerate_utils import GetConstraintMatrices
    from library.enumerate_utils import UpdateConstraintMatrices
    from library.signature_alphabet import SignatureAlphabetFromMorganBit

    # Get alphabet signatures in AS along with their
    # minimum and maximum occurence numbers
    AS, MIN, MAX, IDX, I = {}, {}, {}, {}, 0
    for i in range(morgan.shape[0]):
        if morgan[i] == 0:
            continue
        # get all signature in Alphabet having MorganBit = i
        sig = SignatureAlphabetFromMorganBit(i, Alphabet)
        if verbose==2: print(f'MorganBit {i}, Signature {len(sig)}, {sig}')
        maxi = morgan[i]
        mini = 0 if len(sig) > 1 else maxi
        for j in range(len(sig)):
            AS[I], MIN[I], MAX[I], IDX[I] = sig[j], mini, maxi, i
            I += 1

    # Get Matrices for enumeration
    AS = np.asarray(list(AS.values()))
    IDX = np.asarray(list(IDX.values()))
    MIN = np.asarray(list(MIN.values()))
    MAX = np.asarray(list(MAX.values()))
    Deg = np.asarray([len(AS[i].split('.'))-1 for i in range(AS.shape[0])])
    n1 = AS.shape[0]
    AS, IDX, MIN, MAX, Deg, C = \
    UpdateConstraintMatrices(AS, IDX, MIN, MAX, Deg, verbose=verbose)
    n2 = AS.shape[0]
    if verbose:
        print(f'AS reduction {n1}, {n2}')
    
    # Enumerate all possible vectors occ for the occurence
    # numbers of the signature in AS
    OS  = OccurenceSignature(AS, C, IDX, MIN, MAX, 
                             max_nbr_recursion, 
                             max_nbr_solution=max_nbr_solution)
    OS.printout(verbose=verbose)
    OCC = EnumerateOcc(OS, 0, verbose=verbose)
    return AS, OCC

###############################################################################
# Molecules (smiles) enumeration from Morgan Vector
###############################################################################

def EnumerateMoleculeFromMorgan(morgan, Alphabet,
                                max_nbr_solution=float('inf'),
                                repeat=10,
                                verbose=False):
# Callable function
# Compute all possible signature having a the same Morgan vector 
# than the provided one
# ARGUMENTS:
# The Morgan vector
# max_nbr_solution: maximum nbr of solutions returned 
# max_nbr_recursion: constant used in signature_enumerate
# nbr_component: nbr connected components
# RETURNS:
# The list of molecule matching the Morgan vector of the provided signature

    from library.enumerate_utils import GetConstraintMatrices
    from library.signature_alphabet import SignatureVectorToString
    from library.enumerate_signature import EnumerateMoleculeFromSignature
    
    # Enumerate signatures from Morgan vector
    max_nbr_recursion=1.0e5 # 1.0e5 found after trials
    AS, OCC = EnumerateSignatureFromMorgan(morgan, Alphabet, 
              max_nbr_recursion=max_nbr_recursion,
              max_nbr_solution=max_nbr_solution,
              verbose=verbose)
    if verbose:
        print(f'Number of signatures: {OCC.shape[0]}')
    if max_nbr_solution < float('inf'):
        max_nbr_solution = math.ceil((max_nbr_solution / OCC.shape[0]))

    # Enumerate molecules from signatures 
    SMI = set()
    for i in range(OCC.shape[0]):
        sig = SignatureVectorToString(OCC[i], AS, verbose=False)
        morgansig = MorganVectorFromSignature(sig, Alphabet)
        if np.array_equal(morgan, morgansig) == False:
            continue
        Smi = set()
        for r in range(repeat):
            smi = EnumerateMoleculeFromSignature(sig, sig, Alphabet,
                                             max_nbr_solution=max_nbr_solution,
                                             verbose=False)
            Smi = Smi | set(list(smi))
            if len(Smi):
                break
                
        SMI = SMI | Smi
        if len(SMI) > max_nbr_solution:
            break
            
    # retain solutions having a morgan = provided morgan
    SMImorgan = set()
    for smi in SMI:
        if smi == '':
            continue
        if '.' in smi:
            continue
        sigsmi, mol, smisig = SignatureFromSmiles(smi, Alphabet)
        morgansmi = MorganVe ctorFromSignature(sigsmi, Alphabet)
        if np.array_equal(morgan, morgansig):
            SMImorgan.add(smisig)
        
    return SMI
