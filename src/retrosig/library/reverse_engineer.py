###############################################################################
# This library reverse engineer molecules from signatres or ECFP
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format 
# Authors: Jean-loup Faulon jfaulon@gmail.com
# May 2023
###############################################################################

from library.imports import *

###############################################################################
# Molecular signatures enumeration 
###############################################################################

def SignatureToSignatureMolecule(sig, verbose=False):
# Callable function
# Compute all the possible molecule signatures
# mathing a provided signature
# ARGUMENTS:
# A signature
# RETURNS:
# A set of molecular signatures

    from library.reverse_engineer_utils import GetConstraintMatrices
    from library.diophantine_solver import DiophantineSolver
    
    # get matrices for DiophantineSolver
    AS, NAS, Deg, A, B, C = GetConstraintMatrices(sig, 
                                                  unique=True, 
                                                  verbose=verbose)
    
    # find elementary mode
    R = DiophantineSolver(C, verbose=verbose)
    if R.shape[1] == 0:
        print(f'WARNING DiophantineSover failed at first equation {C.shape}')
        return np.asarray([])
    R = np.unique(R[:NAS.shape[0],:], axis=1)
    R = R[:, np.any(R, axis=0)]
    
    # Constraint matrix = truncated R removing aditional variable
    # constraints + nbr of atom signature in the last column
    nas = np.asarray([-NAS[j] for j in range(len(NAS))]).reshape(-1,1)
    CC = np.concatenate((R, nas), axis=1)
    if verbose==2: print('CC=\n', CC.shape, CC)
    RR = DiophantineSolver(CC, verbose=verbose)
    if verbose==2: print('RR=\n', RR.shape, RR)
    if RR.shape[1] == 0:
        s = 'WARNING DiophantineSover failed at second equation'
        print(f' {s} {C.shape} {CC.shape}')
        return np.asarray([])
    
    # Remove solution not equal to NAS
    Rsol, RRsol, I = {}, {}, 0
    for j in range(RR.shape[1]): 
        rn = RR[RR.shape[0]-1,j] # multiplier matching nbr atom 
        rr = RR[:RR.shape[0]-1,j] # the combination of modes
        #print('----rn, rr', rn, rr)
        r = np.zeros(R.shape[0])
        # Get back the actual atom signatures vector
        for i in range(len(rr)):
            r = r + rr[i] * R[:,i]
            #print('--i, rr[i],R[:,i], r ‚=', i, rr[i], R[:,i], r)
        if rn > 0 and np.sum((r/rn).astype(int) - r/rn) == 0: # rn can divide r
            r = r/rn
        #print('----r, r, NAS', r, r[:NAS.shape[0]], NAS)
        if np.array_equal(r[:NAS.shape[0]], NAS):
            Rsol[I], RRsol[I] = r, rr
            I += 1
    #print('Rsol=', Rsol)
    Rsol = np.asarray(list(Rsol.values()))
    Rsol = np.transpose(np.unique(Rsol, axis=0))
    Rsol = Rsol[:,:NAS.shape[0]]
    RRsol = np.asarray(list(RRsol.values()))
    RRsol = np.unique(Rsol, axis=0)

    if verbose: print(f'RRsol=\n {RRsol.shape} {RRsol}')
    if verbose: print(f'Rsol=\n {Rsol.shape} {Rsol}')

    # Get the molecular signature vector
    # Below need to change

    return np.asarray([sig])

###############################################################################
# Molecules (smiles) enumeration from signature
###############################################################################

def SignatureToMolecule(sig, 
                        max_nbr_solution=float('inf'),
                        max_nbr_recursion=1.0e6,
                        verbose=False):
# Callable function
# Build a molecule matching a provided signature
# ARGUMENTS:
# The signature of a molecule
# max_nbr_solution: maximum nbr of solutions returned 
# max_nbr_recursion constant used in signature_enumerate
# RETURNS:
# The list of smiles

    from library.reverse_engineer_utils import GetConstraintMatrices
    from library.signature_enumerate import EnumerateSignatureToSmiles
    from library.signature_alphabet import SignatureVectorToString

    AS, NAS, Deg, A, B, C = GetConstraintMatrices(sig, 
                                                  unique=False,
                                                  verbose=verbose)
    smi = EnumerateSignatureToSmiles(A, B, AS,
                                     max_nbr_solution=max_nbr_solution,
                                     max_nbr_recursion=max_nbr_recursion,
                                     verbose=verbose)
    return smi

###############################################################################
# Signatures enumeration matching a Morgan vector
###############################################################################

def MorganToSignature(morgansig, Alphabet, 
                      max_nbr_solution=float('inf'),
                      max_nbr_recursion=1.0e5, 
                      verbose=False):
# Callable function
# Compute all possible signature having a the same Morgan vector 
# than the provided one
# ARGUMENTS:
# The Morgan signature of a molecule
# max_nbr_solution: maximum nbr of solutions returned 
# max_nbr_recursion constant used in signature_enumerate
# RETURNS:
# The list of signature matching the Morgan vector of the provided signature

    from library.reverse_engineer_utils import GetConstraintMatrices
    from library.signature_enumerate import EnumerateSignatureToSignature

    # Get Matrices for signatures enumeration
    AS, NAS, Deg, A, B, C = GetConstraintMatrices(morgansig, 
                                                  unique=True, 
                                                  verbose=verbose)
    ASU, OCCU = EnumerateSignatureToSignature(A, B, AS, NAS, 
                Alphabet=Alphabet,
                max_nbr_recursion=max_nbr_recursion,
                max_nbr_solution=max_nbr_solution, 
                verbose=verbose)
    return ASU, OCCU

###############################################################################
# Molecules (smiles) enumeration from Morgan Vector
###############################################################################

def MorganToMolecule(morgansig, Alphabet,
                     max_nbr_solution=float('inf'),
                     max_nbr_recursion=1.0e5, 
                     verbose=False):
# Callable function
# Compute all possible signature having a the same Morgan vector 
# than the provided one
# ARGUMENTS:
# The Morgan signature of a molecule
# max_nbr_solution: maximum nbr of solutions returned 
# max_nbr_recursion constant used in signature_enumerate
# RETURNS:
# The list of signature matching the Morgan vector of the provided signature

    from library.reverse_engineer_utils import GetConstraintMatrices
    from library.signature_enumerate import EnumerateSignatureToSignature
    from library.signature_enumerate import EnumerateSignatureToSmiles
    from library.signature_alphabet import SignatureVectorToString

    # Get Matrices for signatures enumeration
    AS, NAS, Deg, A, B, C = GetConstraintMatrices(morgansig, 
                                                  unique=True, 
                                                  verbose=verbose)
    ASU, OCCU = EnumerateSignatureToSignature(A, B, AS, NAS, 
                Alphabet=Alphabet,
                max_nbr_recursion=max_nbr_recursion,
                max_nbr_solution=max_nbr_solution, 
                verbose=verbose)
    if verbose:
        print(f'Number of signatures: {OCCU.shape[0]}')
    if max_nbr_solution < float('inf'):
        max_nbr_solution = math.ceil((max_nbr_solution / OCCU.shape[0]))
        
    Smi = set()
    for i in range(OCCU.shape[0]):
        sig = SignatureVectorToString(OCCU[i], ASU, verbose=False)
        AS, NAS, Deg, A, B, C = GetConstraintMatrices(sig, 
                                                      unique=False,
                                                      verbose=verbose)
        smi = EnumerateSignatureToSmiles(A, B, AS,
              max_nbr_solution=max_nbr_solution,
              max_nbr_recursion=max_nbr_recursion,
              verbose=verbose)
        Smi = Smi | set(list(smi))
        if len(Smi) > max_nbr_solution:
            return Smi        
    return Smi
