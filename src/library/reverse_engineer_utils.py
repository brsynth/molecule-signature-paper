###############################################################################
# This library compute the matrices necessary for the diophantine solver 
# and the signature enumeration methods
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format 
# Authors: Jean-loup Faulon jfaulon@gmail.com
# May 2023
###############################################################################

from library.imports import *

###############################################################################
# Local functions
###############################################################################

def GetAtomSignatureRootNeighbors(asig):
# Local function
# ARGUMENTS:
# asig: atom signature 
# RETURNS:
# The signature of the root (no type)
# The array of neighbor signature
    asig0, asign = asig.split('.')[0], asig.split('.')[1:]
    return asig0, asign 

def GetBondSignatureOccurence(bsig, asig):
# Local function
# ARGUMENTS:
# bsig bond signature (format: as1^bondtype^as2)
# asig: atom signature 
# RETURNS:
# The left (as1) and right (as2) atom signatures 
# of the provided bond signature (bsig) and 
# the occurence numbers (occ1, occ2) of as1 and as2 in asig (neighbor)

    as1, as2 = bsig.split('^')[0], bsig.split('^')[2]
    btype = bsig.split('^')[1]
    asig0, asign = GetAtomSignatureRootNeighbors(asig)
    asig1, asig2 = btype +'|'+ as1, btype +'|'+ as2
    occ1, occ2 = asign.count(asig1), asign.count(asig2)
    return asig0, as1, as2, occ1, occ2 

def ConstraintMatrix(AS, BS, Deg, verbose=False):
# Local function
# ARGUMENTS:
# cf. GetConstraintMatrices
# RETURNS:
# Constraints between bond and atom signatures
# cf. C.J. Churchwell et al. 
# Journal of Molecular Graphics and Modelling 22 (2004) 263–273
# for further details

    C = np.zeros((BS.shape[0],AS.shape[0]))
    
    # Constraints between bond and atom signatures
    for i in range(BS.shape[0]):
        for j in range(AS.shape[0]):
            asj, bs1, bs2, occ1, occ2 = GetBondSignatureOccurence(BS[i], AS[j])
            if bs1 == asj:
                C[i,j] = occ2
            elif bs2 == asj and bs2 != bs1:
                C[i,j] = -occ1
        if bs1 == bs2:
            # adding even-valued column variable
            C = np.concatenate((C, np.zeros((C.shape[0],1))), axis=1)
            C[i,-1] = -2
    if verbose==2:
        print(f'Bond constraint: {C.shape},\n{C}')

    # The graphicality equation Sum_deg (deg-2)n_deg = 2z  - 2
    # Two cases:
    # Sum_deg (deg-2)n_deg < 0 <=> Sum_deg (deg-2)n_deg + 2z = 0
    #   here max (Z) must be 1, otherwise molecule cannot be connected
    # Sum_deg (deg-2)n_deg > 0 <=> Sum_deg (deg-2)n_deg - 2z = 0
    #   here max (Z) is bounded by Natom
    C = np.concatenate((C, np.zeros((1,C.shape[1]))), axis=0)
    for i in range(AS.shape[0]):        
        C[-1,i] = Deg[i]-2
    C = np.concatenate((C, np.zeros((C.shape[0],1))), axis=1)
    C[-1,-1] = 2
    C = np.concatenate((C, np.zeros((C.shape[0],1))), axis=1)
    C[-1,-1] = -2
    if verbose==2:
        print(f'Graphicality constraint: {C.shape},\n{C}')

    return C

###############################################################################
# Get the matrices necessary for the diophantine solver and the enumeration
###############################################################################

def AtomSignature(sig, unique=False, verbose=False):
# local function
# ARGUMENTS:
# sig: a signature computed with neighbor=True
# cf. signature.py for signature format with neighbor=True
# unique: a flag indicating if the atom signature list 
# must contain only unique atom signatures
# RETURNS:
# AS: an array of atom signature
# NAS, Deg: the occurence nbr (degree) of each atom signature

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
        print(f'LAS, {LAS}, {len(AS)}')
        print(f'AS {AS}, {len(AS)}')
        print(f'Deg {Deg}, {len(Deg)}')
        print(f'NAS, {NAS}, {len(NAS)}')

    return AS, NAS, Deg

def BondMatrices(AS, NAS, Deg, verbose=False):
# Local function
# ARGUMENTS:
# AS: an array of atom signature
# NAS, Deg: the occurence nbr (degree) of each atom signature
# RETURNS:
# BS: an array of bond signatures
# B: an array of bond candidate
#    The last row indicate is used during enumeration
#    and filled with 0 at initialization

    N, K = AS.shape[0], np.max(Deg) 
    unique = False if np.max(NAS) == 1 else True
            
    # Fill ABS, BBS (temp arrays used to find compatible bonds)
    ABS, BBS = [], []
    for i in range(N):
        asig0, asign = GetAtomSignatureRootNeighbors(AS[i])    
        for k in range(K):
            if k < len(asign):
                btype = asign[k].split('|')[0] # bond type
                asigk = asign[k].split('|')[1] # neighbor signature
                ABS.append(btype+'^'+asig0) # type + root signature
                BBS.append(btype+'^'+asigk) # type + neighbor signature
            else:
                ABS.append('')
                BBS.append('')   
    ABS, BBS = np.asarray(ABS), np.asarray(BBS)
    
    # Fill B (bond candidate matrix) and BS (bond signature)
    B, BS = np.zeros((N*K+1, N*K)), []
    B[N*K] = np.zeros(N*K)
    for n in range(N):
        for k in range(K):
            i = n * K + k
            bsi = BBS[i]
            if bsi == '':
                break
            J = np.transpose(np.argwhere(ABS == bsi))[0]
            for j in J:
                if BBS[j] == ABS[i]: 
                    ai, aj = int(i/K), int(j/K)
                    if ai == aj and NAS[ai] < 2:
                        continue # cannot bind an atom to itself
                    B[i,j], B[N*K,i], B[N*K,j] = 1, 1, 1 
                    bt = ABS[i].split('^')[0]
                    si = ABS[i].split('^')[1]
                    sj = ABS[j].split('^')[1]
                    bs = si + '^' + bt + '^' + sj if si < sj else \
                         sj + '^' + bt + '^' + si
                    BS.append(bs)
    BS = list(set(BS)) if unique else BS
    BS.sort()
    BS = np.asarray(BS)

    return B, BS

###############################################################################
# Callable functions
###############################################################################

def GetConstraintMatrices(sig, unique=False, verbose=False):
# Callable function
# ARGUMENTS:
# sig: a molecule signature
# unique: a flag indicating if the atom signature list 
# RETURNS:
# AS: an array of atom signature
# NAS, Deg: the occurence nbr (degree) of each atom signature
# A:  an empty adjacency matrix between the atoms of the molecule
#     with diagonal = atom degree
# B:  an adjacency matrix between the bond candidates
#     of the molecule. The last row indicate is used during enumeration
#     and filled with 0 at initialization
# C:  a constraint matrix between bond signature (row) and 
#     atom signature (columns)

    AS, NAS, Deg = AtomSignature(sig, unique=unique, verbose=verbose)
    N, K = AS.shape[0], np.max(Deg) 
    
    # Fill A (diag = degree, 0 elsewhere)
    A = np.zeros((N, N))
    for i in range(N):
        A[i,i] = Deg[i]
              
    # Get B (bond candidate matrix) and BS (bond signature)
    B, BS = BondMatrices(AS, NAS, Deg, verbose=verbose)  
    
    # Get constraint matrices
    C = ConstraintMatrix(AS, BS, Deg, verbose=verbose)
   
    if verbose: 
        print(f'A {A.shape}, B {B.shape} BS {BS.shape}, C {C.shape}')
    if verbose==2: 
        print(f'A\n {A} \nB\n {B} \nBS\n {BS} \nC\n {C}')
             
    return AS, NAS, Deg, A, B, C

def UpdateConstraintMatrices(AS, IDX, MIN, MAX, Deg, verbose=False):
# Callable function
# Same as above but AS, NAS and Deg are given and
# we remove from AS and all matrices for atoms that cannot be connected
# ARGUMENTS:
# AS: an array of atom signature
# IDX, MIN, MAX, Deg: atom index, min and max atom occurence and degree
# RETURNS:
# Updated AS, IDX, MIN, MAX, Deg and Constraint matrix

    # remove from AS atoms that cannot be bounded
    N, K, I = AS.shape[0], np.max(Deg), [] 
    B, BS = BondMatrices(AS, MAX, Deg, verbose=verbose) 
    for i in range(N):
        keep = True
        for k in range(Deg[i]):
            if np.sum(B[i * K + k]) == 0:
                keep = False
                break
        if keep:
            I.append(i)
    AS, IDX = AS[I], IDX[I]
    MIN, MAX, Deg = MIN[I], MAX[I], Deg[I]
    N, K = AS.shape[0], np.max(Deg) 
                  
    # Get B (bond candidate matrix) and BS (bond signature)
    B, BS = BondMatrices(AS, MAX, Deg, verbose=verbose)  
    
    # Get constraint matrices
    C = ConstraintMatrix(AS, BS, Deg, verbose=verbose)
   
    if verbose: 
        print(f'AS {AS.shape} C {C.shape}')
    if verbose==2: 
        print(f'AS\n {AS} \nC\n {C.shape}')
             
    return AS, IDX, MIN, MAX, Deg, C

            

