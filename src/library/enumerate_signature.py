###############################################################################
# This library enumerate molecules from signatures or morgan vector
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format 
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Apr. 2023
###############################################################################

from library.imports import *

###############################################################################
# MolecularGraph local object used for smiles enumeration from signature
###############################################################################

def AtomSignatureMod(sa):
# Local function
# return rsa, a modified atom signature where '.' is change by '_'
# and '|' is change by '§'
    rsa = copy.copy(sa)
    rsa = rsa.replace(".", "_" )
    rsa = rsa.replace("|", "§" )
    return rsa

class MolecularGraph:
    # A local object used to enumerate molecular graphs
    # for atom signatures or molecules
    def __init__(self, 
                 A, # Adjacency matrix
                 B, # Bond matrix
                 SA, # Atom signature
                 Alphabet, # SignatureAlphabet object
                 max_nbr_recursion=1.0e6, # Max nbr of recursion
                 ai=-1, # Current atom nbr used when enumerating signature up
                 max_nbr_solution=float('inf'), # to produce all solutions
                 nbr_component=1 # nbr connected components
                ):
        
        def AtomicNumCharge(sa):
        # return the atomic number of the root of sa
            sa = sa.split('.')[0] # the root
            sa = sa.split(',')[1] if len(sa.split(',')) > 1 else sa
            m = Chem.MolFromSmiles(sa)
            for a in m.GetAtoms():
                if a.GetAtomMapNum() == 1:
                    return a.GetAtomicNum(), a.GetFormalCharge()
            return -1, 0
        
        self.A, self.B, self.SA, self.Alphabet = A, B, SA, Alphabet
        self.max_nbr_solution = max_nbr_solution
        self.M = self.B.shape[1] # number of bounds
        self.K = int(self.B.shape[1] / self.SA.shape[0]) # nbr of bound/atom
        self.ai = ai # current atom for which signature is expanded
        self.nbr_recursion = 0 # Nbr of recursion
        self.max_nbr_recursion = max_nbr_recursion
        self.nbr_component = nbr_component
        rdmol = Chem.Mol()
        rdedmol = Chem.EditableMol(rdmol)
        for sa in self.SA:
            num, charge = AtomicNumCharge(sa)
            if num == -1:
                print(sa)
            rdatom = Chem.Atom(num)
            rdatom.SetFormalCharge (int(charge))
            rdedmol.AddAtom(rdatom)
        self.mol = rdedmol
        self.imin, self.imax = 0, self.M
    
    def bondtype(self, i):
        # Get the RDKit bond type for bond i from its signature
        ai = int(i/self.K)
        sai, iai = self.SA[ai], i % self.K
        nai = sai.split('.')[iai+1] # the right neighbor
        return str(nai.split('|')[0])

    def getcomponent(self, ai, CC):
        # Return the set of atoms attached to ai
        CC.add(ai)
        J = np.transpose(np.argwhere(self.A[ai] > 0))[0] 
        for aj in J:
            if aj not in CC: # not yet visited and bonded to ai
                CC = self.getcomponent(aj, CC)
        return CC

    def validbond(self, i, j):
        # Check if bond i, j can be created
        ai, aj = int(i/self.K), int(j/self.K)
        if j < i or self.A[ai, aj]:
            return False
        if self.nbr_component > 1:
            return True 
        # check the bond does not create a saturated component
        self.addbond(i, j)
        I = list(self.getcomponent(ai, set()))
        A = np.copy(self.A[I,:])
        A = A[:,I]
        valid = False
        if A.shape[0] == self.A.shape[0]: 
            # component has all atoms
            valid = True
        else:
            Ad = np.diagonal(A)
            Ab = np.sum(A, axis=1) - Ad
            if np.array_equal(Ad, Ab) == False:
                valid = True # not saturated
        self.removebond(i,j)
        return valid

    def candidatebond(self, i):
        # Search all bonds that can be connected to i 
        # according to sef.B (bond matrix)
        if self.B[self.M,i] == 0:
            return [] # The bond is not free
        F = np.multiply(self.B[i], self.B[self.M]) 
        J = np.transpose(np.argwhere(F != 0))[0]
        np.random.shuffle(J)
        return J
    
    def addbond(self, i, j):
        # add a bond 
        from library.signature import SignatureBondType
        self.B[i,j], self.B[j,i] = 2, 2 # 0: forbiden, 1: candidate, 2: formed
        ai, aj = int(i/self.K), int(j/self.K)
        self.A[ai,aj], self.A[aj,ai] = self.A[ai,aj]+1, self.A[aj,ai]+1
        self.B[self.M,i], self.B[self.M,j] = 0, 0 # i and j not free
        bt = self.bondtype(i)
        self.mol.AddBond(int(ai), int(aj), SignatureBondType(bt))
            
    def removebond(self, i, j):
        # delete a bond
        self.B[i,j], self.B[j,i] = 1, 1 
        ai, aj = int(i/self.K), int(j/self.K)
        self.A[ai,aj], self.A[aj,ai] = self.A[ai,aj]-1, self.A[aj,ai]-1
        self.B[self.M,i], self.B[self.M,j] = 1, 1 
        self.mol.RemoveBond(ai, aj)

    def smiles(self, verbose=False):
        from library.signature import SanitizeMolecule
        # get smiles with rdkit
        mol = self.mol.GetMol()
        mol, smi = SanitizeMolecule(mol, 
                                    kekuleSmiles=self.Alphabet.kekuleSmiles,
                                    allHsExplicit=self.Alphabet.allHsExplicit,
                                    isomericSmiles=self.Alphabet.isomericSmiles,
                                    formalCharge=self.Alphabet.formalCharge,
                                    atomMapping=self.Alphabet.atomMapping,
                                    verbose=verbose)
        return set([smi])
     
    def end(self, i, verbose=False):
        # check if the enumeration ends
        # Get the smiles corresponding to the molecular graph
        # make sure all atoms are connected
        if self.nbr_recursion > self.max_nbr_recursion:
            if verbose: print(f'recursion exceeded for enumeration')
            return True, set()
        if i < self.imax:
            return False, set()
        # we are at the end all atoms must be saturated
        Ad = np.diagonal(self.A)
        Ab = np.sum(self.A, axis=1) - Ad
        if np.array_equal(Ad, Ab) == False:
            if verbose==2:
                print(f'sol not saturated\nDiag: {Ad}\nBond: {Ab}')
            return True, set() 
        if verbose == 2: print(f'smi sol found at',self.nbr_recursion)
        # get the smiles
        return True, self.smiles(verbose=verbose)  

def Enumerate(MG, index=-1, verbose=False):
# Local function that build a requested number of
# molecules (in MG.max_nbr_solution)
# matching the matrices in the molecular graph MG
# ARGUMENTS:
# i: the bond number to be connected
# MG: the molecular graph
# RETURNS:
# Sol: a list of smiles

    # start
    if index < 0:
        index = MG.imin
        MG.nbr_recursion = 0
    MG.nbr_recursion += 1
    
    # Recursion end
    end, Sol = MG.end(index, verbose=verbose)
    if end:
        return Sol

    # search all bonds that can be attached to i
    J = MG.candidatebond(index)
    if len(J) == 0:
        return Enumerate(MG, index=index+1, verbose=verbose)

    # Loop over all possible bonds
    Sol = set()
    for j in J:
        if MG.validbond(index, j):
            MG.addbond(index, j)
            sol = Enumerate(MG, index=index+1, verbose=verbose)
            Sol = Sol | sol
            if MG.nbr_recursion > MG.max_nbr_recursion:
                break # time exceeded
            if sol != set() and len(Sol) >= MG.max_nbr_solution:
                break # max_nbr_solution reached
            MG.removebond(index,j)

    return Sol

###############################################################################
# Bond swapping algorithm
###############################################################################
               
def SwapBond(Swap, index, jndex, MG, Sol, Ncc, verbose=False):
# Local function that swap bonds in molecular graph MG
# ARGUMENTS:
# Swap: the bonds swapping array
# Index: the current set of bonds to be swaped
# MG: the molecular graph
# Sol: the set of smiles
# Ncc: nbr of connected components
# RETURNS:
# Sol: a list of smiles

    def SA(i, MG):
        # root signature of atom i
        return MG.SA[i].split('.')[0]
    
    def SwapInit(MG, verbose=False):
    # Local function that creates a swap bonds array
    # in molecular graph MG
    # Swap contain elements in CC0 (first component)
    # then elements of all other CCs
    # returns the index between CC0 and CC1+
    # and the swap array
        Swap = []
        CC0 = MG.getcomponent(0, set())
        CC1 = set([i for i in range(MG.A.shape[0])]) - CC0
        if verbose == 2: print('CC0', CC0)
        if verbose == 2: print('CC1', CC1)
        CCset = (CC0, CC1)
        mol = MG.mol.GetMol()
        for CC in CCset:
            if len(CC) == 0:
                break
            swap = {}
            for i in CC:
                for j in CC:
                    i, j = int(i), int(j)
                    if i != j and MG.A[i,j] and SA(i, MG) <= SA(j, MG):
                        bt = mol.GetBondBetweenAtoms(i, j).GetBondType()
                        s = f'{SA(i, MG)}|{bt}|{SA(j, MG)}'
                        kij, kji = f'{i}, {j}', f'{j}, {i}'
                        if kij in swap.keys() or kji in swap.keys():
                            continue
                        swap[kij] = s
                        if verbose == 2: print(f'swap init ({i},{j}):{s}')
            swap = dict(sorted(swap.items(), key=lambda x:x[1]))
            swap = list(swap.keys())
            for i in range(len(swap)):
                a, b = swap[i].split(',')
                Swap.append( (int(a), int(b)) )
            index = len(Swap) if CC == CC0 and len(CC1) else 0
            
        if verbose == 2: print(f'Swap array: {Swap}\nindex={index}')
        return index, Swap

    def swapping(Swap, i, j, MG, verbose=False):
        # perform the swapping btw i and j
        # return new swap array and index j
        # (-1 if end of array)
        mol = MG.mol.GetMol()
        while True:
            if j == len(Swap):
                return -1, Swap
            (ai, bi) = Swap[i]
            (aj, bj) = Swap[j]
            if ai == bj or aj == bi:
                j += 1
                continue
            if ai == bi or aj == bj:
                j += 1
                continue
            if MG.A[ai, bj] or MG.A[aj, bi]:
                j += 1
                continue
            bti = mol.GetBondBetweenAtoms(ai, bi).GetBondType()
            btj = mol.GetBondBetweenAtoms(aj, bj).GetBondType()
            si = f'{SA(ai, MG)}|{bti}|{SA(bi, MG)}'
            sj = f'{SA(aj, MG)}|{btj}|{SA(bj, MG)}'
            if si == sj :
                break
            j += 1
        if verbose == 2: print(f'perform swapping indices {i, j}')
        if verbose == 2: print(f'delete bonds {ai, bi}:{bti}, {aj, bj}:{btj}') 
        if verbose == 2: print(f'create bonds {ai, bj}:{bti}, {aj, bi}:{bti}')
        MG.mol.RemoveBond(ai, bi)
        MG.mol.RemoveBond(aj, bj)
        MG.mol.AddBond(int(ai), int(bj), bti)
        MG.mol.AddBond(int(aj), int(bi), btj)
        MG.A[ai, bi], MG.A[aj, bj], MG.A[bi, ai], MG.A[bj, aj] = 0, 0, 0, 0
        MG.A[ai, bj], MG.A[aj, bi], MG.A[bj, ai], MG.A[bi, aj] = 1, 1, 1, 1
        Swap[i] = (ai, bj)
        Swap[j] = (aj, bi)
        return j, Swap

    # Start
    if Swap == None:
        MG.nbr_recursion = 0
        jndex, Swap = SwapInit(MG, verbose=verbose)

    # End
    if index == len(Swap):
        return Sol
    if Sol != set() and len(Sol) >= MG.max_nbr_solution:
        return Sol # max_nbr_solution reached      
    MG.nbr_recursion += 1
    if MG.nbr_recursion > MG.max_nbr_recursion:
        if verbose: print(f'recursion exceeded for bond swapping')
        return Sol
        
    # Recursion
    j = jndex if jndex else index+1
    while j < len(Swap):
        j, Swap = swapping(Swap, index, j, MG, verbose=verbose) # swapping
        if j < 0:
            break
        sol = MG.smiles(verbose=verbose)
        ncc = list(sol)[0].count('.') + 1
        # Keep solution ?
        if ncc < Ncc or ncc <= MG.nbr_component \
        or MG.nbr_component == float('inf'):
            if verbose == 2: print(f'sol found ncc={ncc}')
            Sol, Ncc = Sol | sol, ncc
            if len(Sol) >= MG.max_nbr_solution:
                return Sol
            Sol = SwapBond(Swap, index+1, jndex, MG, Sol, ncc, verbose=verbose)
            if len(Sol) >= MG.max_nbr_solution:
                return Sol
        j, Swap = swapping(Swap, index, j, MG, verbose=verbose) # restoring
        j += 1
        
    Sol = SwapBond(Swap, index+1, jndex, MG, Sol, Ncc, verbose=verbose)
    
    return Sol
    
###############################################################################
# Enumerate Molecules (smiles) from Signature
###############################################################################

def EnumerateMoleculeFromSignature(sig, Alphabet,
                                   max_nbr_recursion=1.0e7, 
                                   max_nbr_solution=float('inf'),
                                   nbr_component=1,
                                   verbose=False):
# Callable function
# Build a molecule matching a provided signature
# ARGUMENTS:
# sig: signature (with neighbor) of a molecule 
# max_nbr_solution: maximum nbr of solutions returned 
# max_nbr_recursion: constant used in signature_enumerate
# nbr_component: nbr connected components
# RETURNS:
# The list of smiles

    from library.enumerate_utils import GetConstraintMatrices
    from library.signature_alphabet import SignatureFromSmiles
    
    SMIsig, Nsig = set(), 0
            
    # Get initial molecule
    AS, NAS, Deg, A, B, C = GetConstraintMatrices(sig, 
                                                  unique=False,
                                                  verbose=verbose)

    MG = MolecularGraph(A, B, AS, Alphabet, ai=-1, 
             max_nbr_recursion=max_nbr_recursion, 
             max_nbr_solution=max_nbr_solution)
    MG.nbr_component = float('inf')
    MG.max_nbr_solution = 1
    SMI = Enumerate(MG, verbose=verbose)
    if len(SMI) == 0:
        return np.asarray([])
    Ncc = list(SMI)[0].count('.') + 1
    if verbose: 
        print(f'Enumerate nbr-sol: {len(list(SMI))} Ncc:{Ncc} smi:{list(SMI)[0]}')
            
    # Decrease Ncc
    while Ncc > nbr_component:
        MG.nbr_component = 1
        MG.max_nbr_solution = 1 
        SMI = SwapBond(None, 0, 0, MG, set(), Ncc, verbose=verbose)
        if len(SMI) == 0:
            break
        Ncc = list(SMI)[0].count('.') + 1 
        if verbose: 
                print(f'Decrease CC nbr-sol:\
{len(list(SMI))} Ncc:{Ncc} smi:{list(SMI)[0]}')
        if len(SMI) == 0:
            continue
            
    # Enumerate max_nbr_solution solutions
    MG.nbr_component = 2
    MG.max_nbr_solution = max_nbr_solution
    MG.max_nbr_recursion = 10 * max_nbr_solution # 10 trial per solution
    SMI = SwapBond(None, 0, 0, MG, SMI, Ncc, verbose=verbose)
    if verbose: 
        print(f'CC=1 nbr-sol:{len(list(SMI))} Ncc:{Ncc} smi:{list(SMI)}')

    # retain solutions having a signature = provided sig 
    for smi in SMI:
        if smi == '':
            continue
        if '.' in smi:
            continue
        sigsmi, mol, smisig = SignatureFromSmiles(smi, Alphabet)
        if sigsmi == sig:
            SMIsig.add(smisig)

    return list(SMIsig)

###############################################################################
# Enumerate Signatures from Morgan vector
###############################################################################

def EnumerateSignatureFromMorgan(morgan, Alphabet, timeout=100, 
                                 binarysolution=False, verbose=False):
# Callable function
# Compute all possible signature having a the same Morgan vector 
# than the provided one. Make use of a Python (sympy) diophantine solver
# ARGUMENTS:
# morgan: the Morgan vector
# binarysolution: when True atom signater do not have occurance numbers
#                 but are are duplicated. In other words, the OCC vector
#                 takes only O/1 values
# RETURNS:
# The list of signature strings matching the Morgan vector 
    from library.enumerate_utils import GetConstraintMatrices
    from library.enumerate_utils import UpdateConstraintMatrices
    from library.signature_alphabet import SignatureAlphabetFromMorganBit
    from sympy import Matrix
    from diophantine import solve
    import signal
    
    def SignatureSet(Sig, Occ):
        # Return a set of signature string
        from library.signature_alphabet import SignatureVectorToString
        S = set()
        for i in range(Occ.shape[0]):
            if len(Occ[i]):
                S.add(SignatureVectorToString(Occ[i], Sig))
        return S  

    def handle_timeout(sig, frame):
        raise TimeoutError('took too long')
        
    # Get alphabet signatures in AS along with their
    # minimum and maximum occurence numbers
    # randomize the list of indices
    AS, MIN, MAX, IDX, I = {}, {}, {}, {}, 0
    L = np.arange(morgan.shape[0])
    np.random.shuffle(L)
    for i in list(L):
        if morgan[i] == 0:
            continue
        # get all signature in Alphabet having MorganBit = i
        sig = SignatureAlphabetFromMorganBit(i, Alphabet)
        if verbose: 
            print(f'MorganBit {i}:{morgan[i]}, Nbr in alphabet {len(sig)}')
        (maxi, K) = (1, morgan[i]) if binarysolution else (morgan[i], 1)
        mini = 0 if len(sig) > 1 else maxi
        for j in range(len(sig)):
            for k in range(int(K)):
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
    
    # Get matrix A and vector b for diophantine solver
    A, b, m = C, np.zeros(C.shape[0]), -1
    for i in range(AS.shape[0]):
        mi = int(AS[i].split(',')[0]) # morgan bit
        if mi != m:
            A = np.concatenate((A, P), axis=0) if m != -1 else A
            b = np.concatenate((b, [morgan[m]]), axis=0) if m != -1 else b
            P, m = np.zeros(A.shape[1]).reshape(1,A.shape[1]), mi
        P[0,i] = 1
    if verbose: 
        print(f'A: {A.shape} b: {b.shape}')
        
    A, b = Matrix(A.astype(int)), Matrix(b.astype(int))
    if verbose == 2:
        print(f'A = {A}\nb = {b}')

    # Solve
    signal.signal(signal.SIGALRM, handle_timeout)
    try: 
        signal.alarm(timeout)
        OCC = np.asarray(list(solve(A, b)))
        signal.alarm(0)
    except TimeoutError as exc: 
        OCC = np.asarray([])
        if verbose:
            print(f'Diophantine solver time ({timeout}) exceeded')
    if OCC.shape[0] == 0:
        return []
    OCC = OCC.reshape(OCC.shape[0], OCC.shape[1])
    OCC = OCC[:,:AS.shape[0]]

    Sol = SignatureSet(AS, OCC) 
    
    return list(Sol)

