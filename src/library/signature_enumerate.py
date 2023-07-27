###############################################################################
# This library enumerate signatures for three different purposes
# - Enumerate all atom signatures one radius up for a provided atom signature
# - Enumerate all atom signatures having the same Morgan Vector 
#   than the provided atom signature
# - Enumerate all molecules (smiles) for a provided molecule signature
# Signatures must be computed using neighbor = True
# cf. signature.py for signature format 
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Apr. 2023
###############################################################################

from library.imports import *

###############################################################################
# MolecularGraph local object used for the enumeration
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
                 max_nbr_recursion=1.0e6, # Max nbr of recursion
                 ai=-1, # Current atom nbr used when enumerating signature up
                 max_nbr_solution=float('inf') # to produce all solutions
                ):
        self.A, self.B, self.SA = A, B, SA
        self.max_nbr_solution = max_nbr_solution
        self.M = self.B.shape[1] # number of bounds
        self.K = int(self.B.shape[1] / self.SA.shape[0]) # nbr of bound/atom
        self.ai = ai # current atom for which signature is expanded
        self.nbr_recursion = 0 # Nbr of recursion
        self.max_nbr_recursion = max_nbr_recursion
        self.smiles = True if ai < 0 else False
        if self.smiles: # create an initial editable molecule
            rdmol = Chem.Mol()
            rdedmol = Chem.EditableMol(rdmol)
            for sa in self.SA:
                atomnum = sa.split('|')[0].split(',')[1]
                rdatom = Chem.Atom(int(atomnum))
                rdedmol.AddAtom(rdatom)
            self.mol = rdedmol
            self.imin, self.imax = 0, self.M
        else:
            self.mol = None
            self.imin, self.imax = ai*self.K, (ai+1)*self.K
 
    def signaturebond(self, i):
        # Get the signature of the atom corresponding to bond i
        return self.SA[int(i/self.K)]
    
    def bondtype(self, i):
        # Get the RDKit bond type for bond i from its signature
        ai = int(i/self.K)
        sai, iai = self.SA[ai], i % self.K
        nai = sai.split('.')[iai+1] # the right neighbor
        return str(nai.split('|')[0])
    
    def validbond(self, i, j, sa):
        # Check if bond i, j can be created
        if self.smiles:
            if j < i or self.A[int(i/self.K),int(j/self.K)]:
                return False
            else:
                return True
        else: 
            saj = self.signaturebond(j)
            if saj == sa:
                return False
            else:
                return True
    
    def candidatebond(self, i):
        # Search all bonds that can be connected to i 
        # according to sef.B (bond matrix)
        if self.smiles and self.B[self.M,i] == 0:
            return [] # The bond is not free
        F = np.multiply(self.B[i], self.B[self.M]) if self.smiles \
        else self.B[i]
        J = list(np.transpose(np.argwhere(F != 0))[0])
        return J
    
    def addbond(self, i, j):
        # add a bond 
        from library.signature import SignatureBondType
        self.B[i,j], self.B[j,i] = 2, 2 # 0: forbiden, 1: candidate, 2: formed
        if self.smiles:
            ai, aj = int(i/self.K), int(j/self.K)
            self.A[ai,aj], self.A[aj,ai] = self.A[ai,aj]+1, self.A[aj,ai]+1
            self.B[self.M,i], self.B[self.M,j] = 0, 0 # i and j not free
            bt = self.bondtype(i)
            self.mol.AddBond(int(ai), int(aj), SignatureBondType(bt))
            
    def removebond(self, i, j):
        # delete a bond
        self.B[i,j], self.B[j,i] = 1, 1 
        if self.smiles:
            ai, aj = int(i/self.K), int(j/self.K)
            self.A[ai,aj], self.A[aj,ai] = self.A[ai,aj]-1, self.A[aj,ai]-1
            self.B[self.M,i], self.B[self.M,j] = 1, 1 
            self.mol.RemoveBond(ai, aj)
            
    def atomsignatureup(self, ai, verbose=False):
        # Get the signature of ai one radius up attaching ai 
        # according to self.B (the bound matrix)
        sai, saj, n = self.SA[ai], {}, 0
        for k in range(self.K): # for all neighbors of ai
            i = ai * self.K + k
            J = np.transpose(np.argwhere(self.B[i] == 2))[0]
            if len(J) > 0:
                # Note only one bond (=2: J[0]) is connected to i
                saj[n] = self.signaturebond(J[0])
                n += 1
        # update signature
        sni = list(saj.values()) # all neighbors of ai
        sig = AtomSignatureMod(sai)
        sai = sai.split('.')
        for i in range(len(sni)):
            bondtype = sai[i+1].split('|')[0]
            sig = sig + '.' + bondtype + '|' + AtomSignatureMod(sni[i])
        # sort the neighbors of ai
        signature = sig.split('.')[0]
        sni = sig.split('.')[1:]
        sni.sort() 
        for s in sni:
            signature = signature + '.' + s
        return signature
    
    def endsignature(self, verbose=False):
        # Get the molecular signature one radius up attaching 
        # for atom self.ai
        signature = self.atomsignatureup(self.ai, verbose=verbose)
        return set([signature])

    def getcomponent(self, ai, CC):
        # Return the set of atoms attached to ai
        CC.add(ai)
        J = np.transpose(np.argwhere(self.A[ai] > 0))[0] 
        for aj in J:
            if aj not in CC: # not yet visited and bonded to ai
                CC = self.getcomponent(aj, CC)
        return CC
            
    def endsmiles(self, verbose=False):
        # Get the smiles corresponding to the molecular graph
        # make sure all atoms are attached
        # make sure all atoms are connected
        Ad = np.diagonal(self.A)
        Ab = np.sum(self.A, axis=1) - Ad
        if np.array_equal(Ad, Ab) == False:
            if verbose==2:
                print(f'not saturated\nDiag: {Ad}\nBond: {Ab}')
            return set()
        try:
            rdmol = self.mol.GetMol()
            Chem.SanitizeMol(rdmol)
            smi = Chem.MolToSmiles(rdmol)
            return set([smi])
        except:
            return set(['NO-RDKIT-SMI'])
            
    def end(self, i, verbose=False):
        # check if the enumeration ends
        Sol = set()
        self.nbr_recursion += 1
        if self.nbr_recursion > self.max_nbr_recursion:
            if verbose == 2: print(f'nbr recursion exceeded for smi')
            return True, set()
        if i < self.imax:
            return False, Sol
        Sol = self.endsmiles(verbose=verbose) if self.smiles \
        else self.endsignature(verbose=verbose)
        if self.smiles and len(Sol):
            if verbose == 2: print(f'smi sol found at',self.nbr_recursion)
        return True, Sol

def Enumerate(MG, index=-1, verbose=False):
# Local function that build all signatures or molecules 
# matching the matrices in the molecular graph MG
# ARGUMENTS:
# i: the bond number to be connected
# MG: the molecular graph
# RETURNS:
# Sol: a list of solutions (atom signature or smiles)

    i = MG.imin if index < 0 else index
    #print('entering with i:', i, MG.imax)
    # Recursion end
    end, Sol = MG.end(i, verbose=verbose)
    #print('end', end)
    if end:
        return Sol
 
    # search all bonds that can be attached to i
    J = MG.candidatebond(i)
    if len(J) == 0:
        return Enumerate(MG, index=i+1, verbose=verbose)

    # Loop over all possible bonds
    Sol, sa = set(), ''
    for j in J:
        #print('test J, i,j', J, i,j)
        if MG.validbond(i, j, sa):
            sa = MG.signaturebond(j)
            MG.addbond(i, j)
            #print('add i,j', i,j)
            sol = Enumerate(MG, index=i+1, verbose=verbose)
            Sol = Sol | sol
            if MG.nbr_recursion > MG.max_nbr_recursion:
                break # time exceeded
            if sol != set() and len(Sol) >= MG.max_nbr_solution:
                break # max_nbr_solution reached
            MG.removebond(i,j)

    return Sol

###############################################################################
# Signature enumeration to smiles
###############################################################################

def EnumerateSignatureToSmiles(A, B, AS, 
                               max_nbr_solution=float('inf'), 
                               max_nbr_recursion=1.0e6,
                               verbose=False):
# Callable function
# ARGUMENTS:
# A:  an adjacency matrix between the atoms of a molecule
#     with diagonal = atom degree
# B:  an adjacency matrix between the bond candidates
#     of a molecule. The last row indicates bond candidate 
#     is used (1) during enumeration (0 at initialization)
# AS: an array of atom signature
# RETURNS:
# Array of smiles matching A, B, AS 

        MG = MolecularGraph(A, B, AS, ai=-1, 
                            max_nbr_recursion=max_nbr_recursion, 
                            max_nbr_solution=max_nbr_solution)
        SMI = Enumerate(MG, verbose=verbose)
        return np.asarray(list(SMI))

###############################################################################
# Signature enumeration matching a Morgan vector
###############################################################################

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
        self.Ci = np.asarray([[]])
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
        else: # !!!!
            return True
        # Compute constraint Ci up to index i
        N, J = self.AS.shape[0], i+1
        if self.Ci.shape[1] == 0:
        # get cst the sum of the constraint columns (after AS)
            cst = np.sum(np.absolute(self.C[:,N:]), axis=1)
            if verbose==2: print(f'OccConstraint: N {N} J {J} C {C.shape}')
            # get the constraint matrix up to column J
            # rows having non null element after column J
            # are discarded
            C, Ci, K = self.C[:,:N], {}, 0
            for k in range(C.shape[0]):
                l = np.transpose(np.argwhere(np.absolute(C[k]) > 0))[0]
                if l.shape[0] > 0:
                    if l[-1] < J:
                        Ci[K] = C[k,:J]
                        K += 1
            if K == 0: # no constraint can be checked
                return True
            self.Ci = np.asarray(list(Ci.values()))
            if verbose==2: print(f'OccConstraint for {i} Ci {Ci.shape}')
            if self.Ci.shape[1] == 0:
                return True # no constraint
        # Check constraints up to index i
        if np.sum(np.absolute(self.Ci.dot(self.occ[:J]))) % 2 == 0:
            if verbose==2: print(f'Valid Constraint for {i}')
            return True
        else:
            if verbose==2: print(f'Invalid Constraint for {i}')
            return False
    
    def endocc(self, i, verbose=False):
        if verbose==2 and \
        self.nbr_recursion % int(self.max_nbr_recursion/10) == 0:
            print(f'...{self.nbr_recursion}')
        self.nbr_recursion += 1
        if self.nbr_recursion > self.max_nbr_recursion:
            if verbose==2: print(f'nbr recursion exceeded for occ')
        if i == self.occ.shape[0]:
            #return True # !!!
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
        # reset contraint matrix as the constraint
        # matrix is computed only once for i
        self.Ci = np.asarray([[]]) # Ci not C !!
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

def EnumerateSignatureToSignature(A, B, AS, NAS, 
                                  Alphabet=None, 
                                  max_nbr_recursion=1.0e5,
                                  max_nbr_solution=float('inf'),
                                  verbose=False):
# Callable function
# Provided a signature either
# - Enumerate all signatures one radius up (Alphabet=None)
# - Enumerate all signatures having the same Morgan Vector
# ARGUMENTS:
# A:  an adjacency matrix between the atoms of the molecule
#     with diagonal = atom degree
# B:  an adjacency matrix between the bond candidates
#     of the molecule. The last row indicate is used during enumeration
#     and filled with 0 at initialization
# AS: an array of atom signature
# max_nbr_solution: maximum nbr of solutions returned 
# RETURNS:
# A list of updated signature and their occurence numbers 

    from library.reverse_engineer_utils import UpdateConstraintMatrices
    from library.signature_alphabet import MorganBitFromSignature
    from library.signature_alphabet import SignatureAlphabetFromMorganBit
    
    # Get updated signatures in ASU along with their
    # minimum and maximum occurence numbers
    ASU, MINU, MAXU, IDXU, I = {}, {}, {}, {}, 0
    for i in range(AS.shape[0]):
        if Alphabet == None:
            # Enumerate all signature Up
            MG = MolecularGraph(A, B, AS, 
                                max_nbr_recursion, ai=i, 
                                max_nbr_solution=max_nbr_solution)
            su = list(Enumerate(MG, verbose=verbose))
        else:
            # get all signature in Alphabet having the same Morgan code
            mb = MorganBitFromSignature(AS[i], 
                                        Alphabet, verbose=verbose)
            su = SignatureAlphabetFromMorganBit(mb, Alphabet)
            if AS[i] not in su: # sanity check
                print(f'Error: atom signature {AS[i]}, not in Alphabet')
                sys.exit('Error') 
            
        if verbose==2: print(f'{i}, SA {AS[i]}')
        if verbose==2: print(f'{i}, SU {len(su)}, {su}')
        maxu = NAS[i]
        minu = 0 if len(su) > 1 else maxu
        for j in range(len(su)):
            ASU[I], MINU[I], MAXU[I], IDXU[I] = su[j], minu, maxu, i
            I += 1
            
    # Get Matrices for ASU
    ASU = np.asarray(list(ASU.values()))
    IDXU = np.asarray(list(IDXU.values()))
    MINU = np.asarray(list(MINU.values()))
    MAXU = np.asarray(list(MAXU.values()))
    DegU = np.asarray([len(ASU[i].split('.'))-1 for i in range(ASU.shape[0])])
    n1 = ASU.shape[0]
    ASU, IDXU, MINU, MAXU, DegU, CU = \
    UpdateConstraintMatrices(ASU, IDXU, MINU, MAXU, DegU, verbose=verbose)
    n2 = ASU.shape[0]
    if verbose:
        print(f'ASU reduction {n1}, {n2}')
    
    # Enumerate all possible vectors occ for the occurence
    # numbers of the signature in ASU
    OS  = OccurenceSignature(ASU, CU, IDXU, MINU, MAXU, 
                             max_nbr_recursion, 
                             max_nbr_solution=max_nbr_solution)
    OS.printout(verbose=verbose)
    OCC = EnumerateOcc(OS, 0, verbose=verbose)
    return ASU, OCC

