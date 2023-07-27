###############################################################################
# This library run one step retrosynthesis with different methods
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Jan 2023
###############################################################################

from library.imports import *

###############################################################################
# Reaction object for retrosynthesis
# - ID:  an array of reference reactions
# - Diameter:  an array of rule diameter 
# - Rule: an array of smart string with AA mapping 
# - Score: an array of rule score
# - Direction: an array of rule direction (-1: retro, 1: formward, 0: both)
# - SR: a NsxNr 2D array witn Ns atom signatures and Nr reactions rules
#       SR [i,j] < 1  when the susbstrate of reaction rule j 
#       has the atom signature Signature[i]
#       SignatureReaction[i,j] > 1  hen the product of reaction rule j 
# - SSR: ReLU(-SR), the signature of the reaction substrate
# - SSRsum: the sum per column (reaction) of SSR
# - SRU: same as SR but with unique signature
# - SSRU: ReLU(-SRU)
# - SSRUsum: the sum per column (reaction) of SSRU
###############################################################################

class ReactionObject:
    def __init__(self, Alphabet={}):
        self.filename = ''
        self.Alphabet = Alphabet
        self.ID = [] 
        self.Diameter = [] 
        self.Rule = [] 
        self.Score = [] 
        self.Direction = [] 
        self.SR = [] 
        self.SSR = [] 
        self.SSRsum = []
        self.SRU = [] 
        self.SSRU = [] 
        self.SSRUsum = []
        
    def fill(self, Substrate, Product, 
             ID, Diameter, Rule, Score, Direction, verbose=True):
        
        from library.utils import ReLU
        from library.signature_alphabet import SignatureFromSmiles
    
        SS, SP, I, n = {}, {}, [], 0
        S, P = Substrate, Product
        if S.shape[0] != P.shape[0] or S.shape[0] != ID.shape[0]:
            print(f'Error ID, Substrate and Product must have the same size \
                  {ID.shape} {S.shape} {P.shape}')
            sys.exit('Error')
            
        if self.Alphabet == {}:
            self.ID, self.Diameter = ID, Diameter
            self.Rule, self.Score, self.Direction = Rule, Score, Direction
            return
            
 
        for i in range (S.shape[0]):
            ss, _ = SignatureFromSmiles(S[i], self.Alphabet, 
                                        string=False, verbose=verbose)
            sp, _ = SignatureFromSmiles(P[i], self.Alphabet, 
                                        string=False, verbose=verbose)
            if len(ss) == 0 or len(sp) == 0:
                if verbose:
                    if len(ss) == 0 :
                        print(f'WARNING no substrate signature for {S[i]}')
                    if len(sp) == 0 :
                        print(f'WARNING no product signature for {P[i]}')
                    print(f'skip reaction {i}')
                continue
            if i % 1000 == 0:
                print(f'... processing reactions iteration: {i}')
            # to get the right SS, SP, ID and Rule
            SS[n], SP[n] = ss, sp
            I.append(i) 
            n += 1
        
        # Remove skipped reactions            
        self.ID, self.Diameter = ID[I], Diameter[I]
        self.Rule, self.Score, self.Direction = Rule[I], Score[I], Direction[I]

        # Get SignatureReaction 
        SS = np.asarray(list(SS.values()))
        SP = np.asarray(list(SP.values()))
        self.SR = np.transpose(SP) - np.transpose(SS)
        if verbose:
            print(f'sizes for ID, SignatureReaction, Number of reaction skiped\
                  {self.ID.shape} {self.SR.shape} {S.shape[0]-self.SR.shape[1]}')
                 
        # Get substrate signature and sum per column (reaction)         
        self.SSR = ReLU(-self.SR) 
        self.SSRsum = np.sum(self.SSR, axis = 0)
        self.SRU = np.unique(self.SR, axis=1)
        self.SSRU = ReLU(-self.SRU) 
        self.SSRUsum = np.sum(self.SSRU, axis = 0)
        
    def save(self, filename):
        filename = filename+'.npz' if filename.find('.npz') == -1 else filename
        if self.Alphabet != {}:
            alphabetname = self.Alphabet.filename
        else:
            alphabetname = ''
        np.savez_compressed(filename,
                            filename=filename,
                            alphabetname=alphabetname,
                            ID=self.ID, 
                            Diameter=self.Diameter, 
                            Rule=self.Rule,
                            Score=self.Score,
                            Direction=self.Direction,
                            SR=self.SR, 
                            SSR=self.SSR, 
                            SSRsum=self.SSRsum,
                            SRU=self.SRU, 
                            SSRU=self.SSRU, 
                            SSRUsum=self.SSRUsum)
        
    def printout(self):
        print(f'filename: {self.filename}')
        if self.Alphabet != {}:
            print(f'alphabet name: {self.Alphabet.filename}')
            print(f'alphabet length: {len(self.Alphabet.Dict.keys())}')
        print(f'ID, Diameter, Rule, Score, Direction : {self.ID.shape}')
        print(f'SR: {self.SR.shape}')
        print(f'SSR: {self.SSR.shape}')
        print(f'SSRsum: {self.SSRsum.shape}')
        print(f'SRU: {self.SRU.shape}')
        print(f'SSRU: {self.SSRU.shape}')
        print(f'SSRUsum: {self.SSRUsum.shape}')

def LoadReaction(filename):
    # Load the Alphabet the reaction ID, Smarts-rules, and signatures 
    from library.signature_alphabet import LoadAlphabet
    filename = filename+'.npz' if filename.find('.npz') == -1 else filename
    load = np.load(filename, allow_pickle=True)
    Reaction=ReactionObject()
    Reaction.filename=filename
    alphabetname=load['alphabetname']
    if alphabetname != '':
        Reaction.Alphabet = LoadAlphabet(str(load['alphabetname']))
    Reaction.ID=load['ID']
    Reaction.Diameter=load['Diameter']
    Reaction.Rule=load['Rule']
    Reaction.Score=load['Score']
    Reaction.Direction=load['Direction']
    Reaction.SR=load['SR']
    Reaction.SSR=load['SSR']
    Reaction.SSRsum=load['SSRsum']
    Reaction.SRU=load['SRU']
    Reaction.SSRU=load['SSRU']
    Reaction.SSRUsum=load['SSRUsum']

    return Reaction

###############################################################################
# One step retrosynthesis callable functions
###############################################################################
              
def SanatizeRxnProducts(products, first=False):
# Callable function
# ARGUMENTS:
# The set tuple of tuple of RunReactants
# first: If False sanatize all solutions in the tuple
# RETURNS:
# A list of products 

    P = set()
    # runreactant can produce several solutions
    for i in range(len(products)):
        # get the sanatized molecules corresponding to solution i
        smi = ''
        prod = products[i]
        for p in prod:
            if smi == '':
                smi = Chem.MolToSmiles(p)
            else: 
                smi = smi + '.' + Chem.MolToSmiles(p)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        P.add(smi)
        if first:
            break
    return list(P)

def OneStepRetroRule(mol, Reaction, sanatize=False):
# Callable function
# ARGUMENTS:
# The molecule in RDKit format
# A Reaction object
# RETURNS:
# An dictionary of products 
# key = reaction nbr, value = a list of sanatized products in RDKit format
# The list of sucessfull reaction fired

    firelist, products = [], {}
    for j in range(Reaction.Rule.shape[0]):
        rxn = AllChem.ReactionFromSmarts(Reaction.Rule[j])
        p = rxn.RunReactants((mol,))
        if len(p) > 0:
            firelist.append(j)
            products[str(j)] = SanatizeRxnProducts(p) if sanatize else [p]
    return products, firelist

def OneStepRetroSignature(sigmol, Reaction):
# Callable function
# ARGUMENTS:
# SigMol: the signature vector of the molecule
# A Reaction object
# RETURNS:
# An dictionary of product signatures 
# key = reaction nbr, value = the signature
# The list of sucessfull reaction fired

    firelist, sigprod = [], {}
    I = np.transpose(np.argwhere(sigmol != 0))[0]
    if I.shape[0] > 0:
        SSM = Reaction.SSRU[I]
        SSMsum = np.sum(SSM, axis = 0)
        firelist = np.where(SSMsum == Reaction.SSRUsum)[0]
    for j in firelist:
        sigprod[str(j)] = sigmol + Reaction.SRU[:,j]  
    return sigprod, firelist

def OneStepRetroSignatureRule(mol, sigmol, Reaction, sanatize=False):
# Callable function
# ARGUMENTS:
# molecule: the molecule in RDKit format
# sigmol: the signature vector of the molecule
# A Reaction object
# RETURNS:
# An dictionary of product signature
# key = reaction nbr, value = the signature
# An dictionary of products 
# key = reaction nbr, value = a list of sanatized products in RDKit format
# The list of sucessfull reaction fired

    firelist, sigprod, products = [], {}, {}
    I = np.transpose(np.argwhere(sigmol != 0))[0]
    if I.shape[0] > 0:
        SSM = Reaction.SSR[I]
        SSMsum = np.sum(SSM, axis = 0)
        firelist = np.where(SSMsum == Reaction.SSRsum)[0]
    for j in firelist:
        sigprod[str(j)] = list(sigmol + Reaction.SR[:,j])  
        rxn = AllChem.ReactionFromSmarts(Reaction.Rule[j])
        p = rxn.RunReactants((mol,))
        if len(p) > 0:
            products[str(j)] = SanatizeRxnProducts(p) if sanatize else [p]    
    return sigprod, products, firelist
