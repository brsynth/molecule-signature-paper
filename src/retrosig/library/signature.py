###############################################################################
# This library compute signature on atoms and molecules using RDKit
#
# Molecule signature: the signature of a molecule is composed of the signature 
# of its atoms. The string is separated by ' ' between atom signature
#
# Atom signature: there are sereval ways of computing atom signature depending 
# on the parameters neighbor and userange. In general an atom signature
# is represented by a rooted SMILES string (the root is the atom laleled 1)
#
# Below are format examples for the oxygen atom in phenol with radius = 2
# (and nBits=2048 Morgan Fingerprint size when using neighbor=True)
#  - neighbor = False, userange = False
#    C:C(:C)[OH:1]
#    here the root is the oxygen atom labeled 1: [OH:1] 
#  - neighbor = False, userange = True
#    [O:1]&C[OH:1]&C:C(:C)[OH:1] 
#    same as above but for signature of radius 0 to 2 separated by '&'
#  - neighbor = True, userange = False
#    2,8,91,C[OH:1].SINGLE|2,6,176,C:[C:1](:C)O
#    here the signature is computed for the root and its neighbors
#    for radius-1, root and neighbors are separated by '.'
#    2 is the radius, 8 stands for oxygen the radius-1 signature is C[OH:1], 
#    91 is the Morgan bit of oxygen (for radius=2 and nBits=2048).
#    The oxygen atom is linked by a SINGLE bond to
#    a carbon of signature 2,6,176,C:[C:1](:C)O 
#    (2=radius, 6=carbon, 176=Morgan bit)
#  - neighbor = True, userange = True
#    2,8,91,[O:1]&C[OH:1].SINGLE|2,6,176,[C:1]&C:[C:1](:C)O
#    same as above but for signature of radius 0 to radius-1=1 separated by '&'
#
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Jan 2023 modified June 2023
###############################################################################

from library.imports import *

def SignatureBondType(bt='UNSPECIFIED'):
# Callable function
# Necessary because RDKit functions
# GetBondType (string) != Chem.BondType (RDKit object)
# Must be updated with new RDKit release !!!
    BondType = {
    'UNSPECIFIED': Chem.BondType.UNSPECIFIED,
    'SINGLE':  Chem.BondType.SINGLE,
    'DOUBLE':  Chem.BondType.DOUBLE,
    'TRIPLE':  Chem.BondType.TRIPLE,
    'QUADRUPLE':  Chem.BondType.QUADRUPLE,
    'QUINTUPLE':  Chem.BondType.QUINTUPLE,
    'HEXTUPLE':  Chem.BondType.HEXTUPLE,
    'ONEANDAHALF':  Chem.BondType.ONEANDAHALF,
    'TWOANDAHALF':  Chem.BondType.TWOANDAHALF,
    'THREEANDAHALF':  Chem.BondType.THREEANDAHALF,
    'FOURANDAHALF':  Chem.BondType.FOURANDAHALF,
    'FIVEANDAHALF':  Chem.BondType.FIVEANDAHALF,
    'AROMATIC':  Chem.BondType.AROMATIC,
    'IONIC':  Chem.BondType.IONIC,
    'HYDROGEN':  Chem.BondType.HYDROGEN,
    'THREECENTER':  Chem.BondType.THREECENTER,
    'DATIVEONE':  Chem.BondType.DATIVEONE,
    'DATIVE':  Chem.BondType.DATIVE,
    'DATIVEL':  Chem.BondType.DATIVEL,
    'DATIVER':  Chem.BondType.DATIVER,
    'OTHER':  Chem.BondType.OTHER,
    'ZERO':  Chem.BondType.ZERO }
    return BondType[bt]

###############################################################################
# Sanatize callable function
###############################################################################

def SanatizeMolecule(mol, 
                     isomericSmiles=False, 
                     formalCharge=False, 
                     atomMapping=False,
                     verbose= False):
# Callable function
# ARGUMENTS:
# mol: the molecule in rdkit format
# isomericSmiles: (optional) include information about stereochemistry. 
# formalCharge: (optional) Remove charges on atom when False. 
# atomMapping: (optional) Remove atom mapping when False. 
# allHsExplicit: (optional) if true, H are added.
# RETURNS:
# The sanatized molecule 

    mol = Chem.RemoveHs(mol)
    try:
        mol = Chem.rdmolops.AddHs(mol)
    except:
        if verbose:
            print(f'WARNING: molecule cannot be sanatized (add hydrogen)')
        return None, ''
    
    if not isomericSmiles:
        try:
            Chem.RemoveStereochemistry(mol)
        except:
            if verbose:
                print(f'WARNING: molecule cannot be sanatized (stereochemistry)')
            return None, ''

    if not formalCharge:
        [a.SetFormalCharge(0) for a in mol.GetAtoms()]
        
    if not atomMapping:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]

    return mol

###############################################################################
# signature as a string
###############################################################################

def AtomSignature(atm,
                  radius=2,
                  useRange=False,
                  isomericSmiles=False,
                  kekuleSmiles=False,
                  allHsExplicit=False,
                  verbose=False):
# Local function
# See GetAtomSignature for arguments

    signature = ''
    if atm is None:
        return signature  
    if allHsExplicit == False: # one keep charged hydrogen
        if atm.GetAtomicNum() == 1 and atm.GetFormalCharge() == 0:
            return signature
    mol = atm.GetOwningMol()
    if atm is None:
        return signature
    if radius < 0:
        radius = mol.GetNumAtoms()
    if radius > mol.GetNumAtoms():
        radius = mol.GetNumAtoms()
        
    # Recursive call if UseRange is True
    if useRange == True:
        sp = ''
        for rad in range(radius+1): # yep range
            s = GetAtomSignature(atm,
                                 radius=rad,
                                 useRange=False,
                                 isomericSmiles=isomericSmiles,
                                 kekuleSmiles=kekuleSmiles,
                                 allHsExplicit=allHsExplicit,
                                 verbose=verbose)
            if len(signature) > 0:
                if s == sp:
                    return signature # no need to go higher radius
                else:
                    signature = signature + '&' + s 
            else: # rad=0
                signature = s 
            sp = s
        return signature
    
    # We get in atomToUse and env all neighbors atoms and bonds up to given radius
    atmidx = atm.GetIdx()
    env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atmidx,useHs=True)
    while len(env) == 0 and radius > 0:
        radius = radius - 1
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atmidx,useHs=True)
    if radius>0:
        atoms=set()
        for bidx in env:
            atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        atomsToUse=list(atoms)
    else:
        atomsToUse = [atmidx]
        env=None
        
    # Now we get to the business of computing the atom signature        
    atm.SetAtomMapNum(1)
    try:
        signature  = Chem.MolFragmentToSmiles(mol,
                                              atomsToUse,bondsToUse=env,
                                              rootedAtAtom=atmidx,
                                              isomericSmiles=isomericSmiles,
                                              kekuleSmiles=kekuleSmiles,
                                              canonical=True,
                                              allBondsExplicit=True,
                                              allHsExplicit=allHsExplicit)
        # Chem.MolFragmentToSmiles canonicalizes the rooted fragment 
        # but does not do the job properly.
        # To overcome the issue the atom is mapped to 1, and the smiles 
        # is canonicalized via Chem.MolToSmiles
        signature = Chem.MolFromSmiles(signature)
        if allHsExplicit:
            signature = Chem.rdmolops.AddHs(signature)
        signature = Chem.MolToSmiles(signature)
    except:
        if verbose:
            print(f'WARNING cannot compute atom signature for: \n\
                  molecule {mol} atom num: {atmidx} {atm.GetSymbol()} radius: {radius}') 
        signature =  ''
    atm.SetAtomMapNum(0)
        
    return signature

def GetAtomSignature(atm,
                     Codes=[],
                     radius=2,
                     neighbors=False,
                     useRange=False,
                     isomericSmiles=False,
                     kekuleSmiles=False, 
                     allHsExplicit=False,
                     verbose=False):
# Local function
# ARGUMENTS:
# cf. GetMoleculeSignature
#
# RETURNS:
# A signature (string) where atom signatures are sorted in lexicographic 
# order and separated by ' '
# see GetAtomSignature for atom signatures format

    signature, temp_signature = '', []
    radius = radius-1 if neighbors else radius
    if radius < 0:
        return signature
     
    # We compute atom signature for atm
    signature = AtomSignature(atm, radius=radius, useRange=useRange,
                        isomericSmiles=isomericSmiles,
                        kekuleSmiles=kekuleSmiles,
                        allHsExplicit=allHsExplicit,
                        verbose=verbose)
    if neighbors == False or signature == '' or len(Codes) == 0:
        return signature
    
    # We compute atom signatures for all neighbors
    signature = str(Codes[atm.GetIdx()]) + ',' + signature
    mol = atm.GetOwningMol()
    atmset = atm.GetNeighbors() 
    sig_neighbors, temp_sig = '', []
    for a in atmset:
        s = AtomSignature(a, radius=radius, useRange=useRange,
                          isomericSmiles=isomericSmiles,
                          kekuleSmiles=kekuleSmiles,
                          allHsExplicit=allHsExplicit,
                          verbose=verbose)
        if s != '': 
            bond = mol.GetBondBetweenAtoms(atm.GetIdx(),a.GetIdx())
            s = str(Codes[a.GetIdx()]) + ',' + s
            s = str(bond.GetBondType()) + '|' + s
            temp_sig.append(s)
            
    if len(temp_sig) < 1:
        return '' # no signature because no neighbors
    temp_sig = sorted(temp_sig)
    sig_neighbors = '.'.join(s for s in temp_sig)  
    signature = signature + '.' + sig_neighbors
        
    return  signature

###############################################################################
# Signature Callable functions
###############################################################################

def GetMoleculeSignature(mol, radius=2, nBits=0, neighbors=False,
                         useRange=False, isomericSmiles=False,
                         kekuleSmiles=False, 
                         allHsExplicit=False,
                         verbose=False):
# Callable function
# ARGUMENTS:
# mol: the molecule in rdkit format
# radius: the raduis of the signature, when radius < 0 the radius is set 
#         to the size the molecule
# nBits: number of bits for Morgan bit vector, when = 0 (default)
#        Morgan bit vector is not computed
# neighbors: if true radius-1 signature is computed on the neighbors of the atoms
# UseRange: (optional) , when useRange is True the signature is computed 
# from 0 to radius 
# isomericSmiles: (optional) include information about stereochemistry 
#                 in the SMILES. 
# kekuleSmiles: (optional) use the Kekule form (no aromatic bonds) 
#               in the SMILES. 
# allHsExplicit: (optional) if true, all H counts will be explicitly 
#                indicated in the output SMILES.
#
# RETURNS:
# A signature (string) where atom signatures are sorted in lexicographic 
# order and separated by ' '
# see GetAtomSignature for atom signatures format

    signature, temp_signature = '', []
    
    # First get radius and Morgan bits for all atoms 
    bitInfo = {}
    nB = 2048 if nBits == 0 else nBits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                               nBits=nB, 
                                               bitInfo=bitInfo,
                                               useChirality=isomericSmiles,
                                               useFeatures=True)
    Radius = [-1] * mol.GetNumAtoms()
    Morgan = [-1] * mol.GetNumAtoms()
    Codes =  [] 
    for bit, info in bitInfo.items():
        for atmidx, rad in info:
            if rad > Radius[atmidx]:
                Radius[atmidx] = rad
                Morgan[atmidx] = bit
    for i in range(len(Radius)):
        atm = mol.GetAtomWithIdx(i)
        code = str(int(Radius[i])) + ',' + str(atm.GetAtomicNum())
        code = code + ',' + str(Morgan[i]) if nBits > 0 else code
        Codes.append(code)
    
    # We compoute atom signatures for all atoms
    for atm in mol.GetAtoms():
        if atm.GetAtomicNum() == 1 and atm.GetFormalCharge() == 0:
            continue

        # We compute atom signature for atm
        sig = GetAtomSignature(atm,
                               Codes=Codes,
                               radius=radius,
                               neighbors=neighbors,
                               useRange=useRange,
                               isomericSmiles=isomericSmiles,
                               kekuleSmiles=kekuleSmiles,
                               allHsExplicit=allHsExplicit,
                               verbose=verbose)
        if sig != '':
            temp_signature.append(sig)
            
    # collect the signature for all atoms
    if len(temp_signature) < 1:
        return signature
    temp_signature = sorted(temp_signature)
    signature = ' '.join(sig for sig in temp_signature) 
    
    return  signature

