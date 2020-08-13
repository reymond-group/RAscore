import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, MACCSkeys, QED
from rdkit.DataStructs import cDataStructs

import sys
sys.path.insert(1, '/home/knwb390/Projects/other/')
import SA_Score.sascorer as sascorer

sys.path.insert(1, '/home/knwb390/Projects/other/scscore/scscore')
from standalone_model_numpy import SCScorer

def ecfp(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((0,), dtype=np.int32)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def ecfp_counts(smiles, radius=3, useFeatures=False, useCounts=True):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) 
    size = 2048
    arr = np.zeros((size,), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        arr[nidx] += int(v)
    return arr

def ecfp_features(smiles, radius=3, useFeatures=True, useCounts=False):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) 
    size = 2048
    arr = np.zeros((size,), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        arr[nidx] += int(v)
    return arr

def MACCS_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    arr = np.zeros((0,), dtype=np.int32)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def feature_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.MQNs_(mol)
    
    fp.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    fp.append(rdMolDescriptors.CalcExactMolWt(mol))
    fp.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    fp.append(rdMolDescriptors.CalcFractionCSP3(mol))
    fp.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAliphaticRings((mol)))
    fp.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    fp.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    fp.append(rdMolDescriptors.CalcNumRings(mol))
    fp.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    fp.append(rdMolDescriptors.CalcNumHeterocycles(mol))
    fp.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    fp.append(rdMolDescriptors.CalcTPSA(mol))
    
    return np.array(fp)

def sa_score(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return sascorer.calculateScore(mol)
    except:
        return np.nan

def QEDScore(smi):
     try:
        mol = Chem.MolFromSmiles(smi)
        return QED.qed(mol)
     except:
        return np.nan

def sc_score(smi, sc_model):
    try:
        (smiles, sc_score) = sc_model.get_score_from_smi(smi)
        return sc_score
    except:
        return np.nan