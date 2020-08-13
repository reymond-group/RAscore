import os
import numpy as np

from tensorflow import keras
import tensorflow as tf

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs

class RAScorer:
    """
    Prediction of machine learned retrosynthetic accessibility score
    The RAScore is calculated based on the predictions made on 200,000 compounds sampled from ChEMBL.
    The compounds were subjected to retrosynthetic analysis using a CASP tool (AiZynthfinder) and output used as labels to train a binary classifier.

    This class facilitates predictions from the resulting model
    """

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
    
    def ecfp_counts(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True) 
        size = 2048
        arr = np.zeros((size,), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        return arr
    
    def predict(self, smiles):
        arr = self.ecfp_counts(smiles)
        proba = self.model.predict(arr.reshape(1, -1))
        return proba[0][0]
