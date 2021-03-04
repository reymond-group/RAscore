import os
from zipfile import ZipFile
import numpy as np

from tensorflow import keras
import tensorflow as tf

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs


class RAScorerNN:
    """
    Prediction of machine learned retrosynthetic accessibility score
    The RAScore is calculated based on the predictions made on 200,000 compounds sampled from ChEMBL.
    The compounds were subjected to retrosynthetic analysis using a CASP tool (AiZynthfinder) and output used as labels to train a binary classifier.

    If the compounds are ChEMBL like use the RAscore models.
    Else if the compounds are more exotic, are small fragments, or closely resemble GDB. GBDscore may give a better result.

    This class facilitates predictions from the resulting model.
    """

    def __init__(self, model_path=None):
        """
        Loads the model.

        :param model_path: path to the neural network model (.h5) file
        """
        HERE = os.path.abspath(os.path.dirname(__file__))
        MODEL = os.path.join(HERE, "models/DNN_chembl_fcfp_counts/model.tf")
        if model_path == None:
            self.nn_model = keras.models.load_model(MODEL)
        else:
            self.nn_model = keras.models.load_model(model_path)

    def ecfp_counts(self, smiles):
        """
        Converts SMILES into a counted ECFP6 vector with features.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: ECFP6 counted vector with features
        :rtype: np.array
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        arr = np.zeros((size,), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        return arr

    def predict(self, smiles):
        """
        Predicts score from SMILES.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: score
        :rtype: float
        """
        try:
            arr = self.ecfp_counts(smiles)
        except ValueError:
            print("SMILES could not be converted to ECFP6 count vector")
            return float("NaN")

        try:
            proba = self.nn_model.predict(arr.reshape(1, -1))
            return proba[0][0]
        except:
            print("Prediction not possible")
            return float("NaN")

