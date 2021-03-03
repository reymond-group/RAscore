"""Test the default models can be loaded."""

import unittest


class TestLoad(unittest.TestCase):
    """Test the default models can be loaded."""

    def test_import_rascore(self):
        """Test import is possible."""
        import RAscore
        print(RAscore)

    def test_import_rdkit(self):
        """Test importing RDKit works."""

    def test_nn(self):
        """Test loading the tensorflow/keras models."""
        from RAscore import RAscore_NN

        nn_scorer = RAscore_NN.RAScorerNN()

    def test_xgb(self):
        """Test loading the XGB models."""
        from RAscore import RAscore_XGB

        xgb_scorer = RAscore_XGB.RAScorerXGB()
