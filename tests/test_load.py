"""Test the default models can be loaded."""

import unittest

from RAscore import RAscore_NN, RAscore_XGB


class TestLoad(unittest.TestCase):
    """Test the default models can be loaded."""

    def test_nn(self):
        """Test loading the tensorflow/keras models."""
        nn_scorer = RAscore_NN.RAScorerNN()

    def test_xgb(self):
        """Test loading the XGB models."""
        xgb_scorer = RAscore_XGB.RAScorerXGB()
