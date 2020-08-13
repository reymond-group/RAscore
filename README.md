# Retrosynthetic Accessibility (RA) score
 * RAscore is a score learned from the predictions of a computer aided synthesis planning tool (AiZynthfinder: https://github.com/MolecularAI/aizynthfinder). 
 * RAscore is intended to be a binary score, indicating whether the underlying computer aided synthesis planning tool can find a route (1) or not (0) to a given compound. 
 * The tool has been trained on 200,000 compounds from ChEMBL and so is limited to compounds within similar regions of chemical space. It is intended to predict the retrosyntehtic accessibility of bioactive molecules.
 * Attempts to use the score on more exotic compounds such as those found in the GDB databases will not work.

## Installation 

Follow the steps in the defined order to avoid conflicts.

1. Create an environment:
`conda create --name myenv`

or use an existing environment 

2. Install rdkit 2020.03 and tensorflow 2 (if already installed skip this step)
`conda install -c rdkit rdkit -y`
`conda install -c anaconda tensorflow>=2.1.0 -y`

3. Clone the RAscore repository 
`git clone https://github.com/reymond-group/RAscore.git`

change directory to the repository
`cd RAscore`

4. Install RAscore
`python -m pip install -e .`

## Usage
```
from RAscore import RAscore
scorer = RAscore.RAScorer('<path-to-repo>/RAscore/RAscore/model/model.h5')
scorer.predict('CC1=C(C(=CC=C1)C)N(CC(=O)NC2=CC=C(C=C2)C3=NOC=N3)C(=O)C4CCS(=O)(=O)CC4')
```
