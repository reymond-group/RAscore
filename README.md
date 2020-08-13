# Retrosynthetic Accessibility (RA) score
 RAscore is a score learned from the predictions of a computer aided synthesis planning tool. It is intended to be a binary score, indicating whether the underlying computer aided synthesis planning tool can find a route (1) or not (0) to a given compound. The tool has been trained on 200,000 compounds from ChEMBL and so is limited to compounds within similar regions of chemical space.

## Installation 

Follow the steps in the defined order to avoid conflicts.

Install rdkit 2020.03 (if already installed skip this step)
conda install -c rdkit rdkit -y

Install tensorflow 
conda install -c anaconda tensorflow>=2.1.0 -y

Install RAscore
python -m pip install -e .

