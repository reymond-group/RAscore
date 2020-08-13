## Installation 

Follow the steps in the defined order to avoid conflicts.

1. Create an environment (note: the python version must be >=3.7):
`conda create --name myenv python=3.7`

or use an existing environment 

2. Install rdkit 2020.03 and tensorflow 2 (if already installed skip this step)
```
conda install -c rdkit rdkit -y
conda install -c anaconda tensorflow>=2.1.0 -y
```

3. Clone the RAscore repository 
`git clone https://github.com/reymond-group/RAscore.git`

change directory to the repository
`cd RAscore`

4. Install RAscore
`python -m pip install -e .`

If you want to retrain models, or train your own models using the hyperparameter optimisation framework found in the 'model_building' folder, then the following should be installed in the environemnt also:
`pip install -r requirements.txt`

The SYBA, SCscore and SAscore should also be downloaded for descriptor calculations and training scripts modified to reflect the locations of the models:
https://github.com/lich-uct/syba
https://github.com/connorcoley/scscore
https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score

## Training Models

The folder /example_classification_configs contains example configuration files to train a variety of classifiers. The configuration file should be changed to reflect the paths of the training and test sets, and the output directory. A variety of descriptors have been implemented as found in the file 'descriptors.py'. These can be specified in the configuration file.