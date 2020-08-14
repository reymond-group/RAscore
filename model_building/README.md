## Installation 

Follow the steps in the defined order to avoid conflicts.

1. Create an environment (note: the python version must be >=3.7):\
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

If you want to retrain models, or train your own models using the hyperparameter optimisation framework found in the 'model_building' folder, then the following should be installed in the environemnt also:\
`pip install -r requirements.txt`

The SYBA, SCscore and SAscore should also be downloaded for descriptor calculations and training scripts modified to reflect the locations of the models:
* https://github.com/lich-uct/syba
* https://github.com/connorcoley/scscore
* https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score

## Training Models
### Configuration Files
The folder *example_classification_configs* contains example configuration files to train a variety of classifiers.\
The configuration file should be changed to reflect the paths of the training and test sets, and the output directory.\
A variety of descriptors have been implemented as found in the file *descriptors.py*. These can be specified in the configuration file.\

For instance consider the configuration file below:
* Specify: 
    * the path to the training data (.csv)
    * the path to the test data (.csv)
    * the path to the output directory
    * number of jobs for multiprocessing (for sklearn API)
    * number of cross vlaidation trails
    * training set and validation set size (test set) for internal cross validation
    * the descriptor:
        * ecfp_counts: counted extended connectivity fingerprint of radius 3
        * fcfp_counts: counted extended connectivity fingerprint of radius 3 with features
        * ecfp: extended connectivity fingerprint of radius 3 as a bit vector
        * maccs_keys: MACCS keys
        * features: molecular features as computed by RDkit i.e. molecular quantum numbers, number of rotatable bonds, molecualr weight, fraction of sp3 centres, number of heterocycles etc.
        * SA_score: Synthetic Accessibility score as published by Ertl et. al.
        * SC_score: Synthetic Complexity score as published by Coley et. al.
        * QED_score: Quantitative Estimate of Drug-Likeness as published by Bickerton et. al.
        * SYBA_score: Bayesian estimation of synthetic accessibility as published by Voršilák et. al.
        * custom descriptors can be added by implementing the descriptor in the *descriptors.py* file and adding a reference to both the optimiser and builder scripts
    * the algorithm:
        * AdaBoostClassifier
        * RandomForestClassifier
        * SVC
        * LogisticRegression
        * RidgeClassifier
        * BaggingClassifier
        * ModifiedNB
        * XGBClassifier
    * if running a feed forward neural network classifier specify and check out the example config file for optional parameters to explore:
        * DNNClassifier

```
{
"train_data": "<path-to-dataset>/train.csv",
"test_data": "<path-to-dataset>/test.csv",
"out_dir": "<path-to-output-directory>/RandomForestClassifier/",
"n_jobs": 6, 
"cross_validation": 5,
"train_size": 0.9,
"test_size": 0.1,
"descriptor": "fcfp_counts",
"n_trials": 500,
"algorithm": {"RandomForestClassifier": {
    "max_depth": {
        "low": 10,
        "high": 20
      },
      "n_estimators": {
        "low": 10,
        "high": 100
      },
      "max_features": ["sqrt", "log2"]
    }
}
}
```

### Optimisation
*Note: The hyperparameter optimisation does not use the test set specified but splits the training set into training and validation. The test set is then held out*\
To optimise and find favourable hyperparameters call:\
`python optimiser.py --config <path-to-config-file>`

or for the feed forward neural network call:\
`python optimiser_dnn.py --config <path-to-config-file>`

### Building 
Once optimisation is complete a .csv file containing the optimisation trials and a .json containing the best parameters and best values found will be present in the output folder specified in the config file.

*Note: Building the model uses the test set specified and has not been seen previously during optimisation*

To optimise and find favourable hyperparameters call:\
`python optimiser.py --config <path-to-config-file> --best_params <path-to-best-parameter-file>`

or for the feed forward neural network call:\
`python optimiser_dnn.py --config <path-to-config-file> --best_params <path-to-best-parameter-file>`

