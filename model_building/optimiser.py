import os
import pandas as pd
import numpy as np
import swifter
import json
import pickle
import argparse

import sys
import descriptors
from syba.syba import SybaClassifier

import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn.linear_model
from sklearn.preprocessing import RobustScaler

from ModifiedNB import ModifiedNB

from xgboost import XGBClassifier

import optuna

import sys
sys.path.insert(1, '/home/knwb390/Projects/other/scscore/scscore')
from standalone_model_numpy import SCScorer

class Objective(object):
    def __init__(self, x, y, conf):
        # Hold this implementation specific arguments as the fields of the class.
        self._train_X = x
        self._train_y = y
        self._conf = conf
    
    def __call__(self, trial):
        return self._classifier(trial)
    
    def _classifier(self, trial):
        algorithm = list(self._conf['algorithm'].keys())[0]
    
        if algorithm == 'RandomForestClassifier':
            rf_max_depth = trial.suggest_int("max_depth", self._conf['algorithm'][algorithm]['max_depth']['low'], self._conf['algorithm'][algorithm]['max_depth']['high'])
            rf_n_estimators = trial.suggest_int("n_estimators", self._conf['algorithm'][algorithm]['n_estimators']['low'], self._conf['algorithm'][algorithm]['n_estimators']['high'])
            rf_max_features = trial.suggest_categorical('max_features', self._conf['algorithm'][algorithm]['max_features'])
            
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_features
            )
       
        elif algorithm == 'AdaBoostClassifier':
            learning_rate = trial.suggest_float("learning_rate", self._conf['algorithm'][algorithm]['learning_rate']['low'], self._conf['algorithm'][algorithm]['learning_rate']['high'])
            n_estimators = trial.suggest_int("n_estimators", self._conf['algorithm'][algorithm]['n_estimators']['low'], self._conf['algorithm'][algorithm]['n_estimators']['high'])
            
            classifier_obj = sklearn.ensemble.AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
            
        elif algorithm == 'SVC':
            c = float(trial.suggest_loguniform("C", self._conf['algorithm'][algorithm]['C']['low'], self._conf['algorithm'][algorithm]['C']['high']))
            kernel = trial.suggest_categorical('kernel', self._conf['algorithm'][algorithm]['kernel'])
            gamma = float(trial.suggest_loguniform("gamma", self._conf['algorithm'][algorithm]['gamma']['low'], self._conf['algorithm'][algorithm]['gamma']['high']))
                    
            classifier_obj =sklearn.svm.SVC(C=c, kernel=kernel, gamma=gamma)
            
        elif algorithm == 'LogisticRegression':
            c = float(trial.suggest_loguniform("C", self._conf['algorithm'][algorithm]['C']['low'], self._conf['algorithm'][algorithm]['C']['high']))
            solver = trial.suggest_categorical('solver', self._conf['algorithm'][algorithm]['solver'])
            
            classifier_obj = sklearn.linear_model.LogisticRegression(C=c, solver=solver)
            
        elif algorithm == 'RidgeClassifier':
            alpha = float(trial.suggest_loguniform("alpha", self._conf['algorithm'][algorithm]['alpha']['low'], self._conf['algorithm'][algorithm]['alpha']['high']))
            solver = trial.suggest_categorical('solver', self._conf['algorithm'][algorithm]['solver'])
            
            classifier_obj = sklearn.linear_model.RidgeClassifier(alpha=alpha, solver=solver)
            
        elif algorithm == 'BaggingClassifier':
            n_estimators = trial.suggest_int("n_estimators", self._conf['algorithm'][algorithm]['n_estimators']['low'], self._conf['algorithm'][algorithm]['n_estimators']['high'])
            max_samples = trial.suggest_int("max_samples", self._conf['algorithm'][algorithm]['max_samples']['low'], self._conf['algorithm'][algorithm]['max_samples']['high'])
            
            classifier_obj = sklearn.ensemble.BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples)
           
        elif algorithm == 'ModifiedNB':
            alpha = float(trial.suggest_loguniform("alpha", self._conf['algorithm'][algorithm]['alpha']['low'], self._conf['algorithm'][algorithm]['alpha']['high']))
            classifier_obj = ModifiedNB(alpha=alpha)
        
        elif algorithm == 'XGBClassifier':
            max_depth = trial.suggest_int("max_depth", self._conf['algorithm'][algorithm]['max_depth']['low'], self._conf['algorithm'][algorithm]['max_depth']['high'])
            n_estimators = trial.suggest_int("n_estimators", self._conf['algorithm'][algorithm]['n_estimators']['low'], self._conf['algorithm'][algorithm]['n_estimators']['high'])
            learning_rate = trial.suggest_uniform("learning_rate", self._conf['algorithm'][algorithm]['learning_rate']['low'], self._conf['algorithm'][algorithm]['learning_rate']['high'])
            
            classifier_obj = XGBClassifier(
                max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate 
            )
            
        else:
            print('Classifier not specified')
            
        cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=self._conf['cross_validation'], \
                                                            test_size=self._conf['test_size'], \
                                                            train_size=self._conf['train_size'])
        
        scoring = ('accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc')
        result = sklearn.model_selection.cross_validate(estimator=classifier_obj,
                                                        X=self._train_X,
                                                        y=self._train_y,
                                                        n_jobs=self._conf['n_jobs'],
                                                        cv=cv,
                                                        scoring=scoring,
                                                        return_train_score=True)

        score = result['test_roc_auc'].mean()
        
        try:
            with open(self._conf['out_dir']+'best_params.json', 'w') as outfile:
                json.dump(study.best_params, outfile)

            with open(self._conf['out_dir']+'best_value.txt', 'w') as outfile:
                outfile.write("Best Trial Value: {}".format(study.best_value))
        except:
            pass
            
        return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter optimisation for a model specified in the configuration file and return the best parameters as well as trail history')
    parser.add_argument('-c', '--config', type = str, default = None,
                        help = 'Specify the absolute path to the configuration file (.json)')
    args = parser.parse_args()

config = args.config

with open(config) as json_file:
    conf = json.load(json_file)
    
out_dir = conf['out_dir'] 

best_params_file = out_dir+'best_params.json'
best_value_file = out_dir+'best_value.txt'
out_df_file = out_dir+'trails.csv'
viz_folder = out_dir+'viz/'

if os.path.exists(out_dir):
       pass
else: 
    os.mkdir(out_dir)

if os.path.exists(viz_folder):
       pass
else: 
    os.mkdir(viz_folder)
    
data = pd.read_csv(conf['train_data'], index_col=0)
if 'dataset' in data.columns:
    data.drop(columns='dataset', inplace=True)

if conf['descriptor'] == 'ecfp_counts':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=False, useCounts=True)
elif conf['descriptor'] == 'ecfp':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.ecfp, radius=3, nBits=2048)
elif conf['descriptor'] == 'fcfp_counts':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
elif conf['descriptor'] == 'maccs_keys':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.MACCS_keys)
elif conf['descriptor'] == 'features':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.feature_fp)
elif conf['descriptor'] == 'SA_score':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.sa_score)
elif conf['descriptor'] == 'SC_score':
    sc_model = SCScorer()
    sc_model.restore(os.path.join('/home/knwb390/Projects/other/scscore/', 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
    data['descriptor'] = data['smi'].swifter.apply(descriptors.sc_score, sc_model=sc_model)
elif conf['descriptor'] == 'QED_score':
    data['descriptor'] = data['smi'].swifter.apply(descriptors.QEDScore)
elif conf['descriptor'] == 'SYBA_score':
    syba = SybaClassifier()
    syba.fitDefaultScore()
    data['descriptor'] = data['smi'].swifter.apply(syba.predict)
else:
    print('Descriptor not recognised')

X = np.stack(data['descriptor'].values)
Y = np.stack(data['activity'].values)

if isinstance(X[0], float):
    X = np.array([[i] for i in X])

if conf['descriptor'] == 'features':
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

study = optuna.create_study(direction="maximize")
objective = Objective(X, Y, conf)
study.optimize(objective, n_trials=conf['n_trials'])

with open(best_params_file, 'w') as outfile:
    json.dump(study.best_params, outfile)
    
df = study.trials_dataframe()
df.to_csv(out_df_file)

with open(best_value_file, 'w') as outfile:
    outfile.write("Best Trial Value: {}".format(study.best_value))

"""
history = optuna.visualization.plot_optimization_history(study)
history.write_image(viz_folder+'history.png')
img_bytes = history.to_image(format="png")
with open(viz_folder+'history.pkl', 'wb') as handle:
    pickle.dump(img_bytes, handle, protocol=pickle.HIGHEST_PROTOCOL)

slice_plot = optuna.visualization.plot_slice(study, params=[key for key in study.best_params.keys()])
slice_plot.write_image(viz_folder+'slice_plot.png')
img_bytes = slice_plot.to_image(format="png")
with open(viz_folder+'slice_plot.pkl', 'wb') as handle:
    pickle.dump(img_bytes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study, params=[key for key in study.best_params.keys()])
parallel_coordinate.write_image(viz_folder+'parallel_coordinate.png')
img_bytes = parallel_coordinate.to_image(format="png")
with open(viz_folder+'parallel_coordinate.pkl', 'wb') as handle:
    pickle.dump(img_bytes, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""