import os
import pandas as pd
import numpy as np
import swifter
import json
import pickle
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sys
import descriptors
from syba.syba import SybaClassifier

import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn.linear_model
from sklearn.metrics import accuracy_score, auc, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler

from ModifiedNB import ModifiedNB

from xgboost import XGBClassifier

import optuna

import sys
sys.path.insert(1, '/home/knwb390/Projects/other/scscore/scscore')
from standalone_model_numpy import SCScorer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter optimisation for a model specified in the configuration file and return the best parameters as well as trail history')
    parser.add_argument('-c', '--config', type = str, default = None,
                        help = 'Specify the absolute path to the configuration file (.json)')
    parser.add_argument('-p', '--best_params', type = str, default = None,
                        help = 'Specify the absolute path to the best parameters file (.json) from the optimisation')
    args = parser.parse_args()

config = args.config
best_params = args.best_params

with open(config) as json_file:
    conf = json.load(json_file)
    
with open(best_params) as json_file:
    best_params = json.load(json_file)

out_dir = conf['out_dir'] 
viz_folder = out_dir+'viz/'
metrics_file = out_dir+'metrics.json'
model_file = out_dir+'model.pkl'

if os.path.exists(out_dir):
       pass
else: 
    os.mkdir(out_dir)

if os.path.exists(viz_folder):
       pass
else: 
    os.mkdir(viz_folder)

train_data = pd.read_csv(conf['train_data'], index_col=0)
if 'dataset' in train_data.columns:
    train_data.drop(columns='dataset', inplace=True)

test_data = pd.read_csv(conf['test_data'], index_col=0)
if 'dataset' in test_data.columns:
    test_data.drop(columns='dataset', inplace=True)

if conf['descriptor'] == 'ecfp_counts':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=False, useCounts=True)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=False, useCounts=True)
elif conf['descriptor'] == 'ecfp':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.ecfp, radius=3, nBits=2048)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.ecfp, radius=3, nBits=2048)
elif conf['descriptor'] == 'fcfp_counts':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
elif conf['descriptor'] == 'maccs_keys':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.MACCS_keys)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.MACCS_keys)
elif conf['descriptor'] == 'features':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.feature_fp)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.feature_fp)
elif conf['descriptor'] == 'SA_score':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.sa_score)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.sa_score)
elif conf['descriptor'] == 'SC_score':
    sc_model = SCScorer()
    sc_model.restore(os.path.join('/home/knwb390/Projects/other/scscore/', 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.sc_score, sc_model=sc_model)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.sc_score, sc_model=sc_model)
elif conf['descriptor'] == 'QED_score':
    train_data['descriptor'] = train_data['smi'].swifter.apply(descriptors.QEDScore)
    test_data['descriptor'] = test_data['smi'].swifter.apply(descriptors.QEDScore)
elif conf['descriptor'] == 'SYBA_score':
    syba = SybaClassifier()
    syba.fitDefaultScore()
    train_data['descriptor'] = train_data['smi'].swifter.apply(syba.predict)
    test_data['descriptor'] = test_data['smi'].swifter.apply(syba.predict)
else:
    print('Descriptor not recognised')

train_X = np.stack(train_data['descriptor'].values)
train_y = np.stack(train_data['activity'].values)

test_X = np.stack(test_data['descriptor'].values)
test_y = np.stack(test_data['activity'].values)

if isinstance(train_X[0], float):
    train_X = np.array([[i] for i in train_X])
    test_X = np.array([[i] for i in test_X])

if conf['descriptor'] == 'features':
    scaler = RobustScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

algorithm = list(conf['algorithm'].keys())[0]
    
if algorithm == 'RandomForestClassifier':
    classifier_obj = sklearn.ensemble.RandomForestClassifier(
        max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'], max_features=best_params['max_features']
    )

elif algorithm == 'AdaBoostClassifier':
    classifier_obj = sklearn.ensemble.AdaBoostClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'])

elif algorithm == 'SVC':
    classifier_obj =sklearn.svm.SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])

elif algorithm == 'LogisticRegression':
    classifier_obj = sklearn.linear_model.LogisticRegression(C=best_params['C'], solver=best_params['solver'])

elif algorithm == 'RidgeClassifier':
    classifier_obj = sklearn.linear_model.RidgeClassifier(alpha=best_params['alpha'], solver=best_params['solver'])

elif algorithm == 'BaggingClassifier':
    classifier_obj = sklearn.ensemble.BaggingClassifier(n_estimators=best_params['n_estimators'], max_samples=best_params['max_samples'])

elif algorithm == 'ModifiedNB':
    classifier_obj = ModifiedNB(alpha=best_params['alpha'])

elif algorithm == 'XGBClassifier':
    classifier_obj = XGBClassifier(max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'])

else:
    print('Classifier not specified')

scoring = ('accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc')

classifier_obj.fit(train_X, train_y)                            

predicted_y = classifier_obj.predict(test_X)
if algorithm == 'RidgeClassifier':
    scores_y = classifier_obj.decision_function(test_X)
else:
    scores_y = classifier_obj.predict_proba(test_X)
    scores_y = np.array([s[1] for s in scores_y])

#For sklearn 21

results = {
    'accuracy': str(round(accuracy_score(test_y, predicted_y), 2)),
    'balanced_accuracy_score': str(round(balanced_accuracy_score(test_y, predicted_y), 2)),
    'f1_score': str(round(f1_score(test_y, predicted_y), 2)),
    'matthews_corrcoef': str(round(matthews_corrcoef(test_y, predicted_y), 2)),
    'precision_score': str(round(precision_score(test_y, predicted_y), 2)),
    'average_precision': str(round(sklearn.metrics.average_precision_score(test_y, scores_y), 2)),
    'recall_score': str(round(recall_score(test_y, predicted_y), 2)),
    'roc_auc': str(round(sklearn.metrics.roc_auc_score(test_y, scores_y), 2))
          }

"""
#For sklearn 22.2
roc_curve = plot_roc_curve(classifier_obj, test_X, test_y)
plt.savefig(viz_folder+'roc_auc_curve.png')

prec_rec_curve = plot_precision_recall_curve(classifier_obj, test_X, test_y)
plt.savefig(viz_folder+'precision_recall_curve.png')

results = {
    'accuracy': round(accuracy_score(test_y, predicted_y), 2),
    'balanced_accuracy_score': round(balanced_accuracy_score(test_y, predicted_y), 2),
    'f1_score': round(f1_score(test_y, predicted_y), 2),
    'matthews_corrcoef': round(matthews_corrcoef(test_y, predicted_y), 2),
    'precision_score': round(precision_score(test_y, predicted_y), 2),
    'average_precision': round(prec_rec_curve.average_precision, 2),
    'recall_score': round(recall_score(test_y, predicted_y), 2),
    'roc_auc': round(roc_curve.roc_auc, 2),
    'true positives': float(tp),
    'true negatives': float(tn),
    'false positives': float(fp),
    'false negatives': float(fn)
          }
"""
with open(metrics_file, 'w') as outfile:
    json.dump(results, outfile)

with open(model_file, 'wb') as outfile:
    pickle.dump(classifier_obj, outfile)