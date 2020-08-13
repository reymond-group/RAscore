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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

from sklearn.preprocessing import RobustScaler

import optuna

import sys
sys.path.insert(1, '/home/knwb390/Projects/other/scscore/scscore')
from standalone_model_numpy import SCScorer

def build_model(train_X, best_params):
    layers = []

    inputs = tf.keras.Input(shape=(train_X.shape[1],))
    layers.append(inputs)

    x = Dense(best_params['layer_1'], activation=best_params['activation_1'])(layers[-1])
    layers.append(x)

    for l in range(2, best_params['num_layers']+1):
        x = Dense(best_params['units_{}'.format(l)], activation=best_params['activation_{}'.format(l)])(layers[-1])
        layers.append(x)
        x = Dropout(round(best_params["dropout_{}".format(l)], 1))(layers[-1])
        layers.append(x)

    out = Dense(1, activation='sigmoid', name = 'target')(layers[-1])

    model = Model(inputs=[inputs], outputs=[out])
            
    learning_rate = best_params['learning_rate']

    metrics=[tf.keras.metrics.AUC(),
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives()
            ]

    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(lr=learning_rate),
                  metrics=metrics)
    
    return model

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
model_file = out_dir+'model.h5'

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
    train_y = np.array([[i] for i in train_y])

if conf['descriptor'] == 'features':
    scaler = RobustScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

algorithm = list(conf['algorithm'].keys())[0]

results = {}
    
model = build_model(train_X, best_params)

model.fit(train_X,
            train_y,
            validation_split=conf['test_size'],
            shuffle=True,
            batch_size=conf['batch_size'],
            epochs=conf['epochs'],
            verbose=False
            )
    
score = model.evaluate(test_X, 
                        test_y, 
                        verbose=0)

model.save(model_file)
    
for metric, s in zip(model.metrics_names, score):
    if metric in results.keys():
        results[metric].append(s)
    else:
        results[metric] = []
        results[metric].append(s)
            
for metric, value in results.items():
    results[metric] = round(np.array(value).mean(),2)

with open(metrics_file, 'w') as outfile:
    json.dump(str(results), outfile)