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
from sklearn.model_selection import StratifiedKFold, train_test_split

import optuna

import sys
sys.path.insert(1, '/home/knwb390/Projects/other/scscore/scscore')
from standalone_model_numpy import SCScorer

class Objective(object):
    def __init__(self, x, y, conf):
        # Hold this implementation specific arguments as the fields of the class.
        self._train_X, self._test_x, self._train_Y, self._test_y = train_test_split(x, y, train_size=conf['train_size'], test_size=conf['test_size'], random_state=42, shuffle=True)
        self._conf = conf
    
    def __call__(self, trial):
        return self._classifier(trial)
    
    def _build_model(self, trial, algorithm):
        layers = []
            
        inputs = tf.keras.Input(shape=(self._train_X.shape[1],))
        layers.append(inputs)
        
        x = Dense(
            trial.suggest_categorical("layer_1", self._conf['algorithm'][algorithm]['layer_1']),
            activation=trial.suggest_categorical("activation_1", 
                                                self._conf['algorithm'][algorithm]['activation_1']))(layers[-1])
        layers.append(x)
        
        x = Dropout(self._conf['algorithm'][algorithm]['dropout_1'])(layers[-1])
        layers.append(x)
                                    
        num_layers = trial.suggest_int("num_layers", 2, self._conf['algorithm'][algorithm]['max_layers'])
        for l in range(2, num_layers+1):
            x = Dense(
            trial.suggest_categorical("units_{}".format(l), self._conf['algorithm'][algorithm]['layer_size']),
            activation=trial.suggest_categorical("activation_{}".format(l), 
                                                self._conf['algorithm'][algorithm]['layer_activations']))(layers[-1])
            layers.append(x)
            x = Dropout(round(trial.suggest_float("dropout_{}".format(l), self._conf['algorithm'][algorithm]['layer_droput']['low'], self._conf['algorithm'][algorithm]['layer_droput']['high']), 1))(layers[-1])
            layers.append(x)
        
        out = Dense(1, activation='sigmoid', name = 'target')(layers[-1])
        
        model = Model(inputs=[inputs], outputs=[out])
        
        learning_rate = float(trial.suggest_loguniform("learning_rate", self._conf['algorithm'][algorithm]['learning_rate']['low'], self._conf['algorithm'][algorithm]['learning_rate']['high']))
    
        model.compile(loss='binary_crossentropy', 
                        optimizer=Adam(lr=learning_rate),
                        metrics=[tf.keras.metrics.AUC()])
        
        return model
    
    def _classifier(self, trial):
        algorithm = list(self._conf['algorithm'].keys())[0]
    
        scores =[]
        losses = []

        model = self._build_model(trial, algorithm)
    
        model.fit(self._train_X,
                    self._train_Y,
                    validation_split=self._conf['test_size'],
                    shuffle=True,
                    batch_size=self._conf['batch_size'],
                    epochs=self._conf['epochs'],
                    verbose=False
                    )
        
        score = model.evaluate(self._test_x, 
                                self._test_y, 
                                verbose=0)
        
        scores.append(score[1])
        losses.append(score[0])

        mean_score = np.array(scores).mean()
        #print(model.metrics_names)
        #print(score)

        try:
            with open(self._conf['out_dir']+'best_params.json', 'w') as outfile:
                json.dump(study.best_params, outfile)

            with open(self._conf['out_dir']+'best_value.txt', 'w') as outfile:
                outfile.write("Best Trial Value: {}".format(study.best_value))
        except:
            pass
                              
        return mean_score

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