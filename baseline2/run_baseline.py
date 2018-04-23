#Author: Maurice Weber

import numpy as np
import json
import h5py

import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from scipy import signal

from utils import *


#loading configuration
with open('config.json') as json_data_file:
	config = json.load(json_data_file)


## read params from config file
# training params
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]
MAX_DEPTH = config["run_baseline"]["MAX_DEPTH"]
MAX_FEATURES = config["run_baseline"]["MAX_FEATURES"]
N_TREES = config["run_baseline"]["N_TREES"]

# db params
PCA_COMPONENTS = config["create_db"]["PCA_COMPONENTS"]
HOP = config["create_db"]["HOP"]
N_FFT = config["create_db"]["N_FFT"]
WINDOW_SIZE = config["create_db"]["WINDOW_SIZE"]
DEVICE_FILTER = config["create_db"]["DEVICE_FILTER"]

# DB file
DB_FILENAME = '/data_DEVICES=[%s]_PCA=%i_NFFT=%i_HOP=%i_WINDOW=%f.h5'%\
				("-".join(DEVICE_FILTER), PCA_COMPONENTS, N_FFT, HOP, WINDOW_SIZE)

print('Data used from %s'%(DB_ROOT_DIR + DB_FILENAME))

# get data
with h5py.File(DB_ROOT_DIR + DB_FILENAME, 'r') as hf:

	train_devices = hf['train_devices'][:].astype('U13').tolist()
	train_labels = hf['train_data'][:,0]
	train_features = hf['train_data'][:,1:]

	test_devices = hf['test_devices'][:].astype('U13').tolist()
	test_labels = hf['test_data'][:,0]
	test_features = hf['test_data'][:,1:]


## fit random forest
rf = RandomForestClassifier(n_estimators = N_TREES,
							max_features=MAX_FEATURES,
							max_depth=MAX_DEPTH,
							random_state=0)
rf.fit(train_features, train_labels)

# predictions
train_pred = rf.predict(train_features)
test_pred = rf.predict(test_features)

## get figures for entire data set
train_accuracy = sklearn.metrics.accuracy_score(y_true=train_labels, y_pred=train_pred)
test_accuracy = sklearn.metrics.accuracy_score(y_true=test_labels, y_pred=test_pred)
aucroc_score_train = sklearn.metrics.roc_auc_score(train_labels, train_pred)
aucroc_score_test = sklearn.metrics.roc_auc_score(test_labels, test_pred)
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=test_labels, y_pred=test_pred).ravel()
specificity_score = tn/(tn + fp)
sensitivity_score = tp/(tp + fn)


print('#'*100, "\n")

## print hyperparams
print('----------------------------------------------------------- HYPERPARAMS -----------------------------------------------------------\n')
print('N_TREES = %i\nMAX_FEATUREs = %i\nMAX_DEPTH = %i\n'%(N_TREES, MAX_FEATURES, MAX_DEPTH))

## start printing results
print('----------------------------------------------------------- device: %s -----------------------------------------------------------'%DEVICE_FILTER)
print('Results - specificity score: %f, sensitivity score: %f, test accuracy: %f, train accuracy: %f \n'\
	%(specificity_score, sensitivity_score, test_accuracy, train_accuracy))


## get figures for each device
for device in DEVICE_FILTER:

	if device == "audio track":
		continue

	print('----------------------------------------------------------- device: %s -----------------------------------------------------------'%device)

	# train figures
	train_indexes = np.where(np.isin(train_devices, device))[0]
	assert len(train_indexes) > 0, "something went wrong on train set with device %s"%device
	train_pred_ = [train_pred[i] for i in train_indexes]
	train_labels_ = [train_labels[i] for i in train_indexes]
	train_accuracy = sklearn.metrics.accuracy_score(y_true=train_labels_, y_pred=train_pred_)
	aucroc_score_train = sklearn.metrics.roc_auc_score(train_labels_, train_pred_)
	
	# test figures
	test_indexes = np.where(np.isin(test_devices, device))[0]
	assert len(train_indexes) > 0, "something went wrong on test set with device %s"%device
	test_pred_ = [test_pred[i] for i in test_indexes]
	test_labels_ = [test_labels[i] for i in test_indexes]
	test_accuracy = sklearn.metrics.accuracy_score(y_true=test_labels_, y_pred=test_pred_)
	aucroc_score_test = sklearn.metrics.roc_auc_score(test_labels_, test_pred_)
	tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=test_labels_, y_pred=test_pred_).ravel()
	specificity_score = tn/(tn + fp)
	sensitivity_score = tp/(tp + fn)
	
	# print results
	print('Results - specificity score: %f, sensitivity score: %f, test accuracy: %f, train accuracy: %f \n'\
	%(specificity_score, sensitivity_score, test_accuracy, train_accuracy))



print('#'*100)

























