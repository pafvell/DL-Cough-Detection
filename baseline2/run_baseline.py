#Author: Maurice Weber

import numpy as np
import json
import h5py
import time

import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from scipy import signal

from utils import *


#loading configuration
with open('config.json') as json_data_file:
	config = json.load(json_data_file)

t = time.time()

## read params from config file
# training params
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]
DB_VERSION = config["general"]["DB_VERSION"]
MAX_DEPTH = config["run_baseline"]["MAX_DEPTH"]
MAX_FEATURES = config["run_baseline"]["MAX_FEATURES"]
N_TREES = config["run_baseline"]["N_TREES"]

# db params
PCA_COMPONENTS = config["create_db"]["PCA_COMPONENTS"]
HOP = config["create_db"]["HOP"]
N_FFT = config["create_db"]["N_FFT"]
WINDOW_SIZE = config["create_db"]["WINDOW_SIZE"]
DEVICE_FILTER = config["create_db"]["DEVICE_FILTER"]
BATCH_SIZE = config["create_db"]["BATCH_SIZE"]


# DB File
DB_FILENAME = '/data_%s.h5'%(DB_VERSION)

print('#'*100, "\n")

print('Data used from %s'%(DB_ROOT_DIR + DB_FILENAME))

# get data
with h5py.File(DB_ROOT_DIR + DB_FILENAME, 'r') as hf:

	train_devices = hf['train_devices'][:].astype('U13').tolist()
	train_labels = hf['train_data'][:,0]
	train_features = hf['train_data'][:,1:]
	print("shape of train features: {train_shape}".format(train_shape=np.shape(train_features)))

	test_devices = hf['test_devices'][:].astype('U13').tolist()
	test_labels = hf['test_data'][:,0]
	test_features = hf['test_data'][:,1:]
	print("shape of test features: {test_shape}".format(test_shape=np.shape(test_features)))


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
cm = sklearn.metrics.confusion_matrix(y_true=test_labels, y_pred=test_pred)
specificity = cm[0,0]/(cm[0,0]+cm[0,1])  
sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
precision = cm[1,1]/(cm[0,1]+cm[1,1])

print('#'*100, "\n")

## print hyperparams
print('----------------- HYPERPARAMS -----------------\n')
print('N_TREES = %i\nMAX_FEATURES = %i\nMAX_DEPTH = %i\n'%(N_TREES, MAX_FEATURES, MAX_DEPTH))

## start printing results
print('----------------- device: %s -----------------'%DEVICE_FILTER)
print('RESULTS:')
print('test accuracy: %f'%test_accuracy)
print('train accuracy: %f'%train_accuracy)
print('sensitivity: %f'%sensitivity)
print('specificity: %f'%specificity)
print('precision: %f'%precision)


## get figures for each device
for device in DEVICE_FILTER:

	if device == "audio track":
		continue

	print('----------------- device: %s -----------------'%device)

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
	cm = sklearn.metrics.confusion_matrix(y_true=test_labels_, y_pred=test_pred_)
	specificity = cm[0,0]/(cm[0,0]+cm[0,1])  
	sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
	precision = cm[1,1]/(cm[0,1]+cm[1,1])
	
	# print results
	print('RESULTS:')
	print('test accuracy: %f'%test_accuracy)
	print('train accuracy: %f'%train_accuracy)
	print('sensitivity: %f'%sensitivity)
	print('specificity: %f'%specificity)
	print('precision: %f'%precision)



print('#'*100)


print('time elapsed: ', time.time() - t)






















