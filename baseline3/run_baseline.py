#Author: Filipe Barata

import json
import time

import h5py
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from utils import *

#loading configuration
with open('config.json') as json_data_file:
	config = json.load(json_data_file)

t = time.time()

## read params from config file
# training params
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]
DB_VERSION = config["general"]["DB_VERSION"]
NNEIGHBOURS = config["run_baseline"]["NNEIGHBOURS"]
WEIGHT_TYPE = config["run_baseline"]["WEIGHT_TYPE"]
WEIGHT_METRIC = config["run_baseline"]["WEIGHT_METRIC"]

# db params

HOP = config["create_db"]["HOP"]
FILTERING = config["create_db"]["FILTERING"]
DOWNSAMPLING_FACTOR = config["create_db"]["DOWNSAMPLING_FACTOR"]
SAMPLING_RATE = config["create_db"]["SAMPLING_RATE"]
NPERSEG = config["create_db"]["NPERSEG"]
NOVERLAP = config["create_db"]["NOVERLAP"]
PAPER = config["create_db"]["PAPER"]
ALTERNATIVE_COMP = config["create_db"]["ALTERNATIVE_COMP"]
N_FFT = config["create_db"]["N_FFT"]
WINDOW_SIZE = config["create_db"]["WINDOW_SIZE"]
DEVICE_FILTER = config["create_db"]["DEVICE_FILTER"]


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


## fit knn
knn = KNeighborsClassifier(n_neighbors=NNEIGHBOURS, weights=WEIGHT_TYPE, p=WEIGHT_METRIC)

knn.fit(train_features, train_labels)

# predictions
train_pred = knn.predict(train_features)
test_pred = knn.predict(test_features)

# auc roc curve
probability_test = knn.predict_proba(test_features)
probVec = probability_test[:, 1]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_labels, probVec)
df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
df.to_csv("knn_roc_curve.csv")

## get figures for entire data set
train_accuracy = sklearn.metrics.accuracy_score(y_true=train_labels, y_pred=train_pred)
test_accuracy = sklearn.metrics.accuracy_score(y_true=test_labels, y_pred=test_pred)

aucroc_score_test = sklearn.metrics.roc_auc_score(test_labels, probability_test[:,1])

mcc_test = sklearn.metrics.matthews_corrcoef(test_labels, test_pred)
cm = sklearn.metrics.confusion_matrix(y_true=test_labels, y_pred=test_pred).astype(float)
specificity = cm[0,0]/(cm[0,0]+cm[0,1])  
sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
precision = cm[1,1]/(cm[0,1]+cm[1,1])

print('#'*100, "\n")

## print hyperparams
print('----------------- HYPERPARAMS -----------------\n')
print('NNEIGHBOURS = %i\nWEIGHT_METRIC = %s\nWEIGHT_TYPE = %i\n'% (NNEIGHBOURS, WEIGHT_TYPE, WEIGHT_METRIC))

## start printing results
print('----------------- device: %s -----------------'%DEVICE_FILTER)
print('RESULTS:')
print('test accuracy: %f'%test_accuracy)
print('train accuracy: %f'%train_accuracy)
print('sensitivity: %f'%sensitivity)
print('specificity: %f'%specificity)
print('precision: %f'%precision)
print('mcc: %f'%mcc_test)
print('auc: %f'%aucroc_score_test)

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

	
	# test figures
	test_indexes = np.where(np.isin(test_devices, device))[0]
	assert len(train_indexes) > 0, "something went wrong on test set with device %s"%device
	test_pred_ = [test_pred[i] for i in test_indexes]
	test_labels_ = [test_labels[i] for i in test_indexes]
	test_accuracy = sklearn.metrics.accuracy_score(y_true=test_labels_, y_pred=test_pred_)

	test_features_= [test_features[j] for j in test_indexes]
	probability_test_ = knn.predict_proba(test_features_)
	aucroc_score_test = sklearn.metrics.roc_auc_score(test_labels_, probability_test_[:,1])


	mcc_test = sklearn.metrics.matthews_corrcoef(test_labels_, test_pred_)
	cm = sklearn.metrics.confusion_matrix(y_true=test_labels_, y_pred=test_pred_).astype(float)
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
	print('mcc: %f' %mcc_test)
	print('auc: %f' %aucroc_score_test)

print('#'*100)


print('time elapsed: ', time.time() - t)






















