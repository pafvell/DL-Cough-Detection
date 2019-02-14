#Author: Maurice Weber

import json
import time

import h5py
import sklearn
from sklearn.ensemble import RandomForestClassifier

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

# auc roc curve
probability_test = rf.predict_proba(test_features)
probVec = probability_test[:, 1]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_labels, probVec)
df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
df.to_csv("knn_roc_curve_rf.csv")


# predictions
train_pred = rf.predict(train_features)
test_pred = rf.predict(test_features)

## get figures for entire data set
train_accuracy = sklearn.metrics.accuracy_score(y_true=train_labels, y_pred=train_pred)
test_accuracy = sklearn.metrics.accuracy_score(y_true=test_labels, y_pred=test_pred)

aucroc_score_test = sklearn.metrics.roc_auc_score(test_labels, probability_test[:,1])

cm = sklearn.metrics.confusion_matrix(y_true=test_labels, y_pred=test_pred).astype(float)
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[0, 0]
TN = cm[1, 1]

sen = TP / (TP + FN)
spec = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
ACC = (TP + TN) / (TP + FP + FN + TN)
MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

print('#'*100, "\n")

## print hyperparams
print('----------------- HYPERPARAMS -----------------\n')
print('N_TREES = %i\nMAX_FEATURES = %i\nMAX_DEPTH = %i\n'%(N_TREES, MAX_FEATURES, MAX_DEPTH))

## start printing results
print('----------------- device: %s -----------------'%DEVICE_FILTER)
print('RESULTS:')
print('sen: %f' % sen)
print('spec: %f' % spec)
print('PPV: %f' % PPV)
print('NPV: %f' % NPV)
print('ACC: %f' % ACC)
print('MCC: %f' % MCC)
print('auc: %f' % aucroc_score_test)


mother_acc = []
mother_auc = []
mother_spec = []
mother_sen= []
mother_mcc = []
mother_ppv = []
mother_npv = []

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

	test_features_ = [test_features[j] for j in test_indexes]
	probability_test_ = rf.predict_proba(test_features_)
	aucroc_score_test = sklearn.metrics.roc_auc_score(test_labels_, probability_test_[:, 1])

	cm = sklearn.metrics.confusion_matrix(y_true=test_labels_, y_pred=test_pred_).astype(float)
	FP = cm[0, 1]
	FN = cm[1, 0]
	TP = cm[0, 0]
	TN = cm[1, 1]

	sen = TP / (TP + FN)
	spec = TN / (TN + FP)
	PPV = TP / (TP + FP)
	NPV = TN / (TN + FN)
	ACC = (TP + TN) / (TP + FP + FN + TN)
	MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
	
	# print results
	print('RESULTS:')
	print('sen: %f' %sen)
	print('spec: %f' % spec)
	print('PPV: %f' % PPV)
	print('NPV: %f' % NPV)
	print('ACC: %f' % ACC)
	print('MCC: %f' % MCC)
	print('auc: %f' %aucroc_score_test)

	mother_acc.append(ACC)
	mother_sen.append(sen)
	mother_spec.append(spec)
	mother_ppv.append(PPV)
	mother_npv.append(NPV)
	mother_auc.append(aucroc_score_test)
	mother_mcc.append(MCC)

acc_av = np.mean(mother_acc)
acc_sd = np.std(mother_acc)
aucroc_av = np.mean(mother_auc)
aucroc_sd = np.std(mother_auc)
spec_av = np.mean(mother_spec)
spec_sd = np.std(mother_spec)
sens_av = np.mean(mother_sen)
sens_sd = np.std(mother_sen)
mcc_av = np.mean(mother_mcc)
mcc_sd = np.std(mother_mcc)
ppv_av = np.mean(mother_ppv)
ppv_sd = np.std(mother_ppv)
npv_av = np.mean(mother_npv)
npv_sd = np.std(mother_npv)

print('#' * 100, "\n")
print('#' * 100, "\n")
print('SUMMARY RESULTS:')
print('test accuracy: (mean %f, +/- SD %f)' % (acc_av, acc_sd))
print('aucroc score test: (mean %f, +/- SD %f)' % (aucroc_av, aucroc_sd))
print('specificity: (mean %f, +/- SD %f)' % (spec_av, spec_sd))
print('sensitivity: (mean %f, +/- SD %f)' % (sens_av, sens_sd))
print('ppv: (mean %f, +/- SD %f)' % (ppv_av, ppv_sd))
print('npv: (mean %f, +/- SD %f)' % (npv_av, npv_sd))
print('mcc: (mean %f, +/- SD %f)' % (mcc_av, mcc_sd))


print('#'*100)


print('time elapsed: ', time.time() - t)






















