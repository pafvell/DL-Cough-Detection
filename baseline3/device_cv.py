# Author: Filipe Barata

import json
import time

import h5py
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from utils import *

# loading configuration
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
DB_FILENAME = '/data_%s.h5' % (DB_VERSION)

print('#' * 100, "\n")

print('Data used from %s' % (DB_ROOT_DIR + DB_FILENAME))

# get data
with h5py.File(DB_ROOT_DIR + DB_FILENAME, 'r') as hf:
    train_devices = hf['train_devices'][:].astype('U13').tolist()
    train_labels = hf['train_data'][:, 0]
    train_features = hf['train_data'][:, 1:]
    print("shape of train features: {train_shape}".format(train_shape=np.shape(train_features)))

    test_devices = hf['test_devices'][:].astype('U13').tolist()
    test_labels = hf['test_data'][:, 0]
    test_features = hf['test_data'][:, 1:]
    print("shape of test features: {test_shape}".format(test_shape=np.shape(test_features)))


mother_test_accuracy = []
mother_aucroc_score_test = []
mother_specificity = []
mother_sensitivity = []

for device in DEVICE_FILTER:

    if device == "audio track":
        continue

    id_not_device_train = np.array(train_devices) != device
    tmp_train_labels = train_labels[id_not_device_train]
    tmp_train_features = train_features[id_not_device_train]

    id_not_device_test = np.array(test_devices) == device
    tmp_test_labels = test_labels[id_not_device_test]
    tmp_test_features = test_features[id_not_device_test]



## fit random forest
    knn = KNeighborsClassifier(n_neighbors=NNEIGHBOURS, weights=WEIGHT_TYPE, p=WEIGHT_METRIC)

    knn.fit(tmp_train_features, tmp_train_labels)

    # predictions
    train_pred = knn.predict(tmp_train_features)
    test_pred = knn.predict(tmp_test_features)

    ## get figures for entire data set
    train_accuracy = sklearn.metrics.accuracy_score(y_true=tmp_train_labels, y_pred=train_pred)
    test_accuracy = sklearn.metrics.accuracy_score(y_true=tmp_test_labels, y_pred=test_pred)
    aucroc_score_train = sklearn.metrics.roc_auc_score(tmp_train_labels, train_pred)
    aucroc_score_test = sklearn.metrics.roc_auc_score(tmp_test_labels, test_pred)
    cm = sklearn.metrics.confusion_matrix(y_true=tmp_test_labels, y_pred=test_pred).astype(float)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])

    print('#' * 100, "\n")

    ## print hyperparams
    print('----------------- HYPERPARAMS -----------------\n')
    print('NNEIGHBOURS = %i\nWEIGHT_METRIC = %s\nWEIGHT_TYPE = %i\n' % (NNEIGHBOURS, WEIGHT_TYPE, WEIGHT_METRIC))

    DEVICE_FILTER_array = np.array(DEVICE_FILTER)

    ## start printing results
    print('----------------- devices trained: %s -----------------' % DEVICE_FILTER_array[DEVICE_FILTER_array != device])
    print('----------------- device tested: %s -----------------' % device)
    print('RESULTS:')
    print('test accuracy: %f' % test_accuracy)
    print('train accuracy: %f' % train_accuracy)
    print('sensitivity: %f' % sensitivity)
    print('specificity: %f' % specificity)
    print('precision: %f' % precision)

    mother_aucroc_score_test.append(aucroc_score_test)
    mother_test_accuracy.append(test_accuracy)
    mother_specificity.append(specificity)
    mother_sensitivity.append(sensitivity)

acc_av = np.mean(mother_test_accuracy)
acc_sd = np.std(mother_test_accuracy)
aucroc_av = np.mean(mother_aucroc_score_test)
aucroc_sd = np.std(mother_aucroc_score_test)
spec_av = np.mean(mother_specificity)
spec_sd = np.std(mother_specificity)
sens_av = np.mean(mother_sensitivity)
sens_sd = np.std(mother_sensitivity)

print('#' * 100, "\n")
print('#' * 100, "\n")
print('SUMMARY RESULTS:')
print('test accuracy: (mean %f, +/- SD %f)' % (acc_av, acc_sd))
print('aucroc score test: (mean %f, +/- SD %f)' % (aucroc_av, aucroc_sd))
print('sensitivity: (mean %f, +/- SD %f)' % (spec_av, spec_sd))
print('specificity: (mean %f, +/- SD %f)' % (sens_av, sens_sd))

print('#' * 100)

print('time elapsed: ', time.time() - t)


