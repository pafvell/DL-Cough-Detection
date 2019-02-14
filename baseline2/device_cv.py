#Author: Filipe Barata

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





mother_acc = []
mother_auc = []
mother_spec = []
mother_sen= []
mother_mcc = []
mother_ppv = []
mother_npv = []

for device in DEVICE_FILTER:

    if device == "audio track":
        continue

    # DB File
    DB_FILENAME = '/data_%s%s.h5' % (DB_VERSION, device)

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

    id_not_device_train = np.array(train_devices) != device
    tmp_train_labels = train_labels[id_not_device_train]
    tmp_train_features = train_features[id_not_device_train]

    id_not_device_test = np.array(test_devices) == device
    tmp_test_labels = test_labels[id_not_device_test]
    tmp_test_features = test_features[id_not_device_test]



    ## fit random forest
    rf = RandomForestClassifier(n_estimators = N_TREES,
                                max_features=MAX_FEATURES,
                                max_depth=MAX_DEPTH,
                                random_state=0)
    rf.fit(train_features, train_labels)

    # predictions
    train_pred = rf.predict(tmp_train_features)
    test_pred = rf.predict(tmp_test_features)

    ## get figures for entire data set
    train_accuracy = sklearn.metrics.accuracy_score(y_true=tmp_train_labels, y_pred=train_pred)
    test_accuracy = sklearn.metrics.accuracy_score(y_true=tmp_test_labels, y_pred=test_pred)

    probability_test_ = rf.predict_proba(tmp_test_features)
    aucroc_score_test = sklearn.metrics.roc_auc_score(tmp_test_labels, probability_test_[:, 1])


    cm = sklearn.metrics.confusion_matrix(y_true=tmp_test_labels, y_pred=test_pred).astype(float)
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

    print('#' * 100, "\n")

    ## print hyperparams
    print('----------------- HYPERPARAMS -----------------\n')
    print('N_TREES = %i\nMAX_FEATURES = %i\nMAX_DEPTH = %i\n'%(N_TREES, MAX_FEATURES, MAX_DEPTH))

    DEVICE_FILTER_array = np.array(DEVICE_FILTER)

    ## start printing results
    print('----------------- devices trained: %s -----------------' % DEVICE_FILTER_array[DEVICE_FILTER_array != device])
    print('----------------- device tested: %s -----------------' % device)
    print('RESULTS:')
    print('sen: %f' % sen)
    print('spec: %f' % spec)
    print('PPV: %f' % PPV)
    print('NPV: %f' % NPV)
    print('ACC: %f' % ACC)
    print('MCC: %f' % MCC)
    print('auc: %f' % aucroc_score_test)

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

print('#' * 100)

print('time elapsed: ', time.time() - t)


