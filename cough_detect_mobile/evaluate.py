#!/usr/bin/python
# Authors: Kevin Kipfer, Filipe Barata

import argparse
import importlib
import json


import pandas as pd



import tqdm



import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, matthews_corrcoef

from utils import *

# ******************************************************************************************************************
_CHECKPOINTS = {
    'htc': './checkpoints/cnnv3l3_mobilehtc',
    'iphone': './checkpoints/cnnv3l3_mobileiphone',
    'samsung': './checkpoints/cnnv3l3_mobilesamsung',
    'studio': './checkpoints/cnnv3l3_mobilestudio',
    'tablet': './checkpoints/cnnv3l3_mobiletablet'
}

tf.set_random_seed(0)

# loading config file
parser = argparse.ArgumentParser(description='Script to evaluate a given Model')
parser.add_argument('--config',
                    type=str,
                    #default='./configs/config_C3L3.json',
                    default='./config.json',
                    help='path to the file config.json which stores all the necessary parameters')
parser.add_argument('--ckpt_dir',
                    type=str,
                    default='./checkpoints/cnnv3l3_mobile',
                    help='path to the file config.json which stores all the necessary parameters')
args = parser.parse_args()

# loading configuration
with open(args.config) as json_data_file:
    config = json.load(json_data_file)

control_config = config["controller"]  # reads the config for the controller file
config_db = config["dataset"]
config_train = control_config["training_parameter"]

if args.ckpt_dir:
    CKPT_DIR = args.ckpt_dir
else:
    CKPT_DIR = config_train["checkpoint_dir"]

if "NFFT" in config_db:
    NFFT = config_db["NFFT"]
else:
    NFFT = 2048


# ******************************************************************************************************************

def plot_roc(fpr, tpr, auc_roc_score):
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label="ROC curve (Area={:.4f})".format(auc_roc_score))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    # save figure
    plot_filename = "roc_curve.png"
    fig.savefig(plot_filename)
    print("saved roc curve as {}".format(plot_filename))


def classification_report(y_true, y_pred, y_probs=None, sanity_check=False, print_report=True,
                          write_aucroccurve_tofile=False):
    cm = confusion_matrix(y_true, y_pred)

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
    auc_roc_score = 0


    if y_probs:
        # compute auc roc score + make plot
        fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_probs)
        auc_roc_score = skmetrics.roc_auc_score(y_true, y_probs)
        if write_aucroccurve_tofile:
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})


            df.to_csv("knn_roc_curve_cnn.csv")
        #print('auc_roc_score: ', auc_roc_score)
        #plot_roc(fpr, tpr, auc_roc_score)
    if print_report:
        print('sen: %f' % sen)
        print('spec: %f' % spec)
        print('PPV: %f' % PPV)
        print('NPV: %f' % NPV)
        print('ACC: %f' % ACC)
        print('MCC: %f' % MCC)
        print('auc: %f' % auc_roc_score)

    if sanity_check:
        print('(SANITY CHECK - our precision: %f vs sklearn precision: %f)' % (
            PPV, precision_score(y_true, y_pred)))
        print(
            '(SANITY CHECK - our sensitivity: %f vs sklearn recall: %f)' % (sen, recall_score(y_true, y_pred)))

    return ACC, sen, spec, PPV, NPV, MCC, auc_roc_score


def test(
        model_name=control_config["model"],
        hop_length=config_db["HOP"],
        bands=config_db["BAND"],
        window=config_db["WINDOW"],
        size_cub=control_config["spec_size"],
        batch_size=config_train["batch_size"],
        num_estimator=config_train["num_estimator"],
        num_filter=config_train["num_filter"],
        split_id=config_db["split_id"],
        participants=config_db["test"],
        sources=config_db["allowedSources"],
        db_root_dir=config_db["DB_ROOT_DIR"],
        checkpoint_dir=CKPT_DIR,
        nfft=NFFT):
    print('read checkpoints: %s' % checkpoint_dir)
    checkpoint_dir = checkpoint_dir + '/cv%d' % split_id

    print('evaluate model:' + model_name)
    model = importlib.import_module(model_name)

    # TODO restore any checkpoint
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest_ckpt:
        raise IOError('Invalid checkpoint path: %s! It is not possible to evaluate the model.' % checkpoint_dir)
    print('restore checkpoint:%s' % latest_ckpt)

    # Create the session that we'll use to execute the model
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    )
    sess = tf.Session(config=sess_config)

    input_tensor = tf.placeholder(tf.float32, shape=[bands, size_cub], name='Input')
    x = tf.expand_dims(input_tensor, 0)
    _, output_tensor, logits = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter,
                                                 is_training=False, include_logits=True)
    class_probabilities = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    saver.restore(sess, latest_ckpt)

    # get data and predict
    X_cough, X_other, _, _ = get_imgs(split_id=config_db["split_id"],
                                      db_root_dir=config_db["DB_ROOT_DIR"],
                                      listOfParticipantsInTestset=config_db["test"],
                                      listOfParticipantsInValidationset=config_db["validation"],
                                      listOfAllowedSources=config_db["allowedSources"]
                                      )

    print('nr of samples coughing (test): %d' % len(X_cough))
    print('nr of samples NOT coughing (test): %d' % len(X_other))

    X = X_cough + X_other
    y = [1] * len(X_cough) + [0] * len(X_other)

    predictions, class_probs_list = [], []
    for x in X:  # make_batches(X, batch_size):
        x = preprocess(x, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
        # x = np.expand_dims(x, 0)
        preds_sample, class_probs_sample = sess.run([output_tensor, class_probabilities], {input_tensor: x})
        predictions.append(preds_sample)
        class_probs_list.append(class_probs_sample[0][1])  # probability of positive class

    print('class probs shape: ', np.shape(class_probs_list))

    print()
    print('********************************************************************************')
    print('Evaluate over Everything:')
    ACC, sen, spec, PPV, NPV, MCC, auc_roc_score = classification_report(y, predictions, y_probs=class_probs_list,
                                                          write_aucroccurve_tofile=True)

    X = list(zip(X, y, predictions))
    sources = ["studio", "iphone", "samsung", "htc", "tablet", "audio track"]
    for mic in sources:
        Xlittle = [x for x in X if mic in get_device(x[0])]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + mic)
            classification_report(y_true, y_pred)

    kinds = ["Close (cc)", "Distant (cd)", "01_Throat Clearing", "02_Laughing", "03_Speaking", "04_Spirometer"]

    for kind in kinds:
        Xlittle = [x for x in X if kind in x[0]]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + kind)
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            print('Confusion Matrix: \n', cm)
            print('accuracy: ', acc)

    return ACC, sen, spec, PPV, NPV, MCC, auc_roc_score


def test_cv(device,
            model_name=control_config["model"],
            hop_length=config_db["HOP"],
            bands=config_db["BAND"],
            window=config_db["WINDOW"],
            size_cub=control_config["spec_size"],
            batch_size=config_train["batch_size"],
            num_estimator=config_train["num_estimator"],
            num_filter=config_train["num_filter"],
            split_id=config_db["split_id"],
            participants=config_db["test"],
            sources=config_db["allowedSources"],
            db_root_dir=config_db["DB_ROOT_DIR"],
            checkpoint_dir=CKPT_DIR,
            nfft=NFFT):
    checkpoint_dir = checkpoint_dir + device
    print('read checkpoints: %s' % checkpoint_dir)
    checkpoint_dir = checkpoint_dir + '/cv%d' % split_id

    print('evaluate model:' + model_name)
    model = importlib.import_module(model_name)

    # TODO restore any checkpoint
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest_ckpt:
        raise IOError('Invalid checkpoint path: %s! It is not possible to evaluate the model.' % checkpoint_dir)
    print('restore checkpoint:%s' % latest_ckpt)

    # Create the session that we'll use to execute the model
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    )
    sess = tf.Session(config=sess_config)

    input_tensor = tf.placeholder(tf.float32, shape=[bands, size_cub], name='Input')
    x = tf.expand_dims(input_tensor, 0)
    _, output_tensor, logits = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter,
                                                 is_training=False, include_logits=True)
    class_probabilities = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    saver.restore(sess, latest_ckpt)

    # get data and predict
    X_cough, X_other, _, _ = get_imgs(split_id=config_db["split_id"],
                                      db_root_dir=config_db["DB_ROOT_DIR"],
                                      listOfParticipantsInTestset=config_db["test"],
                                      listOfParticipantsInValidationset=config_db["validation"],
                                      listOfAllowedSources=config_db["allowedSources"],
                                      device_cv=True,
                                      device=device
                                      )

    print('nr of samples coughing (test): %d' % len(X_cough))
    print('nr of samples NOT coughing (test): %d' % len(X_other))

    X = X_cough + X_other
    y = [1] * len(X_cough) + [0] * len(X_other)

    predictions, class_probs_list = [], []
    for x in X:  # make_batches(X, batch_size):
        x = preprocess(x, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
        # x = np.expand_dims(x, 0)
        preds_sample, class_probs_sample = sess.run([output_tensor, class_probabilities], {input_tensor: x})
        predictions.append(preds_sample)
        class_probs_list.append(class_probs_sample[0][1])  # probability of positive class

    print('class probs shape: ', np.shape(class_probs_list))

    print()
    print('********************************************************************************')
    print('Evaluate over Everything:')
    ACC, sen, spec, PPV, NPV, MCC, auc_roc_score = classification_report(y, predictions, y_probs=class_probs_list,
                                                           write_aucroccurve_tofile=False)

    kinds = ["Close (cc)", "Distant (cd)", "01_Throat Clearing", "02_Laughing", "03_Speaking", "04_Spirometer"]

    for kind in kinds:
        Xlittle = [x for x in X if kind in x[0]]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + kind)
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            print('Confusion Matrix: \n', cm)
            print('accuracy: ', acc)

    return ACC, sen, spec, PPV, NPV, MCC, auc_roc_score


def test_ensemble(model_name=control_config["model"],
                  hop_length=config_db["HOP"],
                  bands=config_db["BAND"],
                  window=config_db["WINDOW"],
                  size_cub=control_config["spec_size"],
                  batch_size=config_train["batch_size"],
                  num_estimator=config_train["num_estimator"],
                  num_filter=config_train["num_filter"],
                  split_id=config_db["split_id"],
                  participants=config_db["test"],
                  sources=config_db["allowedSources"],
                  db_root_dir=config_db["DB_ROOT_DIR"],
                  checkpoint_dirs=_CHECKPOINTS,
                  nfft=NFFT):

    print("***************************** TESTING ENSEMBLE OF 5 CNNs *****************************")

    checkpoints = {}
    for mic, chckpt in checkpoint_dirs.items():
        current_dir = checkpoint_dirs[mic] + '/cv%d' % split_id
        checkpoints[mic] = tf.train.latest_checkpoint(current_dir)
        if not checkpoints[mic]:
            raise IOError(
                'Invalid checkpoint path: %s! It is not possible to evaluate the model with checkpount' % current_dir)
        print('restore checkpoint %s' % checkpoints[mic])

    print('evaluate model:' + model_name)
    model = importlib.import_module(model_name)

    # get data
    X_cough, X_other, _, _ = get_imgs(split_id=config_db["split_id"],
                                      db_root_dir=config_db["DB_ROOT_DIR"],
                                      listOfParticipantsInTestset=config_db["test"],
                                      listOfParticipantsInValidationset=config_db["validation"],
                                      listOfAllowedSources=config_db["allowedSources"])

    print('nr of samples coughing (test): %d' % len(X_cough))
    print('nr of samples NOT coughing (test): %d' % len(X_other))

    X = X_cough + X_other
    y = [1] * len(X_cough) + [0] * len(X_other)

    # preprocess

    # build graph
    input_tensor = tf.placeholder(tf.float32, shape=[bands, size_cub], name='Input')
    x = tf.expand_dims(input_tensor, 0)
    _, output_tensor, logits = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter,
                                                 is_training=False, include_logits=True)
    class_probabilities = tf.squeeze(tf.nn.softmax(logits))
    print(class_probabilities)

    saver = tf.train.Saver()

    # create list to store predictions of each model in the ensemble
    predictions_per_model = list()

    # Create the session that we'll use to execute the model
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    )

    for mic, chkpt in checkpoints.items():
        print("evaluating {}...".format(mic))
        with tf.Session(config=sess_config) as sess:
            saver.restore(sess, chkpt)

            class_probabilities_eval = []
            for x in tqdm.tqdm(X):  # make_batches(X, batch_size):
                x = preprocess(x, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
                # x = np.expand_dims(x, 0)
                class_probs_sample = sess.run(class_probabilities, {input_tensor: x})
                class_probabilities_eval.append(class_probs_sample)

        predictions_per_model.append(class_probabilities_eval)

    # store as numpy
    predictions_per_model = np.array(predictions_per_model)

    # average voting
    probs_avg_vote = np.mean(predictions_per_model, axis=0)
    probs_avg_vote_cough = list(probs_avg_vote[:, 1])
    preds_avg_vote = np.argmax(probs_avg_vote, axis=1)

    print()
    print('********************************************************************************')
    print('Evaluate over Everything:')
    ACC, sen, spec, PPV, NPV, MCC, auc_roc_score = classification_report(y, preds_avg_vote, y_probs=probs_avg_vote_cough,
                                                          write_aucroccurve_tofile=True)

    X = list(zip(X, y, preds_avg_vote))
    sources = ["studio", "iphone", "samsung", "htc", "tablet", "audio track"]
    for mic in sources:
        Xlittle = [x for x in X if mic in get_device(x[0])]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + mic)
            classification_report(y_true, y_pred)

    kinds = ["Close (cc)", "Distant (cd)", "01_Throat Clearing", "02_Laughing", "03_Speaking", "04_Spirometer"]

    for kind in kinds:
        Xlittle = [x for x in X if kind in x[0]]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + kind)
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            print('Confusion Matrix: \n', cm)
            print('accuracy: ', acc)

    return ACC, sen, spec, PPV, NPV, MCC, auc_roc_score


def test_ensemble_cv(device,
                     model_name=control_config["model"],
                  hop_length=config_db["HOP"],
                  bands=config_db["BAND"],
                  window=config_db["WINDOW"],
                  size_cub=control_config["spec_size"],
                  batch_size=config_train["batch_size"],
                  num_estimator=config_train["num_estimator"],
                  num_filter=config_train["num_filter"],
                  split_id=config_db["split_id"],
                  participants=config_db["test"],
                  sources=config_db["allowedSources"],
                  db_root_dir=config_db["DB_ROOT_DIR"],
                  checkpoint_dirs=_CHECKPOINTS,
                  nfft=NFFT):

    print("***************************** TESTING ENSEMBLE OF 4 CNNs *****************************")

    checkpoints = {}
    for mic, chckpt in checkpoint_dirs.items():
        current_dir = checkpoint_dirs[mic] + '/cv%d' % split_id
        checkpoints[mic] = tf.train.latest_checkpoint(current_dir)
        if not checkpoints[mic]:
            raise IOError(
                'Invalid checkpoint path: %s! It is not possible to evaluate the model with checkpount' % current_dir)
        print('restore checkpoint %s' % checkpoints[mic])

    print('evaluate model:' + model_name)
    model = importlib.import_module(model_name)

    # get data
    X_cough, X_other, _, _ = get_imgs(split_id=config_db["split_id"],
                                      db_root_dir=config_db["DB_ROOT_DIR"],
                                      listOfParticipantsInTestset=config_db["test"],
                                      listOfParticipantsInValidationset=config_db["validation"],
                                      listOfAllowedSources=config_db["allowedSources"],
                                      device= device)

    print('nr of samples coughing (test): %d' % len(X_cough))
    print('nr of samples NOT coughing (test): %d' % len(X_other))

    X = X_cough + X_other
    y = [1] * len(X_cough) + [0] * len(X_other)

    # preprocess

    # build graph
    input_tensor = tf.placeholder(tf.float32, shape=[bands, size_cub], name='Input')
    x = tf.expand_dims(input_tensor, 0)
    _, output_tensor, logits = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter,
                                                 is_training=False, include_logits=True)
    class_probabilities = tf.squeeze(tf.nn.softmax(logits))
    print(class_probabilities)

    saver = tf.train.Saver()

    # create list to store predictions of each model in the ensemble
    predictions_per_model = list()

    # Create the session that we'll use to execute the model
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    )

    for mic, chkpt in checkpoints.items():
        print("evaluating {}...".format(mic))
        with tf.Session(config=sess_config) as sess:
            saver.restore(sess, chkpt)

            class_probabilities_eval = []
            for x in tqdm.tqdm(X):  # make_batches(X, batch_size):
                x = preprocess(x, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
                # x = np.expand_dims(x, 0)
                class_probs_sample = sess.run(class_probabilities, {input_tensor: x})
                class_probabilities_eval.append(class_probs_sample)

        predictions_per_model.append(class_probabilities_eval)

    # store as numpy
    predictions_per_model = np.array(predictions_per_model)

    # average voting
    probs_avg_vote = np.mean(predictions_per_model, axis=0)
    probs_avg_vote_cough = list(probs_avg_vote[:, 1])
    preds_avg_vote = np.argmax(probs_avg_vote, axis=1)

    print()
    print('********************************************************************************')
    print('Evaluate over Everything:')
    ACC, sen, spec, PPV, NPV, MCC, auc_roc_score = classification_report(y, preds_avg_vote, y_probs=probs_avg_vote_cough,
                                                          write_aucroccurve_tofile=True)

    X = list(zip(X, y, preds_avg_vote))
    sources = ["studio", "iphone", "samsung", "htc", "tablet", "audio track"]
    for mic in sources:
        Xlittle = [x for x in X if mic in get_device(x[0])]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + mic)
            classification_report(y_true, y_pred)

    kinds = ["Close (cc)", "Distant (cd)", "01_Throat Clearing", "02_Laughing", "03_Speaking", "04_Spirometer"]

    for kind in kinds:
        Xlittle = [x for x in X if kind in x[0]]
        if len(Xlittle) > 0:
            path, y_true, y_pred = zip(*Xlittle)
            print()
            print('********************************************************************************')
            print('Evaluate ' + kind)
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            print('Confusion Matrix: \n', cm)
            print('accuracy: ', acc)

    return ACC, sen, spec, PPV, NPV, MCC, auc_roc_score



if __name__ == '__main__':
    if  config["DEVICE_CV"]:


        device = "htc"
        ACC, sen, spec, PPV, NPV, MCC, auc_roc_score = test_cv(device)


        # print('#' * 100)

    elif config["DEVICE_CV_EXP2"]:

        # device = "samsung"
        # print("*************************" +device+"******************************")
        # checkpoints = {
        #     'htc': './checkpoints/cnnv3l3_mobilesamsunghtc',
        #     'iphone': './checkpoints/cnnv3l3_mobileiphonesamsung',
        #     'tablet': './checkpoints/cnnv3l3_mobilesamsungtablet',
        #     'studio': './checkpoints/cnnv3l3_mobilestudiosamsung'
        # }

        device = "iphone"
        print("*************************" +device+"******************************")
        checkpoints = {
            'htc': './checkpoints/cnnv3l3_mobileiphonehtc',
            'samsung': './checkpoints/cnnv3l3_mobileiphonesamsung',
            'tablet': './checkpoints/cnnv3l3_mobileiphonetablet',
            'studio': './checkpoints/cnnv3l3_mobilestudioiphone'
        }

        # device = "tablet"
        # print("*************************" +device+"******************************")
        # checkpoints = {
        #     'htc': './checkpoints/cnnv3l3_mobiletablethtc',
        #     'samsung': './checkpoints/cnnv3l3_mobilesamsungtablet',
        #     'iphone': './checkpoints/cnnv3l3_mobileiphonetablet',
        #     'studio': './checkpoints/cnnv3l3_mobilestudiotablet'
        # }

        # device = "studio"
        # print("*************************" +device+"******************************")
        # checkpoints = {
        #     'htc': './checkpoints/cnnv3l3_mobilestudiohtc',
        #     'samsung': './checkpoints/cnnv3l3_mobilestudiosamsung',
        #     'iphone': './checkpoints/cnnv3l3_mobilestudioiphone',
        #     'tablet': './checkpoints/cnnv3l3_mobilestudiotablet'
        # }

        # device = "htc"
        # print("*************************" +device+"******************************")
        # checkpoints = {
        #     'studio': './checkpoints/cnnv3l3_mobilestudiohtc',
        #     'samsung': './checkpoints/cnnv3l3_mobilesamsunghtc',
        #     'iphone': './checkpoints/cnnv3l3_mobileiphonehtc',
        #     'tablet': './checkpoints/cnnv3l3_mobiletablethtc'
        # }

        test_ensemble_cv(device, checkpoint_dirs=checkpoints)

    else:
        #test()
        test_ensemble()

