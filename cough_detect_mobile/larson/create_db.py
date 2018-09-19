# Authors: Maurice Weber

import time
import h5py
import json
from tqdm import tqdm

from sklearn import decomposition
from scipy import signal as scipy_signal
from scipy.spatial.distance import cdist

from utils import *

# loading configuration
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

# read params from config file
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]
DB_VERSION = config["general"]["DB_VERSION"]
PCA_COMPONENTS = config["create_db"]["PCA_COMPONENTS"]
HOP = config["create_db"]["HOP"]
N_FFT = config["create_db"]["N_FFT"]
WINDOW_SIZE = config["create_db"]["WINDOW_SIZE"]
DEVICE_FILTER = config["create_db"]["DEVICE_FILTER"]
BATCH_SIZE = config["create_db"]["BATCH_SIZE"]

DB_FILENAME = '/data_%s.h5' % DB_VERSION

RECONSTRUCT = True

print('Data will be stored to %s' % (DB_ROOT_DIR + DB_FILENAME))


def stack_spectrograms(filenames):
    """
    :param filenames: list containing the names of the files to be processed
    :return:	A matrix X with vectorized spectrograms as rows
                A list labels with entries 1 (cough sound) and 0 (other sound)
    """

    if len(filenames) < 1:
        raise ValueError("list of filenames is empty")

    x = None
    stft = None
    labels = []
    device_classes = []
    count = 0

    for fn in filenames:

        # get signal of importance
        sig, sample_rate = extract_signal_of_importance(f=fn, window=WINDOW_SIZE)
        stft = np.abs(librosa.stft(y=sig, n_fft=N_FFT, window=scipy_signal.hamming, hop_length=HOP))
        stft_vec = np.reshape(stft, (np.size(stft),), 'F')
        stft_vec /= np.linalg.norm(stft_vec)
        label = int("Coughing" in fn)
        device = get_device(fn)

        if count == 0:
            x = stft_vec
            device_classes = [device]
            labels = [label]
        else:
            x = np.vstack([x, stft_vec])
            device_classes.append(device)
            labels.append(label)

        count += 1

    return device_classes, labels, x, np.shape(stft)


def generate_cough_model(file_list, batch_size=BATCH_SIZE, pca=None, training=True):
    # compute first batch
    device_classes, labels, features, stft_shape = stack_spectrograms(file_list[:batch_size])

    # compute spectrograms + get labels
    for idx in tqdm(range(batch_size, len(file_list), batch_size)):
        batch = file_list[idx:(idx + batch_size)]
        device_classes_, labels_, features_, _ = stack_spectrograms(batch)

        features = np.vstack([features, features_])
        device_classes.extend(device_classes_)
        labels.extend(labels_)

    # compute model
    # PCA on vectorized spectrograms
    features_mean = features.mean(1).reshape(np.shape(features)[0], 1)
    if training:
        pca = decomposition.PCA()
        pca.n_components = PCA_COMPONENTS
        features_projected = pca.fit_transform(features - features_mean) + features_mean

    else:
        features_projected = pca.transform(features - features_mean) + features_mean

    # Decompress data with inverse PCA transform
    features_reduced = pca.inverse_transform(features_projected)
    residual_error = np.diagonal(cdist(features, features_reduced, 'euclidean'))

    # reconstruct spectrogram to compute energy features for training set
    energy_features = np.array([]).reshape(0, 3)
    for i in range(np.shape(features_reduced)[0]):
        # compute mean decibel energy for entire reconstructed spectrogram plus for frequencies above and below the
        # mean frequency
        stft_reduced = np.reshape(features_reduced[i, :], (stft_shape[0], stft_shape[1]))
        stft_reduced_db = librosa.power_to_db(S=stft_reduced ** 2)  # convert magnitude spectrogram to decibel units
        energy = np.mean(librosa.feature.rmse(S=stft_reduced_db))
        energy_low_freq = np.mean(librosa.feature.rmse(S=stft_reduced_db[:(stft_shape[0] // 2), ]))
        energy_high_freq = np.mean(librosa.feature.rmse(S=stft_reduced_db[(stft_shape[0] // 2):, ]))

        energy_features_ = np.hstack([energy, energy_low_freq, energy_high_freq])

        energy_features = np.vstack([energy_features, energy_features_])

    # merge features into single data matrix
    cough_model = np.column_stack((features_projected, energy_features, residual_error))

    print("shape of labels: {label_shape}, shape of cough model: {cough_shape}"
          .format(label_shape=np.shape(labels), cough_shape=np.shape(cough_model)))

    return device_classes, labels, cough_model, pca


def main():
    list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav',
                            '04_Coughing/Distant (cd)/p17_tablet-108.wav',
                            '04_Coughing/Distant (cd)/p17_iphone-108.wav',
                            '04_Coughing/Distant (cd)/p17_samsung-108.wav']

    print('use data from root path %s' % DB_ROOT_DIR)
    cough_all = find_files(DB_ROOT_DIR + "/04_Coughing", "wav", recursively=True)
    assert len(cough_all) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'

    cough_all = remove_broken_files(DB_ROOT_DIR, list_of_broken_files, cough_all)
    other = find_files(DB_ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)
    trainlist_cough = list(cough_all)
    trainlist_other = list(other)

    # test participants
    test_participants = config["data_split"]["test"]
    testlist_other, testlist_cough = [], []

    # split files into test- and training-set
    for name in cough_all:
        if get_device(name) in DEVICE_FILTER:
            for nameToExclude in test_participants:
                if nameToExclude in name:
                    testlist_cough.append(name)
                    trainlist_cough.remove(name)
        else:
            trainlist_cough.remove(name)

    for name in other:
        if get_device(name) in DEVICE_FILTER:
            for nameToExclude in test_participants:
                if nameToExclude in name:
                    testlist_other.append(name)
                    trainlist_other.remove(name)
        else:
            trainlist_other.remove(name)

    print()
    print('------------------------------------------------------------------')
    print('COMPOSITION OF DATASET')
    print('nr of samples coughing (test): %d' % len(testlist_cough))
    print('nr of samples NOT coughing (test): %d' % len(testlist_other))
    print('nr of samples coughing (train): %d' % len(trainlist_cough))
    print('nr of samples NOT coughing (train): %d' % len(trainlist_other))
    t1 = len(testlist_cough) + len(testlist_other)
    t2 = len(trainlist_cough) + len(trainlist_other)
    print('total nr of samples: (train) %d + (test) %d = (total) %d' % (t2, t1, t1 + t2))
    print()

    print("#" * 10, "processing training data", "#" * 10)
    train_list = trainlist_cough
    train_list.extend(trainlist_other)
    np.random.shuffle(train_list)
    print("number of train samples:", len(train_list))
    train_devices, train_labels, train_features, pca = generate_cough_model(train_list)

    print("#" * 10, "processing test data", "#" * 10)
    test_list = testlist_cough
    test_list.extend(testlist_other)
    np.random.shuffle(test_list)
    print("number of test samples:", len(test_list))
    test_devices, test_labels, test_features, _ = generate_cough_model(test_list, pca=pca, training=False)

    # store everything
    with h5py.File(DB_ROOT_DIR + DB_FILENAME, 'w') as hf:

        hf.create_dataset("train_devices", data=np.string_(train_devices))
        hf.create_dataset("train_data", data=np.hstack((
            np.asmatrix(train_labels).T,
            train_features)))

        hf.create_dataset("test_devices", data=np.string_(test_devices))
        hf.create_dataset("test_data", data=np.hstack((
            np.asmatrix(test_labels).T,
            test_features)))


if __name__ == "__main__":
    t = time.time()
    main()
    print('time elapsed: ', time.time() - t)
