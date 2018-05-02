#Authors: Maurice Weber

import numpy as np
import pandas as pd
import h5py
import json
from itertools import chain
from tqdm import tqdm

import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import signal as SCIPY_SIGNAL

from utils import *


#loading configuration
with open('config.json') as json_data_file:
	config = json.load(json_data_file)

# read params from config file
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]
PCA_COMPONENTS = config["create_db"]["PCA_COMPONENTS"]
HOP = config["create_db"]["HOP"]
N_FFT = config["create_db"]["N_FFT"]
WINDOW_SIZE = config["create_db"]["WINDOW_SIZE"]
DEVICE_FILTER = config["create_db"]["DEVICE_FILTER"]
BATCH_SIZE = config["create_db"]["BATCH_SIZE"]

DB_FILENAME = '/data_DEVICES=[%s]_BATCHSIZE=%i_PCA=%i_NFFT=%i_HOP=%i_WINDOW=%f.h5'%\
				("-".join(DEVICE_FILTER), BATCH_SIZE, PCA_COMPONENTS, N_FFT, HOP, WINDOW_SIZE)

print('Data will be stored to %s'%(DB_ROOT_DIR + DB_FILENAME))


def stack_spectrograms(filenames):
	'''
	:param filenames: list containing the names of the files to be processed
	:return:	A matrix X with vectorized spectrograms as rows
				A list labels with entries 1 (cough sound) and 0 (other sound)
	'''

	if len(filenames) < 1:
		raise ValueError("list of filenames is empty")

	X = None
	labels = []
	device_classes = []
	count = 0

	for fn in filenames:

		# get signal of importance
		sig, sample_rate = extract_Signal_Of_Importance(f = fn, window=WINDOW_SIZE)
		if np.size(sig) / sample_rate != WINDOW_SIZE:
			continue
		
		# win_length=int(sample_rate*0.016)
		stft = np.abs(librosa.stft(y = sig, n_fft=N_FFT, window=SCIPY_SIGNAL.hamming, hop_length=HOP))
		stft_vec = np.reshape(stft, (np.size(stft),), 'F')
		stft_vec /= np.linalg.norm(stft_vec)
		label = int("Coughing" in fn)
		device = get_device(fn)

		if count == 0:
			X = stft_vec
			device_classes = [device]
			labels = [label] 
		else:
			X = np.vstack([X, stft_vec])
			device_classes.append(device)
			labels.append(label)
		
		count += 1

	return device_classes, labels, X, np.shape(stft)




def generate_cough_model(file_list, batch_size=BATCH_SIZE):

	idx = 0
	batch_num = 1

	assert batch_size % 2 == 0

	# for idx in tqdm(range(0, len(file_list) - batch_size + 1, batch_size)):
	for idx in tqdm(range(0, len(file_list), batch_size)):
	# while idx <= (len(file_list) - batch_size):

		batch = file_list[idx:(idx+batch_size)]
		device_classes, labels, features, stft_shape = stack_spectrograms(batch)

		## compute model
		# PCA on vectorized spectrograms
		features = sklearn.preprocessing.scale(features, axis=0, with_mean=True, with_std=False)
		pca = decomposition.PCA()
		pca.n_components = PCA_COMPONENTS
		features_projected = pca.fit_transform(features)

		# Decompress data with inverse PCA transform
		features_reduced = pca.inverse_transform(features_projected)
		residual_error = np.mean((features - features_reduced) ** 2, axis = 1)

		# reconstruct spectrogram to compute energy features for training set
		for i in range(np.shape(features_reduced)[0]):

			stft_reduced = np.reshape(features_reduced[i,:], (stft_shape[0], stft_shape[1]))
			
			# compute energy
			energy = np.mean(librosa.feature.rmse(S=stft_reduced))
			energy_low_freq = np.mean(librosa.feature.rmse(S=stft_reduced[:(stft_shape[0] // 2),]))
			energy_high_freq = np.mean(librosa.feature.rmse(S=stft_reduced[(stft_shape[0] // 2):,]))
			energy_features_ = np.hstack([energy, energy_low_freq, energy_high_freq])
			if i == 0:
				energy_features = energy_features_
			else:
				energy_features = np.vstack([energy_features, energy_features_])

		# merge features into single data matrix
		cough_model_ = np.column_stack((features_projected, energy_features, residual_error))

		if batch_num == 1:
			cough_model = cough_model_
			all_labels = labels
			all_devices = device_classes
		else:
			cough_model = np.vstack([cough_model, cough_model_])
			all_labels.extend(labels)
			all_devices.extend(device_classes)

		batch_num += 1

	print("shape of labels: {label_shape}, shape of cough model: {cough_shape}"\
		.format(label_shape=np.shape(all_labels), cough_shape=np.shape(cough_model)))

	return all_devices, all_labels, cough_model



def main():

	list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', \
							'04_Coughing/Distant (cd)/p17_tablet-108.wav', \
							'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

	print ('use data from root path %s'%DB_ROOT_DIR)
	coughAll = find_files(DB_ROOT_DIR + "/04_Coughing", "wav", recursively=True)
	assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'
	
	coughAll = remove_broken_files(DB_ROOT_DIR, list_of_broken_files, coughAll)
	other = find_files(DB_ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)
	trainListCough = list(coughAll)
	trainListOther = list(other)

	# test participants
	listOfParticipantsInTestset = config["data_split"]["test"]
	testListOther, testListCough = [], []

	#split files into test- and training-set
	for name in coughAll:
		if get_device(name) in DEVICE_FILTER:
			for nameToExclude in listOfParticipantsInTestset:
				if nameToExclude in name:
					testListCough.append(name)
					trainListCough.remove(name)
		else:
			trainListCough.remove(name)

	for name in other:
		if get_device(name) in DEVICE_FILTER:
			for nameToExclude in listOfParticipantsInTestset:
				if nameToExclude in name:
					testListOther.append(name)
					trainListOther.remove(name)
		else:
			trainListOther.remove(name)

	print()
	print('------------------------------------------------------------------')
	print('COMPOSITION OF DATASET')
	print('nr of samples coughing (test): %d' % len(testListCough))
	print('nr of samples NOT coughing (test): %d' % len(testListOther))
	print('nr of samples coughing (train): %d' % len(trainListCough))
	print('nr of samples NOT coughing (train): %d' % len(trainListOther))
	t1 = len(testListCough) + len(testListOther)
	t2 = len(trainListCough) + len(trainListOther)
	print('total nr of samples: (train) %d + (test) %d = (total) %d' % (t2, t1, t1+t2))
	print()


	print("#"*10, "processing training data", "#"*10)
	train_list = trainListCough
	train_list.extend(trainListOther)
	np.random.shuffle(train_list)
	print("number of train samples:", len(train_list))
	train_devices, train_labels, train_features = generate_cough_model(train_list)

	print("#"*10, "processing test data", "#"*10)
	test_list = testListCough
	test_list.extend(testListOther)
	np.random.shuffle(test_list)
	print("number of test samples:", len(test_list))
	test_devices, test_labels, test_features = generate_cough_model(test_list)


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
	main()






























