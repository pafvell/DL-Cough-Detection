import glob
import random
import os, fnmatch, sys

import numpy as np
import pandas as pd
import librosa

ROOT_DIR = '../../Audio_Data'


def find_files(root, fntype, recursively=False):
	fntype = '*.'+fntype

	if not recursively:
		return glob.glob(os.path.join(root, fntype))

	matches = []
	for dirname, subdirnames, filenames in os.walk(root):
		for filename in fnmatch.filter(filenames, fntype):
			matches.append(os.path.join(dirname, filename))
	
	return matches


def split_train_test_list(n_samples=-1):


	#participants used in the test-set
	listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] 
	list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
							'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

	##
	# READING COUGH DATA
	#
	#

	print ('use data from root path %s'%ROOT_DIR)

	coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
	assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'


	#remove broken files
	for broken_file in list_of_broken_files:
		broken_file = os.path.join(ROOT_DIR, broken_file)
		if broken_file in coughAll:
			print ( 'file ignored: %s'%broken_file )
			coughAll.remove(broken_file)


	#split cough files into test- and training-set
	testListCough = []
	trainListCough = coughAll
	for name in coughAll:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListCough.append(name)
				trainListCough.remove(name)

	print('nr of test samples coughing: %d' % len(testListCough))

	##
	# READING OTHER DATA
	#
	#

	other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

	testListOther = []
	trainListOther = other
	for name in other:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListOther.append(name)
				trainListOther.remove(name)

	print('nr of test samples NOT coughing: %d' % len(testListOther))

	if n_samples > 0:
		trainListCough = trainListCough[0:n_samples]
		trainListOther = trainListOther[0:n_samples]
		testListCough = testListCough[0:(n_samples // 5)]
		testListOther = testListOther[0:(n_samples // 5)]

	return trainListCough, trainListOther, testListCough, testListOther



def extract_Signal_Of_Importance(file_name):

	try:
		X, sample_rate = librosa.load(file_name)
	except Exception as e:
		print("An error occurred while parsing file ", file_name, " :\n")
		print(e)
		return [], -1

	maxValue = np.max(np.abs(X))
	absX = np.abs(X)
	indMax = absX.tolist().index(maxValue)
	numberOfSamples = np.ceil(sample_rate * 0.160)  # averge time of half a cough
	startInd = int(np.max(indMax - numberOfSamples, 0))
	maxLeng = np.size(X)

	if startInd + 2*numberOfSamples > maxLeng - 1:
		endInd = int(maxLeng - 1)
		startInd = int(endInd - 2 * numberOfSamples)
	else:
		endInd = int(startInd + 2*numberOfSamples)

	signal = X[startInd:endInd]
	return signal, sample_rate


def extract_feature(file_name):

	signal, sample_rate = extract_Signal_Of_Importance(file_name)
	stft = np.abs(librosa.stft(signal))
	mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(signal, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal),sr=sample_rate).T,axis=0)
	return mfccs,chroma,mel,contrast,tonnetz


def compute_features_matrix(fileNames):

	Xmfccs = np.empty((0,40))
	Xchroma = np.empty((0,12))
	Xmel = np.empty((0,128))
	Xcontrast = np.empty((0,7))
	Xtonnetz = np.empty((0,6))

	labels = np.empty((0,1))

	i = 0

	for fn in fileNames:

		label = '0'
		if 'Coughing' in fn:
			label = '1'

		labels = np.vstack([labels, label])

		mfccs,chroma,mel,contrast,tonnetz = extract_feature(fn)

		# stack data
		Xmfccs = np.vstack([Xmfccs, mfccs])
		Xchroma = np.vstack([Xchroma, chroma])
		Xmel = np.vstack([Xmel, mel])
		Xcontrast = np.vstack([Xcontrast, contrast])
		Xtonnetz = np.vstack([Xtonnetz, tonnetz])

		if i > 10:
			break

		i += 1

	return None


def store_features(labels, Xmfccs, Xchroma, Xmel, Xcontrast, Xtonnetz):

	data = np.hstack([labels, Xmfccs])
	df = pd.DataFrame(data=data)
	df.to_pickle('./features/mfccs.h5')
	print('./features/mfccs.h5 saved')

	data = np.hstack([labels, Xchroma])
	df = pd.DataFrame(data=data)
	df.to_pickle('./features/chroma.h5')
	print('./features/chroma.h5 saved')

	data = np.hstack([labels, Xmel])
	df = pd.DataFrame(data=data)
	df.to_pickle('./features/mel.h5')
	print('./features/mel.h5 saved')

	data = np.hstack([labels, Xcontrast])
	df = pd.DataFrame(data=data)
	df.to_pickle('./features/contrast.h5')
	print('./features/contrast.h5 saved')

	data = np.hstack([labels, Xtonnetz])
	df = pd.DataFrame(data=data)
	df.to_pickle('./features/tonnetz.h5')
	print('./features/tonnetz.h5 saved')


	return None





if __name__ == "__main__":

	trainList, testList = split_train_test_list()

	print('train list length: %d'%len(trainList))
	print('test list length: %d'%len(testList))










































