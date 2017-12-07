import glob
import random
import copy

import numpy as np
import pandas as pd
import librosa



def split_train_test_list(TEST_RATIO=0.2, SAMPLE_SIZE=1000):

	# generate random list of test participants
	testParticipants = []
	for i in np.random.choice(range(1,48), int(47*TEST_RATIO), replace=False):
		if i < 10:
			testParticipants.append("p0" + str(i))
		else:
			testParticipants.append("p" + str(i))


	# read cough data
	coughCloseList = glob.glob("./AudioData/Coughing/Close (cc)/*.wav")
	coughDistantList = glob.glob("./AudioData/Coughing/Distant (cd)/*.wav")
	coughAll = coughCloseList
	coughAll.extend(coughDistantList)

	testListCough = []
	trainListCough = coughAll[:]
	for name in coughAll:
	    for nameToExclude in testParticipants:
	        if nameToExclude in name:
	            
	            testListCough.append(name)
	            trainListCough.remove(name)



	# read other data
	throat = glob.glob("./AudioData/Other Control Sounds/01_Throat Clearing/*.wav")
	laughing =  glob.glob("./AudioData/Other Control Sounds/02_Laughing/*.wav")
	speaking = glob.glob("./AudioData/Other Control Sounds/03_Speaking/*.wav")

	other = throat[:]
	other.extend(laughing)
	other.extend(speaking)

	testListOther = []
	trainListOther = other[:]

	for name in other:
		for nameToExclude in testParticipants:
			if nameToExclude in name:
				
				testListOther.append(name)
				trainListOther.remove(name)

	# train and test on subset



	if SAMPLE_SIZE > 0:


		TRAIN_SIZE = (1 - TEST_RATIO) * SAMPLE_SIZE
		TEST_SIZE = TEST_RATIO * SAMPLE_SIZE

		random.seed(42)
		train_random_coughs_numbers = random.sample(range(0,np.size(trainListCough)), int(TRAIN_SIZE/2))
		train_random_other_numbers = random.sample(range(0,np.size(trainListOther)), int(TRAIN_SIZE/2))

		test_random_coughs_numbers = random.sample(range(0,np.size(testListCough)), int(TEST_SIZE/2))
		test_random_other_numbers = random.sample(range(0,np.size(testListOther)), int(TEST_SIZE/2))

		trainFileNames = [trainListCough[x] for x in train_random_coughs_numbers]
		trainFileNames.extend([trainListOther[x] for x in train_random_other_numbers])
		
		testFileNames = [testListCough[x] for x in test_random_coughs_numbers]
		testFileNames.extend([testListOther[x] for x in test_random_other_numbers])

	else:

		trainFileNames = trainListCough
		trainFileNames.extend(trainListOther)

		testFileNames = testListCough
		testFileNames.extend(testListOther)

	return trainFileNames, testFileNames




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
	

	trainListOther, testListOther, trainListCough, testListCough = split_train_test_list()

	compute_features_matrix(testListCough)










































