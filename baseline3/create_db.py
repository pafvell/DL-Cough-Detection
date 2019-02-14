#Authors: Filipe Barata

import json
import time

import h5py
import librosa
import cv2

from utils import *

#loading configuration
with open('config.json') as json_data_file:
	config = json.load(json_data_file)

# read params from config file
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]
DB_VERSION = config["general"]["DB_VERSION"]
HOP = config["create_db"]["HOP"]
#NEW
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

DB_FILENAME = '/data_%s.h5'%(DB_VERSION)

RECONSTRUCT = True

print('Data will be stored to %s'%(DB_ROOT_DIR + DB_FILENAME))




def stack_humoments(filenames, windowsize):
	'''
	Input:
		A list of filenames
	Output:
		A matrix X with vectorized hu moments as rows
		A list labels with entries 1 (cough sound) and 0 (other sound)
	'''
	import multiprocessing
	from joblib import Parallel, delayed

	labels = []
	X = []
	device_classes = []

	numcores = multiprocessing.cpu_count()

	results = Parallel(n_jobs=numcores)(delayed(computeOneIteration)(fn, windowsize) for fn in filenames)

	for i in range(0, len(results)):

		try:
			tuple0 = results[i][0]
			tuple1 = results[i][1]
			tuple2 = results[i][2]
		except Exception or TypeError as e:

			#continues whenever a file is not long enough
			#print(e)
			continue


		if i == 0:
			X = tuple0
			labels = [tuple1]
			device_classes = [tuple2]
		else:
			X = np.vstack([X, tuple0])
			labels.append(tuple1)
			device_classes.append(tuple2)

	return device_classes, labels, X


def extract_Signal_Of_Importance(file_name, half_windowsize, filtering = FILTERING, downsampling_factor = DOWNSAMPLING_FACTOR, sampling_rate = SAMPLING_RATE
	):
	'''
	Input:
		file_name the file to be loaded
		window_size the window to be cut out
	Output:
		A window of the audio signal of size window_size centered around the max absolute value of the entire sequence
	'''
	from scipy import signal

	try:
		X, sample_rate = librosa.load(file_name, sr = sampling_rate)
		if filtering:
			X = signal.decimate(X, downsampling_factor)
			sample_rate = sample_rate / downsampling_factor
	except Exception or ValueError as e:
		print("An error occurred while parsing file ", file_name, " :\n")
		print(e)
		return [], -1



	maxValue = np.max(np.abs(X))
	absX = np.abs(X)
	indMax = absX.tolist().index(maxValue)
	numberOfSamples = np.ceil(sample_rate * half_windowsize)
	startInd = int(np.max(indMax - numberOfSamples, 0))
	maxLeng = np.size(X)

	if startInd + 2*numberOfSamples > maxLeng - 1:
		endInd = int(maxLeng - 1)
		startInd = int(endInd - 2*numberOfSamples)
	else:
		endInd = int(startInd + 2*numberOfSamples)

	signal = X[startInd:endInd]
	return signal, sample_rate



def normalizePSD(psd):
	(n, m) = np.shape(psd)
	psd_norm = np.zeros((n, m))
	for i in range(0, m):
		psd_norm[:, i] = psd[:, i] * 1.0/n * np.linalg.norm(psd[:, i])**2
	return psd_norm

def computeLocalHuMoments(stardardized_time_signal, sample_rate, hop_length=HOP, w=5, alternativeComp=ALTERNATIVE_COMP, nmels=75, nfft=N_FFT, paper =PAPER):
	import skimage.util
	import skimage.measure
	import scipy.fftpack
	import scipy.signal

	if paper:
		f,t,psd = scipy.signal.spectrogram(x=stardardized_time_signal, fs=sample_rate, window=('kaiser',3.5), nperseg = NPERSEG, noverlap= NOVERLAP, nfft=N_FFT, scaling='density', return_onesided=True, mode='psd')

		melFilterBank = librosa.filters.mel(sample_rate, nfft, n_mels=nmels, fmin=0.0, fmax=2000.0)
		psd_norm = normalizePSD(psd)

		energymatrix = np.log(np.matmul(melFilterBank,psd_norm))

	else:

		mfcc = librosa.feature.melspectrogram(y=stardardized_time_signal, sr=sample_rate, n_mels=75, power=1, hop_length=hop_length,n_fft=N_FFT)
		energymatrix = np.log(mfcc)


	(n,m) = np.shape(energymatrix)
	if m%w != 0:
		mzero = m + w - m%w
	else:
		mzero = m

	energymatrix_resized = np.zeros((n,mzero))
	energymatrix_resized[:n, :m] = energymatrix
	energymatrix_blocks = skimage.util.view_as_blocks(energymatrix_resized, block_shape=(w, w))

	nrow = np.shape(energymatrix_blocks)[0]
	ncol = np.shape(energymatrix_blocks)[1]
	humatrix = np.zeros((nrow,ncol))
	for i in range(0,nrow):
		for j in range(0,ncol):
			if alternativeComp:
				M = skimage.measure.moments(energymatrix_blocks[i,j])
				cr = M[1, 0] / M[0, 0]
				cc = M[0, 1] / M[0, 0]
				momentscentral = skimage.measure.moments_central(energymatrix_blocks[i,j], (cr, cc))
				normalizedCentralmoments= skimage.measure.moments_normalized(momentscentral)
				humoments = skimage.measure.moments_hu(normalizedCentralmoments)

			else:
				humoments = cv2.HuMoments(cv2.moments(energymatrix_blocks[i,j])).flatten()

			humatrix[i, j] = humoments[0]

	TQ = scipy.fftpack.dct(humatrix,axis=1)
	result = TQ[1:-1,:]
	return result



def standardize(timeSignal):
         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)
         timeSignal = (timeSignal - minValue)/(maxValue - minValue)
         return timeSignal


def computeOneIteration(fn, windowsize):

	Signal, sr = extract_Signal_Of_Importance(fn, windowsize/2)
	if round(np.size(Signal) / float(sr), 2) == windowsize:


		TQ_matrix = computeLocalHuMoments(Signal, sr)
		device = get_device(fn)

		# reshape spectrogram into row vector
		row_vec = np.empty((0))
		for i in range(np.shape(TQ_matrix)[1]):
			row_vec = np.hstack([row_vec, TQ_matrix[:, i]])

		# get label
		label = int("Coughing" in fn)

		X = row_vec
		labels = label

		return X, labels, device





def generate_cough_model(file_list):

	# compute first batch
	device_classes, labels, features = stack_humoments(file_list, WINDOW_SIZE)


	print("shape of labels: {label_shape}, shape of cough model: {cough_shape}"\
		.format(label_shape=np.shape(labels), cough_shape=np.shape(features)))

	return device_classes, labels, features



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

	#mean_train = np.mean(train_features, axis=0)
	#std_train = np.std(train_features, axis=0)

	#train_features -= mean_train
	#train_features /= std_train


	print("#"*10, "processing test data", "#"*10)
	test_list = testListCough
	test_list.extend(testListOther)
	np.random.shuffle(test_list)
	print("number of test samples:", len(test_list))
	test_devices, test_labels, test_features = generate_cough_model(test_list)

	#test_features -= mean_train
	#test_features /= std_train

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






























