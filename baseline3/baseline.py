#Authors: Filipe Barata, Maurice Weber, Kevin Kipfer

import cv2
import librosa
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from utils import *

ROOT_DIR = '../../Audio_Data'

# hyperparameters
n_neighbors = 3
filtering = False
downsampling_factor = 2
sampling_rate = 22050
window_size = 0.45
nperseg = 441
noverlap= 221
nfft = 4096
hop_length = 256
alternativeComp=False
paper = True
weights='distance'
p=2



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

	numcores = multiprocessing.cpu_count()

	results = Parallel(n_jobs=numcores)(delayed(computeOneIteration)(fn, windowsize) for fn in filenames)

	for i in range(0, len(results)):

		try:
			tuple0 = results[i][0]
			tuple1 = results[i][1]
		except Exception or TypeError as e:
			#print(e)
			continue


		if i == 0:
			X = tuple0
			labels = [tuple1]
		else:
			X = np.vstack([X, tuple0])
			labels.append(tuple1)

	return labels, X

def extract_Signal_Of_Importance(file_name, half_windowsize, filtering = filtering, downsampling_factor = downsampling_factor, sampling_rate = sampling_rate
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

def computeLocalHuMoments(stardardized_time_signal, sample_rate, hop_length=hop_length, w=5, alternativeComp=alternativeComp, nmels=75, nfft=nfft, paper = paper):
	import skimage.util
	import skimage.measure
	import scipy.fftpack
	import scipy.signal

	if paper:
		f,t,psd = scipy.signal.spectrogram(x=stardardized_time_signal, fs=sample_rate, window=('kaiser',3.5), nperseg = nperseg, noverlap= noverlap, nfft=nfft, scaling='density', return_onesided=True, mode='psd')

		melFilterBank = librosa.filters.mel(sample_rate, nfft, n_mels=nmels, fmin=0.0, fmax=2000.0)
		psd_norm = normalizePSD(psd)

		energymatrix = np.log(np.matmul(melFilterBank,psd_norm))

	else:

		mfcc = librosa.feature.melspectrogram(y=stardardized_time_signal, sr=sample_rate, n_mels=75, power=1, hop_length=hop_length,n_fft=nfft)
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

		# reshape hu moment into row vector
		row_vec = np.empty((0))
		for i in range(np.shape(TQ_matrix)[1]):
			row_vec = np.hstack([row_vec, TQ_matrix[:, i]])

		# get label
		label = int("Coughing" in fn)

		X = row_vec
		labels = label

		return X, labels





def generate_cough_model(trainListCough,
	trainListOther,
	testListCough,
	testListOther,
	window_size = window_size):
	'''
	Input:
		Lists with filenames for training and test data, cough, and non-cough sounds
	Output:
		list of labels for training and test data
		feature matrices cough_model_train and cough_model_test
	'''

	#trainListCough = trainListCough[:10]
	#trainListOther = trainListOther[:10]

	print("computing cough model for training data...")
	trainListCough.extend(trainListOther)
	train_filenames = trainListCough
	all_train_labels, cough_model_train = stack_humoments(train_filenames, window_size)


	## compute model for test data
	print("computing cough model for test data...")

	#testListCough = testListCough[:10]
	#testListOther = testListOther[:10]

	testListCough.extend(testListOther)
	test_filenames = testListCough

	labels_test, cough_model_test = stack_humoments(test_filenames, window_size)

	return all_train_labels, cough_model_train, labels_test, cough_model_test



def split_train_test_list():
	'''
	This procedure splits the data files stored in the root directory ROOT_DIR into test and training data
	'''


	# participants used in the test-set, generated at random
	listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] 
	list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
							'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']



	## Reading cough data
	print ('use data from root path %s'%ROOT_DIR)

	coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
	assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'


	# remove broken files
	for broken_file in list_of_broken_files:
		broken_file = os.path.join(ROOT_DIR, broken_file)
		if broken_file in coughAll:
			print ('file ignored: %s'%broken_file )
			coughAll.remove(broken_file)


	# split cough files into test- and training-set
	testListCough = []
	trainListCough = coughAll
	for name in coughAll:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListCough.append(name)
				trainListCough.remove(name)

	print('nr of test samples coughing: %d' % len(testListCough))



	## Reading other data
	other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

	testListOther = []
	trainListOther = other
	for name in other:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListOther.append(name)
				trainListOther.remove(name)

	print('nr of test samples NOT coughing: %d' % len(testListOther))

	return trainListCough, trainListOther, testListCough, testListOther




if __name__ == "__main__":

	# get lists for datafiles; split into training and test sets
	trainListCough, trainListOther, testListCough, testListOther = split_train_test_list()

	# compute cough model
	y_train, X_train, y_test, X_test = generate_cough_model(trainListCough, trainListOther, 
															testListCough, testListOther)

	# train random Forest Classifier
	knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
	knn.fit(X_train, y_train)

	predictions_test = knn.predict(X_test)
	predictions_train = knn.predict(X_train)

	train_accuracy = np.sum(predictions_train == y_train)/float(len(X_train))
	test_accuracy = np.sum(predictions_test == y_test)/float(len(X_test))

	probability_test = knn.predict_proba(X_test)
	knn_score = knn.score(X_test, y_test)

	probVec = probability_test[:, 1]

	fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, probVec)
	df = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds': thresholds})
	df.to_csv("knn_roc_curve_rf.csv")

	aucroc_score_test = roc_auc_score(y_test, predictions_test)
	aucroc_score_train = roc_auc_score(y_train, predictions_train)

	test_mcc = sklearn.metrics.matthews_corrcoef(y_test,predictions_test)
	train_mcc = sklearn.metrics.matthews_corrcoef(y_train, predictions_train)

	confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predictions_test)

	# print accuracy
	print('*********  RESULTS *********')
	print('test accuracy: %f'%test_accuracy)
	print('train accuracy: %f'%train_accuracy)
	print('auc roc score test: %f'%aucroc_score_test)
	print('auc roc score train: %f'%aucroc_score_train)
	print(confusion_matrix)
	print('test matthew correlation coeff.: %f'%test_mcc)
	print('train matthew correlation coeff.: %f'%train_mcc)
























