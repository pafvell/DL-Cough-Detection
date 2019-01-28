#Authors: Filipe Barata, Maurice Weber, Kevin Kipfer

import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from utils import *


# hyperparameters
pca_components = 10
n_trees = 500
max_depth = 10
max_features = 12
n_freq = 1025
time_frames = 7



def stack_humoments(filenames,
					print_every_n_steps=50
					):
	'''
	Input:
		A list of filenames
	Output:
		A matrix X with vectorized spectrograms as rows
		A list labels with entries 1 (cough sound) and 0 (other sound)
	'''

	count = 0

	for fn in filenames:

		# get signal of importance
		Signal, sr = extract_Signal_Of_Importance(fn)
		if np.size(Signal) / sr != 0.05:
			continue

		TQ_matrix = computeLocalHuMoments(Signal, sr)

		# reshape spectrogram into row vector
		row_vec = np.empty((0))
		for i in range(np.shape(TQ_matrix)[1]):
			row_vec = np.hstack([row_vec, TQ_matrix[:,i]])


		# get label
		label = int("Coughing" in fn)
		
		if count == 0:
			X = row_vec
			labels = [label]
		else:
			X = np.vstack([X, row_vec])
			labels.append(label)
		
		count += 1

		if count % print_every_n_steps == 0:
			print(count, " files processed.")

	return labels, X



def generate_cough_model(trainListCough,
	trainListOther,
	testListCough,
	testListOther,
	batch_size = 256):
	'''
	Input:
		Lists with filenames for training and test data, cough, and non-cough sounds
	Output:
		list of labels for training and test data
		feature matrices cough_model_train and cough_model_test
	'''

	print("computing cough model for training data...")
	trainListCough.extend(trainListOther)
	train_filenames = trainListCough
	all_train_labels, cough_model_train = stack_humoments(train_filenames)


	## compute model for test data
	print("computing cough model for test data...")
	
	testListCough.extend(testListOther)
	test_filenames = testListCough

	labels_test, cough_model_test = stack_humoments(test_filenames)

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
	rf = KNeighborsClassifier(n_neighbors=1, weights='distance', p=2)
	rf.fit(X_train, y_train)

	predictions_test = rf.predict(X_test)
	predictions_train = rf.predict(X_train)

	train_accuracy = np.sum(predictions_train == y_train)/len(X_train)
	test_accuracy = np.sum(predictions_test == y_test)/len(X_test)

	aucroc_score_test = roc_auc_score(y_test, predictions_test)
	aucroc_score_train = roc_auc_score(y_train, predictions_train)

	# print accuracy
	print('*********  RESULTS *********')
	print('test accuracy: %f'%test_accuracy)
	print('train accuracy: %f'%train_accuracy)
	print('auc roc score test: %f'%aucroc_score_test)
	print('auc roc score train: %f'%aucroc_score_train)

























