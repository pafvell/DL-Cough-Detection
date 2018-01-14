import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier

from preprocessData import *

'''

n_trees = 500 trees, 
test_size = 0.2
pca components = 10
entire data set used
test accuracy: 0.833697234352

'''

# hyperparams
pca_components = 15
n_trees = 500
test_size = 0.2
n_freq = 1025
time_frames = 14
max_samples = -1


def stack_spectrograms(filenames):

	count = 0

	for fn in filenames:

		# get signal of importance
		Signal, sr = extract_Signal_Of_Importance(fn)
		if np.size(Signal) / sr != 0.32:
			continue
		
		stft = np.abs(librosa.stft(Signal))

		# reshape into row vector
		# row_vec = np.reshape(stft.T, (n_freq*time_frames))
		row_vec = np.empty((0))
		for i in range(np.shape(stft)[1]):
			row_vec = np.hstack([row_vec, stft[:,i]])

		# normalize vector
		row_vec /= np.linalg.norm(row_vec)

		# get label
		label = int("Coughing" in fn)
		
		if count == 0:
			X = row_vec
			labels = [label]
		else:
			X = np.vstack([X, row_vec])
			labels.append(label)
		
		count += 1

		print(count, " files parsed.")

		if count >= max_samples and max_samples > 0:
			break

	return labels, X



def generate_cough_model(train_filenames, test_filenames):


	## load data
	print("Reading Data...")
	labels_train, X_train = stack_spectrograms(train_filenames)
	print("shape of training features: ", np.shape(X_train))
	labels_test, X_test = stack_spectrograms(test_filenames)
	print("shape of testing features: ", np.shape(X_test))


	## compute model
	print("computing cough model for training and test data")
	# PCA on stacked data
	X_train = sklearn.preprocessing.scale(X_train, axis=0, with_mean=True, with_std=False)
	X_test = sklearn.preprocessing.scale(X_test, axis=0, with_mean=True, with_std=False)
	pca = decomposition.PCA()
	pca.n_components = pca_components
	X_reduced_train = pca.fit_transform(X_train)
	X_reduced_test = pca.transform(X_test)


	# reconstruct stacked spectrograms
	X_projected_train = pca.inverse_transform(X_reduced_train)
	X_projected_test = pca.inverse_transform(X_reduced_test)
	residual_error_train = np.mean((X_train - X_projected_train) ** 2, axis = 1)
	residual_error_test = np.mean((X_test - X_projected_test) ** 2, axis = 1)


	# reconstruct spectrogram to compute energy features for training set
	for i in range(np.shape(X_projected_train)[0]):

		stft_reduced = np.reshape(X_projected_train[i,:], (n_freq, time_frames))
		
		# compute energy of entire FFt
		energy = np.mean(librosa.feature.rmse(S=stft_reduced))
		if i == 0:
			energy_features_train = energy
		else:
			energy_features_train = np.vstack([energy_features_train, energy])


	# reconstruct spectrogram to compute energy features for test set
	for i in range(np.shape(X_projected_test)[0]):

		stft_reduced = np.reshape(X_projected_test[i,:], (n_freq, time_frames))
		
		# compute energy of entire FFt
		energy = np.mean(librosa.feature.rmse(S=stft_reduced))
		if i == 0:
			energy_features_test = energy
		else:
			energy_features_test = np.vstack([energy_features_test, energy])


	# merge features into single data matrix
	cough_model_train = np.column_stack((X_reduced_train, energy_features_train, residual_error_train))
	cough_model_test = np.column_stack((X_reduced_test, energy_features_test, residual_error_test))


	return labels_train, cough_model_train, labels_test, cough_model_test




if __name__ == "__main__":

	# get lists for datafiles; split into training and test sets
	trainList, testList = split_train_test_list(TEST_RATIO = test_size, SAMPLE_SIZE = max_samples)

	# compute cough model
	y_train, X_train, y_test, X_test = generate_cough_model(trainList, testList)

	# train random Forest Classifier
	rf = RandomForestClassifier(n_estimators = n_trees)
	rf.fit(X_train, y_train)
	predictions = rf.predict(X_test)

	# print accuracy
	print('test accuracy: ', np.sum(predictions == y_test)/len(X_test))

























