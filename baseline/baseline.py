import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from preprocessData import *

'''

n_trees = 500 trees, 
test_size = 0.2
pca components = 10
entire data set used
test accuracy: 0.833697234352

'''

# hyperparams
pca_components = 10
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

		if count % 10 == 0:
			print(count, " files processed.")

		if count >= max_samples and max_samples > 0:
			break

	return labels, X



def generate_cough_model(trainListCough,
	trainListOther,
	testListCough,
	testListOther,
	batch_size = 256):
	
	assert batch_size % 2 == 0

	batch_size_half = batch_size // 2

	max_idx =  min(len(trainListCough), len(trainListOther)) - batch_size_half - 1
	counter = 0

	print("computing cough model for training data...")

	for idx in range(0, max_idx, batch_size_half):

		counter += 1
		end = idx + batch_size_half
		train_batch_cough = trainListCough[idx:end]
		train_batch_other = trainListOther[idx:end]
		train_batch_cough.extend(train_batch_other)

		train_filenames = train_batch_cough

		print('processing batch %d'%counter)
		labels_train, X_train = stack_spectrograms(train_filenames)

		## compute model
		# PCA on stacked data
		X_train = sklearn.preprocessing.scale(X_train, axis=0, with_mean=True, with_std=False)
		pca = decomposition.PCA()
		pca.n_components = pca_components
		X_reduced_train = pca.fit_transform(X_train)

		# reconstruct stacked spectrograms
		X_projected_train = pca.inverse_transform(X_reduced_train)
		residual_error_train = np.mean((X_train - X_projected_train) ** 2, axis = 1)

		# reconstruct spectrogram to compute energy features for training set
		for i in range(np.shape(X_projected_train)[0]):

			stft_reduced = np.reshape(X_projected_train[i,:], (n_freq, time_frames))
			
			# compute energy of entire FFt
			energy = np.mean(librosa.feature.rmse(S=stft_reduced))
			if i == 0:
				energy_features_train = energy
			else:
				energy_features_train = np.vstack([energy_features_train, energy])

		# merge features into single data matrix
		cough_model_train_ = np.column_stack((X_reduced_train, energy_features_train, residual_error_train))

		if counter == 1:
			cough_model_train = cough_model_train_
			all_train_labels = labels_train
		else:
			cough_model_train = np.vstack([cough_model_train, cough_model_train_])
			all_train_labels.extend(labels_train)



	# compute test set cough model
	testListCough.extend(testListOther)
	test_filenames = testListCough
	print('length of test files: ',len(test_filenames))
	labels_test, X_test = stack_spectrograms(test_filenames)

	## compute model for test data
	print("computing cough model for test data...")
	# PCA on stacked data
	X_test = sklearn.preprocessing.scale(X_test, axis=0, with_mean=True, with_std=False)
	pca = decomposition.PCA()
	pca.n_components = pca_components
	X_reduced_test = pca.fit_transform(X_test)

	# reconstruct stacked spectrograms
	X_projected_test = pca.inverse_transform(X_reduced_test)
	residual_error_test = np.mean((X_test - X_projected_test) ** 2, axis = 1)

	# reconstruct spectrogram to compute energy features for test set
	for i in range(np.shape(X_projected_test)[0]):

		stft_reduced = np.reshape(X_projected_test[i,:], (n_freq, time_frames))
		
		# compute energy of entire FFT
		energy = np.mean(librosa.feature.rmse(S=stft_reduced))
		if i == 0:
			energy_features_test = energy
		else:
			energy_features_test = np.vstack([energy_features_test, energy])


	# merge features into single data matrix
	cough_model_test = np.column_stack((X_reduced_test, energy_features_test, residual_error_test))


	return all_train_labels, cough_model_train, labels_test, cough_model_test




if __name__ == "__main__":

	# get lists for datafiles; split into training and test sets
	trainListCough, trainListOther, testListCough, testListOther = split_train_test_list(n_samples=2048)

	# compute cough model
	y_train, X_train, y_test, X_test = generate_cough_model(trainListCough, trainListOther, 
															testListCough, testListOther)

	print('******************')
	print('y_train shape: ', np.shape(y_train))
	print('X_train shape: ', np.shape(X_train))
	print('y_test shape: ', np.shape(y_test))
	print('X_test shape: ', np.shape(X_test))
	print('******************')

	# train random Forest Classifier
	rf = RandomForestClassifier(n_estimators = n_trees, max_features=int(12))
	rf.fit(X_train, y_train)

	predictions_test = rf.predict(X_test)
	predictions_train = rf.predict(X_train)

	train_accuracy = np.sum(predictions_train == y_train)/len(X_train)
	test_accuracy = np.sum(predictions_test == y_test)/len(X_test)

	aucroc_score_test = roc_auc_score(y_test, predictions_test)
	aucroc_score_train = roc_auc_score(y_train, predictions_train)

	# print accuracy
	print('test accuracy: %f'%test_accuracy)
	print('train accuracy: %f'%train_accuracy)
	print('auc roc score test: %f'%aucroc_score_test)
	print('auc roc score train: %f'%aucroc_score_train)

























