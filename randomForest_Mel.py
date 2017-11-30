import numpy as np
import librosa
import glob
import os
import sklearn
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




def compute_melspec(fn):

	# accuracy ~0.71

	X, sample_rate = librosa.load(fn)
	melspec = librosa.feature.melspectrogram(X, sr=sample_rate).T # time x n_mel
	mel = np.mean(melspec, axis=0) # take mean over time
	return melspec



def compute_stacked_melspec(fn):

	'''
	alternative model (closer to baseline):
	take first n timepoints, i.e. cut melspec matrix at n,
	then stack rows into a single vector representing a sound
	--> how determine n?
	'''

	# accuracy ~0.81

	X, sample_rate = librosa.load(fn)
	melspec = librosa.feature.melspectrogram(X, sr=sample_rate).T # time x n_mel
	keep_time = 50
	if melspec.shape[0] >= keep_time:
		stacked_melspec = np.empty(0)
		for i in range(keep_time):
			stacked_melspec = np.hstack([stacked_melspec, melspec[i,:]])
		stacked_melspec /= np.linalg.norm(stacked_melspec)

		return stacked_melspec
	else:
		return np.empty(0)


def parse_audio_files(parent_dir, sub_dirs, file_ext = "*.wav"):

	features, labels = np.empty((0,6400)), np.empty(0)
	count_files = 0

	for sub_dir in sub_dirs:
		for sub_dir2 in sub_dirs[sub_dir]:
			for fn in glob.glob(os.path.join(parent_dir, sub_dir, sub_dir2, file_ext)):
				try:
					#mel = compute_melspec(fn)
					mel = compute_stacked_melspec(fn)
					
				except Exception as e:
					print("Error encountered while parsing file: ", fn)
					print("Error message: ", e)
					continue

				if len(mel) > 0:

					features = np.vstack([features, mel])

					new_label = 0
					if sub_dir == '04_coughing':
						new_label = 1

					labels = np.append(labels, new_label)
				
				count_files += 1
				if count_files % 100 == 0:
					print('checkpoint: parsed ', count_files, 'files.')

				

	return np.array(features), np.array(labels, dtype=np.int)


# hyperparameters
test_size = 0.2
pca_components = 250
n_trees = 1000

# directories with data
parent_dir = 'data'
sub_dirs = {
			'04_coughing': {'Close (cc)', 'Distant (cd)'},
			'05_Other Control Sounds': {'01_Throat Clearing', '02_Laughing', '03_Speaking', '04_Spirometer'}
			}



# compute features, split into train and test set
X, y = parse_audio_files(parent_dir, sub_dirs)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

# subtract means
X_train = sklearn.preprocessing.scale(X_train, axis=0, with_mean=True, with_std=False)
X_test = sklearn.preprocessing.scale(X_test, axis=0, with_mean=True, with_std=False)

# pca decomposition
pca = decomposition.PCA()
pca.n_components = pca_components
Xtrain_reduced = pca.fit_transform(X_train) #the data were going to train on
Xtest_reduced = pca.transform(X_test)


# fit random forest
rf = RandomForestClassifier(n_estimators = n_trees)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# accuracy
print('test accuracy: ', np.sum(predictions == y_test)/len(X_test))























































