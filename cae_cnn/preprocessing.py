# Authors: Kevin Kipfer, Filipe Barata, Maurice Weber

import librosa
import numpy as np


def denoise_spectrogram(spect):
	'''
	this has to be implemented
	'''

	return spect


def standardize(timeSignal):

	maxValue = np.max(timeSignal)
	minValue = np.min(timeSignal)
	timeSignal = (timeSignal - minValue)/(maxValue - minValue) 

	return timeSignal



def extract_Signal_Of_Importance(signal, window, sample_rate ):
	"""
	extract a window around the maximum of the signal
	input: 	signal
	window -> size of a window
	sample_rate 
	"""

	window_size = int(window * sample_rate)			

	start = max(0, np.argmax(np.abs(signal)) - (window_size // 2))
	end = min(np.size(signal), start + window_size)
	signal = signal[start:end]

	length = np.size(signal)
	assert length <= window_size, 'extracted signal is longer than the allowed window size'
	if length < window_size:
		#pad zeros to the signal if too short
		signal = np.concatenate((signal, np.zeros(window_size-length))) 
	return signal


def fetch_samples(files,
	is_training=True,
	hop_length=120,
	bands=16,
	window=0.16,
	do_denoise=False):
	'''
	load, preprocess, normalize a sample
	input: a list of strings
	output: the processed features from each sample path in the input
	'''
	batch_features = []
	for f in files:
		try:
			timeSignal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
		except ValueError as e:
			print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
			raise e

		timeSignal = extract_Signal_Of_Importance(timeSignal, window, sample_rate)

		timeSignal = standardize(timeSignal)

		mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)

		if do_denoise:
			mfcc = denoise_spectrogram(mfcc)

		batch_features.append(mfcc)

		return np.array(batch_features)











































































































