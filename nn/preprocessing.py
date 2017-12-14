#Author: Kevin Kipfer

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

plt.style.use('ggplot')



#########################################################################################
#
# Data Augmentation
#
#########################################################################################


def augment_sound(signal):

	#TODO augment
	
	# z.B. http://ofai.at/~jan.schlueter/code/augment/

	return signal




#########################################################################################
#
# Extracting Features
#
#########################################################################################



def standardize(signal):

	#TODO

	



        print ( 'max signal: %d'%np.max(signal))


	
	return signal


def extract_Signal_Of_Importance(signal, frames, augmentation_factor=0, sample_rate=22050, hop_length=512):
        """
	extract a window around the maximum of the signal
	input: 	signal
		frames -> nr of frames gives the size of the window
		avg_t_cough -> average time of a cough
        """
        window_size = hop_length * (frames - 1) 
        start = max(0, np.argmax(np.abs(signal)) - window_size // 2 + augmentation_factor)
        end = min(np.size(signal), start + window_size)
        start = max(0, end - window_size)
        signal = signal[start:end]
        length = np.size(signal)
        assert length <= window_size, 'extracted signal is longer than the allowed window size'
        if length < window_size:
                #pad zeros to the signal if 
                signal = np.concatenate((signal, np.zeros(window_size-length))) 
        return signal


def fetch_samples(files, 
		  is_training=True, 
		  bands = 20, 
		  frames = 41):
	"""
	load, preprocess, normalize a sample
	input: a list of strings
	output: the processed features from each sample path in the input
	"""
	batch_features = []
	for f in files:
                try:
                       signal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e
                signal = extract_Signal_Of_Importance(signal, frames, sample_rate)
                if is_training:
                      augment_sound(signal)
                mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                mfcc = standardize(mfcc)
                batch_features.append(mfcc)

	batch_features = np.asarray(batch_features).reshape(len(files),frames,bands)
	return np.array(batch_features)



