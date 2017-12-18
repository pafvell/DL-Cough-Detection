#Author: Kevin Kipfer

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import signal
import matplotlib.pyplot as plt

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



def standardize(timeSignal):

	#TODO

         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)

         timeSignal = (timeSignal - minValue)/(maxValue - minValue)
         return timeSignal


def extract_Signal_Of_Importance(signal, frames, sample_rate, augmentation_factor=0, hop_length=176):
        """
	extract a window around the maximum of the signal
	input: 	signal
		frames -> nr of frames gives the size of the window
		avg_t_cough -> average time of a cough
        """

        window_size = int(0.16 * sample_rate)#hop_length * (frames - 1)

        start = max(0, np.argmax(np.abs(signal)) - (window_size // 2))
        end = min(np.size(signal), start + window_size)
        signal = signal[start:end]

        length = np.size(signal)
        assert length <= window_size, 'extracted signal is longer than the allowed window size'
        if length < window_size:
                #pad zeros to the signal if 
                signal = np.concatenate((signal, np.zeros(window_size-length))) 
        return signal


def fetch_samples(files, 
		  is_training=True, 
		  bands = 16,
		  frames = 64):
	"""
	load, preprocess, normalize a sample
	input: a list of strings
	output: the processed features from each sample path in the input
	"""
	batch_features = []
	for f in files:
                try:
                       timeSignal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e
                timeSignal = extract_Signal_Of_Importance(timeSignal, frames, sample_rate)
                if is_training:
                      augment_sound(timeSignal)
                timeSignal = standardize(timeSignal)

                mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=120)

                batch_features.append(mfcc)

	#batch_features = np.asarray(batch_features).reshape(len(files),frames,bands)
	return np.array(batch_features)



