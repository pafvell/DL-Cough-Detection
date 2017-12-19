#Author: Kevin Kipfer

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import signal
import matplotlib.pyplot as plt

plt.style.use('ggplot')

maxValue = 1.7
minValue = -1.8

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

'''
def fit_scale(timeSignal):
         global maxValue
         global minValue

         maxValue_ = np.max(timeSignal)
         minValue_ = np.min(timeSignal)
         if maxValue_ > maxValue:
               print ( 'new max: %f vs %f'%(maxValue_, maxValue))
               maxValue = maxValue_
         if minValue_ < minValue:
               minValue = minValue_
               print ( 'new min: %f vs %f'%(minValue_, minValue))
'''

def standardize(timeSignal):

	 #TODO
         maxValue_ = np.max(timeSignal)
         minValue_ = np.min(timeSignal)
         timeSignal = (timeSignal - minValue)/(maxValue - minValue) 

         #but since timeSignal is in [-1.8,1.7]
         #timeSignal /= 1.8
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
		  bands = 16,
		  window = 0.16):
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

                timeSignal = extract_Signal_Of_Importance(timeSignal, window, sample_rate)

                if is_training:
                      augment_sound(timeSignal)

                #fit_scale(timeSignal)
                timeSignal = standardize(timeSignal)

                mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)

                batch_features.append(mfcc)

	#batch_features = np.asarray(batch_features).reshape(len(files),frames,bands)
	return np.array(batch_features)



