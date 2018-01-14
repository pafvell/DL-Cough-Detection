#Author: Kevin Kipfer

import librosa
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import signal
from scipy.ndimage.morphology import binary_erosion, binary_dilation
#import matplotlib.pyplot as plt

#plt.style.use('ggplot')

maxValue = 1.7
minValue = -1.8

#########################################################################################
#
# Data Augmentation
# z.B. http://ofai.at/~jan.schlueter/code/augment/
#
#########################################################################################


def pitch_shift(signal, sr, n_steps=5):
  '''
  
  '''
  # as in https://librosa.github.io/librosa/generated/librosa.effects.pitch_shift.html#librosa.effects.pitch_shift

  signal = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

  return signal


def time_stretch(signal, rate):
  '''
  Input:
    signal; sound signal to be stretched
    rate; stretch factor: if rate < 1 then signal is slowed down, otherwise sped up
  Output:
    stretched/compressed signal
  CAUTION: changes time length of signal -> apply this before extract_signal_of_importance, consider cough window size
  '''
  # as in https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html#librosa.effects.time_stretch
  
  signal = librosa.effects.time_stretch(y=signal, rate=rate)
  
  return signal


def time_shift(spect):
  '''
  Input:
    Spectrogram to be augmented
  Output:
    Spectrogram cut into two pieces along time dimension. Then second part is placed before the first
  '''
  spect_length = spect.shape[1]
  idx = np.random.randint(int(spect_length*0.4), int(spect_length*0.6))
  spect_ = np.hstack([spect[:,idx:], spect[:,:idx]])

  return spect_


def add_noise(signal):
  '''
  Input:
    sound signal; time series vector, standardized
  Output:
    sound signal + gaussian noise
  '''
  std = 0.025 * np.max(signal)
  noise_vec = np.random.randn(signal.shape[0])*std
  return signal + noise_vec


def denoise_spectrogram(spect, threshold=1, filter_size = (2,2)):
  """
  input:
    spectrogram, matrix
  output:
    denoised spectrogram, binary matrix as in bird singing paper
  """

  # map to [0,1]
  minVal = np.min(spect)
  maxVal = np.max(spect)
  spect = (spect - minVal)/(maxVal - minVal)

  # convert to binary
  row_medians = np.tile(np.median(spect, axis=1, keepdims=True), (1, spect.shape[1]))
  col_medians = np.tile(np.median(spect, axis=0, keepdims=True), (spect.shape[0], 1))
  spect_ = (spect > threshold * row_medians).astype('int') * (spect > threshold * col_medians).astype('int')
  
  # apply erosion + dilation
  structure_filter = np.ones(filter_size)
  spect_ = binary_erosion(spect_, structure=structure_filter)
  spect_ = binary_dilation(spect_, structure=structure_filter)

  return spect_



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
		  window = 0.16,
		  do_denoise=False,
      augment_data=False):
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

                if augment_data and is_training:

                  # pitch shift
                  timeSignal_pitch_shift = pitch_shift(timeSignal, sr=sample_rate)
                  timeSignal_pitch_shift = extract_Signal_Of_Importance(timeSignal_pitch_shift, window, sample_rate)
                  timeSignal_pitch_shift = standardize(timeSignal_pitch_shift)
                  mfcc_pitch_shift = librosa.feature.melspectrogram(y=timeSignal_pitch_shift, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)

                  # # stretch signal
                  # timeSignal_stretched = time_stretch(timeSignal, rate=1.2)
                  # timeSignal_stretched = extract_Signal_Of_Importance(timeSignal_stretched, window, sample_rate)
                  # timeSignal_stretched = standardize(timeSignal_stretched)
                  # mfcc_stretched = librosa.feature.melspectrogram(y=timeSignal_stretched, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)
                  
                  # add noise
                  timeSignal_addnoise = add_noise(timeSignal)
                  timeSignal_addnoise = extract_Signal_Of_Importance(timeSignal_addnoise, window, sample_rate)
                  timeSignal_addnoise = standardize(timeSignal_addnoise)
                  mfcc_addnoise = librosa.feature.melspectrogram(y=timeSignal_addnoise, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)
                  

                #fit_scale(timeSignal)
                timeSignal = extract_Signal_Of_Importance(timeSignal, window, sample_rate)

                timeSignal = standardize(timeSignal)
                mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)
                
                if augment_data and is_training:
                  # time shift
                  mfcc_time_shift = time_shift(mfcc)

                if do_denoise:
                  mfcc = denoise_spectrogram(mfcc)
                  mfcc_pitch_shift = denoise_spectrogram(mfcc_pitch_shift)
                  # mfcc_stretched = denoise_spectrogram(mfcc_stretched)
                  mfcc_addnoise = denoise_spectrogram(mfcc_addnoise)
                  mfcc_time_shift = denoise_spectrogram(mfcc_time_shift)


                batch_features.append(mfcc)

                if augment_data and is_training:
                  batch_features.append(mfcc_pitch_shift)
                  # batch_features.append(mfcc_stretched)
                  batch_features.append(mfcc_addnoise)
                  batch_features.append(mfcc_time_shift)

	#batch_features = np.asarray(batch_features).reshape(len(files),frames,bands)
	return np.array(batch_features)



















