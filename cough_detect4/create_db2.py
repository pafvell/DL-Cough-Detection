
import os.path
import fnmatch
import os
import datetime as dt
import tensorflow as tf
from tqdm import tqdm
import argparse
from collections import defaultdict
import json

import numpy as np
import os, sys, math, shutil, time, threading
import librosa

from utils import *


#loading configuration
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

ROOT_DIR = config["ROOT_DIR"]
config_db = config["create_db2"]
DB_ROOT_DIR = config_db["DB_ROOT_DIR"]

HOP=config_db["HOP"] #61 #56 #224,#112,#56,
WINDOW=config_db["WINDOW"]
BAND=config_db["BAND"]
VERSION=config["DB_version"]


CREATE_DB = config_db["CREATE_DB"]



###################################################################################################################################################################

#Data Augmentation Parameters
DO_DATA_AUGMENTATION = config_db["DO_DATA_AUGMENTATION"]
DATA_AUGMENT_METHOD = config_db["DATA_AUGMENT_METHOD"]
NOISE_STDEV = config_db["NOISE_STDEV"]
CREATE_N_SAMPLES = config_db["CREATE_N_SAMPLES"]

AUGM_LIST = config_db["AUGM_LIST"]


'''
def add_noise(signal, sigma=NOISE_STDEV):
        """
        Input:
        sound signal; time series vector, standardized
        Output:
        sound signal + gaussian noise
        """
        std = sigma * np.max(signal)
        noise_mat = np.random.randn(signal.shape[0])*std
        return signal + noise_mat


def pitch_shift(signal, sample_rate, n_steps):

        # as in https://librosa.github.io/librosa/generated/librosa.effects.pitch_shift.html#librosa.effects.pitch_shift

        return librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=n_steps)


def pitch_shift2(signal, sample_rate, bins_per_octave=24, pitch_pm=4):

        pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)   
        y_pitch = librosa.effects.pitch_shift(signal.astype('float64'), 
                                              sample_rate, 
                                              n_steps=pitch_change, 
                                              bins_per_octave=bins_per_octave)
        return y_pitch


def time_stretch(signal, sample_rate, window_size, stretch_factor=1.2):

        # as in https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html#librosa.effects.time_stretch

        signal = librosa.effects.time_stretch(y=signal, rate=stretch_factor)

        return extract_Signal_Of_Importance(signal=signal, window=window_size, sample_rate=sample_rate)



def apply_augment(signal, sample_rate, window_size, n_samples, method=DATA_AUGMENT_METHOD):

        #TODO
        #https://www.kaggle.com/CVxTz/audio-data-augmentation
        #https://www.kaggle.com/huseinzol05/sound-augmentation-librosa

        if method not in AUGM_LIST:
          raise NotImplementedError("augmentation method \"%s\" has not been implemented yet"%method)

        else:

          if method == None:
            return signal

          elif method == "add_noise":
            return add_noise(signal=signal)

          elif method == "pitch_shift":
            n_steps = config_db["PITCH_SHIFT"][n_samples]
            return pitch_shift(signal=signal, sample_rate=sample_rate, n_steps = n_steps)

          elif method == "pitch_shift2":
            return pitch_shift2(signal=signal, sample_rate=sample_rate)

          elif method == "time_stretch":
            return time_stretch(signal=signal, sample_rate=sample_rate, window_size=window_size)
'''

###################################################################################################################################################################


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


def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

def _string_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


            
def create_dataset(files1, 
                   files0, 
                   db_name, 
                   db_full_path=ROOT_DIR,
                   hop_length=HOP,
		   bands = BAND,
		   window = WINDOW, #0.16
                   do_augmentation=False,
                   create_n_samples=CREATE_N_SAMPLES):
        """
	     load, preprocess, normalize a sample
	     input: a list of strings
	     output: the processed features from each sample path in the input
        """

        print ('save %s samples'%db_name)
        db_filename = os.path.join(db_full_path, db_name + VERSION + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(db_filename)


        #if do_augmentation:
        #  print("data augmenting: %s"%do_augmentation)
        #  print("number of samples %d"%create_n_samples)


        def store_example(files, label): #, create_n_samples=CREATE_N_SAMPLES):

            for f in tqdm(files):
                try:
                       time_signal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e

                time_signal = extract_Signal_Of_Importance(time_signal, window, sample_rate)
                time_signal = standardize(time_signal)


                #if not do_augmentation:
                #  create_n_samples=0

                #for j in range(create_n_samples):

                      #if j>=1:
                      #   time_signal = apply_augment(time_signal, sample_rate=sample_rate, window_size=window, n_samples=(j-1))

                mfcc = librosa.feature.melspectrogram(y=time_signal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)

                example = tf.train.Example(features=tf.train.Features(feature={
                                                                        'height': _int64_feature(bands),
                                                                        'width': _int64_feature(mfcc.shape[1]),
                                                                        'depth': _int64_feature(1),
                                                                        'data': _floats_feature(mfcc),
                                                                        'label': _int64_feature(label),
                                                                        }))
                writer.write(example.SerializeToString())
                
        store_example(files1, 1)
        store_example(files0, 0)
        writer.close()

            
def test_shape(files1, 
               hop_length=HOP,
	       bands = BAND,
	       window = WINDOW):

            import matplotlib.pyplot as plt
            import librosa.display

            for f in files1:
                try:
                       timeSignal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e

                timeSignal = extract_Signal_Of_Importance(timeSignal, window, sample_rate)

                timeSignal = standardize(timeSignal)

                mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)
                #mfcc = librosa.feature.delta(mfcc)

                size_cub=mfcc.shape[1]
                print ('mfcc shape: '+str(mfcc.shape))
                print ('mfcc max: '+str(np.max(mfcc)))

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfcc, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout()
                plt.show()
                break



    
def main(unused_args):
       tf.set_random_seed(0)
       list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', \
                               '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
                               '04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

       ##
       # READING DATA
       #
       #

       print ('use data from root path %s'%DB_ROOT_DIR)
       coughAll = find_files(DB_ROOT_DIR + "/04_Coughing", "wav", recursively=True)
       assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'
       trainListCough = coughAll = remove_broken_files(DB_ROOT_DIR, list_of_broken_files, coughAll)
       trainListOther = other = find_files(DB_ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

       ##
       # Make Validation Set
       #
       #
       '''
       listOfParticipantsInValidationset=config["test"]
       validationListOther, validationListCough = [], []
       for name in coughAll:
           for nameToExclude in listOfParticipantsInValidationset:
              if nameToExclude in name:
                  validationListCough.append(name)
                  trainListCough.remove(name)
       for name in other:
           for nameToExclude in listOfParticipantsInValidationset:
              if nameToExclude in name:
                  validationListOther.append(name)
                  trainListOther.remove(name)
       coughAll = trainListCough
       other = trainListOther

       create_dataset(validationListCough, validationListOther, 'test')

       t1 = len(validationListCough)
       t2 = len(validationListOther)
       print('total nr of test samples: (cough) %d + (other) %d = %d ' % (t1, t2, t1+t2))
       '''

       ##
       # Make Sets
       #
       #

       if CREATE_DB:
              #for i in range(5):
                   #listOfParticipantsInTrainset=config["cv_partition%d"%i]
                   listOfParticipantsInTrainset=config["test"]
                   testListOther, testListCough = [], []
                   trainListCough = list(coughAll)
                   trainListOther = list(other)

                   #split files into test- and training-set
                   for name in coughAll:
                       for nameToExclude in listOfParticipantsInTrainset:
                           if nameToExclude in name:
                              testListCough.append(name)
                              trainListCough.remove(name)

                   for name in other:
                       for nameToExclude in listOfParticipantsInTrainset:
                           if nameToExclude in name:
                              testListOther.append(name)
                              trainListOther.remove(name)


                   print()
                   print('------------------------------------------------------------------')
                   print('PARTITION %d'%i)
                   print('nr of samples coughing (test): %d' % len(testListCough))
                   print('nr of samples NOT coughing (test): %d' % len(testListOther))
                   print('nr of samples coughing (train): %d' % len(trainListCough))
                   print('nr of samples NOT coughing (train): %d' % len(trainListOther))
                   t1 = len(testListCough) + len(testListOther)
                   t2 = len(trainListCough) + len(trainListOther)
                   print('total nr of samples: (train) %d + (test) %d = (total) %d' % (t2, t1, t1+t2))
                   print()

                   # START STORING DATA TO TFRECORDS
                   create_dataset(trainListCough, trainListOther, 'train_%d'%i, do_augmentation=DO_DATA_AUGMENTATION)
                   create_dataset(testListCough, testListOther, 'test_%d'%i)


       
       else:
              test_shape(trainListCough)



if __name__ == '__main__':
       tf.app.run()    


