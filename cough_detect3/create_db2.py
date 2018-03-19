
import os.path
import fnmatch
import os
import datetime as dt
import tensorflow as tf
from tqdm import tqdm
import argparse
from collections import defaultdict

import numpy as np
import os, sys, math, shutil, time, threading
import librosa

from utils import *




ROOT_DIR = './Audio_Data'

HOP=56 #224,#112,#56,
VERSION='2'


def standardize(timeSignal):

	 #TODO
         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)

         #maxValue = 1.7
         #minValue = -1.8

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


def add_noise(signal, sigma=0.05):
        '''
        Input:
        sound signal; time series vector, standardized
        Output:
        sound signal + gaussian noise
        '''
        std = sigma * np.max(signal)
        noise_mat = np.random.randn(signal.shape[0])*std
        return signal + noise_mat


def apply_augment(signal, method=None):

        if method == None:
          return signal

        elif method == "add_noise":
          return add_noise(signal)

        else:
          raise NotImplementedError("augmentation method \"%s\" has not been implemented yet"%method)


def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

def _string_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


            
def create_dataset(files1, files0, db_name, 
                  db_full_path='./Audio_Data',
                  hop_length=HOP,
		              bands = 16,
		              window = 0.16,
                  do_denoise=False,
                  data_augment=False,
                  augment_method=None,
                  create_n_samples=100):
        """
	load, preprocess, normalize a sample
	input: a list of strings
	output: the processed features from each sample path in the input
        """

        print ('save %s samples'%db_name)
        db_filename = os.path.join(db_full_path, db_name + VERSION + '_%d.tfrecords'%HOP)
        print(db_filename)
        writer = tf.python_io.TFRecordWriter(db_filename)
        
        create_n_samples = (1 - data_augment) + data_augment * create_n_samples

        if data_augment:
          print("data augmenting: %s"%data_augment)
          print("number of samples %d"%create_n_samples)
          print("method: %s"%augment_method)
        else:
          augment_method = None


        def store_example(files, label):

            for f in tqdm(files):

                try:
                       timeSignal_raw, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e

                timeSignal_raw = extract_Signal_Of_Importance(timeSignal_raw, window, sample_rate)

                for _ in range(create_n_samples):

                  timeSignal = apply_augment(signal=timeSignal_raw, method=augment_method)

                  #fit_scale(timeSignal)
                  timeSignal = standardize(timeSignal)

                  mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length, n_fft=512)

                  size_cub=mfcc.shape[1]
                  
                  example = tf.train.Example(features=tf.train.Features(feature={
                                                                          'height': _int64_feature(size_cub),
                                                                          'width': _int64_feature(size_cub),
                                                                          'depth': _int64_feature(1),
                                                                          'data': _floats_feature(mfcc),
                                                                          'label': _int64_feature(label),
                                                                          }))

                  writer.write(example.SerializeToString())
            

        store_example(files1, 1)
        store_example(files0, 0)
        writer.close()


    
def main(unused_args):

       listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] #participants used in the test-set

       list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
                               '04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

       ##
       # READING COUGH DATA
       #
       #

       print ('use data from root path %s'%ROOT_DIR)

       coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
       assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'

       #remove broken files
       for broken_file in list_of_broken_files:
           broken_file = os.path.join(ROOT_DIR, broken_file)
           if broken_file in coughAll:
                 print ( 'file ignored: %s'%broken_file )
                 coughAll.remove(broken_file)

       #split cough files into test- and training-set
       testListCough = []
       trainListCough = coughAll
       for name in coughAll:
           for nameToExclude in listOfParticipantsToExcludeInTrainset:
              if nameToExclude in name:
                  testListCough.append(name)
                  trainListCough.remove(name)

       print('nr of samples coughing: %d' % len(testListCough))

       ##
       # READING OTHER DATA
       #
       #

       other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

       testListOther = []
       trainListOther = other
       for name in other:
           for nameToExclude in listOfParticipantsToExcludeInTrainset:
              if nameToExclude in name:
                  testListOther.append(name)
                  trainListOther.remove(name)

       print('nr of samples NOT coughing: %d' % len(testListOther))


       ##
       # START STORING DATA TO TFRECORDS
       #
       #

       tf.set_random_seed(0)

       create_dataset(trainListCough, trainListOther, 'train', data_augment=True, augment_method="add_noise")
       create_dataset(testListCough, testListOther, 'test')




if __name__ == '__main__':
       tf.app.run()    


