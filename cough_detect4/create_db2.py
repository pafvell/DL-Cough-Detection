
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
from sklearn.model_selection import train_test_split



#loading config file
parser = argparse.ArgumentParser()
parser.add_argument('-config', 
                     type=str,
                     default='config.json',
                     help='store a json file with all the necessary parameters')
args = parser.parse_args()

#loading configuration
with open(args.config) as json_data_file:
           config = json.load(json_data_file)
config_db = config["dataset"]



def get_imgs(	split_id=config_db["split_id"],
		db_root_dir = config_db["DB_ROOT_DIR"],
    listOfParticipantsInTestset=config_db["test"],
    listOfParticipantsInValidationset=config_db["validation"],
		listOfAllowedSources=config_db["allowedSources"]
	    ):
       '''
	possible experiment splits:
	1)	standard test/training a la Felipe:  split files into test- and training-set by excluding all for the testset selected persons (stored in config.json /test)
	2)	standard validation a la Felipe: randomly split off a validation set out of the training_set and train on the rest
	3)	standard test/training a la Maurice: like 1 but  only consider certain types of microphones (defined in config.json /allowedSources)
	4)	standard validation a la Maurice: like 2 but  only consider certain types of microphones (defined in config.json /allowedSources)
	5)	standard test/training a la Kevin: like 3 but instead of splitting by selected persons only split randomly 80:20 

       '''

       list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', \
                               '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
                               '04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

       ##
       # READING DATA
       #
       #

       print ('use data from root path %s'%db_root_dir)
       coughAll = find_files(db_root_dir + "/04_Coughing", "wav", recursively=True)
       assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'
       coughAll = remove_broken_files(db_root_dir, list_of_broken_files, coughAll)
       other = find_files(db_root_dir + "/05_Other Control Sounds", "wav", recursively=True)

       #print( 'CoughAll: %d'%len(coughAll))
       #print( 'Other: %d'%len(other))
       #print( 'Total: %d'%(len(coughAll)+len(other)))

       #3+4) additional choosable tests: only consider certain types of microphones 
       if split_id > 2: 
                coughAll = [c for c in coughAll for allowedMic in listOfAllowedSources if allowedMic in get_device(c)]
                other	 = [c for c in other for allowedMic  in listOfAllowedSources if allowedMic in get_device(c)]

       #5) additional choosable tests: when only considering certain types of microphones 
       #if split_id==5:
       #         trainListOther, testListOther = train_test_split(other, test_size=0.20, random_state=42) 
       #         trainListCough, testListCough = train_test_split(coughAll, test_size=0.20, random_state=42) 
       #         return testListCough, testListOther, trainListCough, trainListOther

       ##
       # Make Sets
       #
       #

       testListOther, testListCough = [], []
       trainListCough = list(coughAll)
       trainListOther = list(other)

       #1) split files into test- and training-set by excluding all for the testset selected persons
       for name in coughAll:
                for nameToExclude in listOfParticipantsInTestset:
                        if nameToExclude in name:
                              testListCough.append(name)
                              trainListCough.remove(name)

       for name in other:
                for nameToExclude in listOfParticipantsInTestset:
                        if nameToExclude in name:
                              testListOther.append(name)
                              trainListOther.remove(name)

       #2) randomly split off a validation set out of the training_set
       if split_id%2==0:
                #trainListOther, testListOther = train_test_split(trainListOther, test_size=0.10, random_state=42) 
                #trainListCough, testListCough = train_test_split(trainListCough, test_size=0.10, random_state=42) 

                testListOther, testListCough = [], []
                coughAll = list(trainListCough)
                other = list(trainListOther) 

                for name in coughAll:
                        for nameToExclude in listOfParticipantsInValidationset:
                              if nameToExclude in name:
                                     testListCough.append(name)
                                     trainListCough.remove(name)

                for name in other:
                        for nameToExclude in listOfParticipantsInValidationset:
                              if nameToExclude in name:
                                     testListOther.append(name)
                                     trainListOther.remove(name)

       return testListCough, testListOther, trainListCough, trainListOther


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


def preprocess(	sound_file,
                bands,
                hop_length,
                window,
                ):

                try:
                       time_signal, sample_rate = librosa.load(sound_file, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e

                time_signal = extract_Signal_Of_Importance(time_signal, window, sample_rate)
                time_signal = standardize(time_signal)
                mfcc = librosa.feature.melspectrogram(y=time_signal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)
                return mfcc

            
def create_dataset(files1, 
        files0, 
        db_name, 
        hop_length=config_db["HOP"],
        bands = config_db["BAND"],
        window = config_db["WINDOW"],
        db_full_path=config["ROOT_DIR"],
        version=config["DB_version"]
        ):
        """
	     load, preprocess, normalize a sample
	     input: a list of strings
	     output: the processed features from each sample path in the input
        """

        print ('save %s samples'%db_name)
        db_filename = os.path.join(db_full_path, db_name + version + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(db_filename)

        def store_example(files, label): 

            for f in tqdm(files):
                mfcc = preprocess(f, bands=bands, hop_length=hop_length, window=window)
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

            
def test_shape (files1, 
                sound_id=0, #list id of the sample that should be displayed
                hop_length=config_db["HOP"],
                bands = config_db["BAND"],
                window = config_db["WINDOW"],
                ):

                import matplotlib.pyplot as plt
                import librosa.display

                f = files1[sound_id]
                mfcc = preprocess(f, bands=bands, hop_length=hop_length, window=window)
                
                print ('mfcc shape: '+str(mfcc.shape))
                print ('mfcc max: '+str(np.max(mfcc)))

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfcc, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout()
                plt.show()


def print_stats(test_coughs, test_other, train_coughs, train_other, name=''):
                   print()
                   print('------------------------------------------------------------------')
                   print('PARTITION: '+str(name))
                   print('nr of samples coughing (test): %d' % len(test_coughs))
                   print('nr of samples NOT coughing (test): %d' % len(test_other))
                   print('nr of samples coughing (train): %d' % len(train_coughs))
                   print('nr of samples NOT coughing (train): %d' % len(train_other))
                   t1 = len(test_coughs) + len(test_other)
                   t2 = len(train_coughs) + len(train_other)
                   print('total nr of samples: (train) %d + (test) %d = (total) %d' % (t2, t1, t1+t2))
                   print('------------------------------------------------------------------')
                   print()

    
def main(unused_args):
       tf.set_random_seed(0)

       # Store data to TFRECORDS
       testListCough, testListOther, trainListCough, trainListOther = get_imgs()
       name = config_db["split_id"]
       if config_db["CREATE_DB"]:
                   create_dataset(trainListCough, trainListOther, 'train_'+str(name))
                   create_dataset(testListCough, testListOther, 'test_'+str(name))
                   print_stats(testListCough, testListOther, trainListCough, trainListOther, name)
       else:
                   print_stats(testListCough, testListOther, trainListCough, trainListOther, name)
                   test_shape(trainListCough)
                   test_shape(trainListOther)



if __name__ == '__main__':
       tf.app.run()    


