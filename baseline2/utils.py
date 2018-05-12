#Authors: Maurice Weber

import glob
import random
import os, fnmatch, sys

import numpy as np
import pandas as pd
import librosa

ROOT_DIR = '../../Audio_Data'


def find_files(root, fntype, recursively=False):

	fntype = '*.'+fntype

	if not recursively:
		return glob.glob(os.path.join(root, fntype))

	matches = []
	for dirname, subdirnames, filenames in os.walk(root):
		for filename in fnmatch.filter(filenames, fntype):
			matches.append(os.path.join(dirname, filename))
	
	return matches


def standardize(timeSignal):
         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)
         timeSignal = (timeSignal - minValue)/(maxValue - minValue) 
         return timeSignal


def extract_Signal_Of_Importance(f, window, do_standardize=True):
        """
        f: filename of the sound file to be loaded
        extract a window around the maximum of the signal
        input: 	signal
        window -> size of a window in seconds
        sample_rate 
        """

        signal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')

        window_size = int(window * sample_rate)

        start = max(0, np.argmax(np.abs(signal)) - (window_size // 2))
        end = min(np.size(signal), start + window_size)
        signal = signal[start:end]

        length = np.size(signal)
        assert length <= window_size, 'extracted signal is longer than the allowed window size'
        if length < window_size:
                #pad zeros to the signal if too short
                signal = np.concatenate((signal, np.zeros(window_size-length)))

        if do_standardize:        
                signal = standardize(signal)

        return signal, sample_rate


def remove_broken_files(root, list_of_broken_files, files):
       for broken_file in list_of_broken_files:
           broken_file = os.path.join(root, broken_file)
           if broken_file in files:
                 print ( 'file ignored: %s'%broken_file )
                 files.remove(broken_file)
       return files


def get_raw_device(filename):

	device = filename.split("/")[-1].split("_")[1].split("-")[0]
	return device


def get_device(filename):

	devices_dict = {
				'Rode': "studio",
				'SamsungS5': "samsung",
				'SamsunsgS5': "samsung",
				'audio track': "audio track",
				'htc': "htc",
				'iPhone': "iphone",
				'iphone': "iphone",
				'rode': "studio",
				's5': "samsung",
				's6': "samsung",
				'samsung': "samsung",
				'samsungS5': "samsung",
				'tablet': "tablet",
				'tablet2': "tablet"
				}

	try:
		device = devices_dict[filename.split("/")[-1].split("_")[1].split("-")[0]]
		return device
	except:
		raise KeyError("unknown device for file %s"%filename)
























