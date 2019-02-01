#Authors: Filipe Barata, Maurice Weber, Kevin Kipfer

import fnmatch
import glob
import os

import numpy as np

ROOT_DIR = '../../Audio_Data'



def standardize(timeSignal):
         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)
         timeSignal = (timeSignal - minValue)/(maxValue - minValue)
         return timeSignal


ROOT_DIR = '../../Audio_Data'


def find_files(root, fntype, recursively=False):
	fntype = '*.' + fntype

	if not recursively:
		return glob.glob(os.path.join(root, fntype))

	matches = []
	for dirname, subdirnames, filenames in os.walk(root):
		for filename in fnmatch.filter(filenames, fntype):
			matches.append(os.path.join(dirname, filename))

	return matches



def remove_broken_files(root, list_of_broken_files, files):
	for broken_file in list_of_broken_files:
		broken_file = os.path.join(root, broken_file)
		if broken_file in files:
			print ('file ignored: %s' % broken_file)
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
		raise KeyError("unknown device for file %s" % filename)











































