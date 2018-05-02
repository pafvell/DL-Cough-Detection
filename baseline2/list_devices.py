import numpy as np
import json
from utils import *

'''
Script prints out all names of different devices stored in the data folders
'''

#loading configuration
with open('config.json') as json_data_file:
	config = json.load(json_data_file)


# read params from config file
DB_ROOT_DIR = config["general"]["DB_ROOT_DIR"]


list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', \
							'04_Coughing/Distant (cd)/p17_tablet-108.wav', \
							'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

coughAll = find_files(DB_ROOT_DIR + "/04_Coughing", "wav", recursively=True)
assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'

coughAll = remove_broken_files(DB_ROOT_DIR, list_of_broken_files, coughAll)
other = find_files(DB_ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)
trainListCough = list(coughAll)
trainListOther = list(other)

all_files = trainListOther + trainListCough

print("toal number of files: %i"%(len(all_files)))


devices = []
for f in all_files:
	devices.append(get_raw_device(f))

print(np.unique(devices))