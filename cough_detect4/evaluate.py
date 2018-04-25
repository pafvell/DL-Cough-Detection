#!/usr/bin/python
#Authors: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading
import importlib
import json
import argparse
from utils import *
from create_db2 import get_imgs, preprocess
from sklearn.metrics import accuracy_score, confusion_matrix

tf.set_random_seed(0)

#******************************************************************************************************************

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
control_config = config["controller"] # reads the config for the controller file
config_db = config["dataset"] 
config_train = control_config["training_parameter"]


#******************************************************************************************************************

def classification_report(y, preds):
	cm = confusion_matrix(y,preds)
	print ('Confusion Matrix: \n', cm)

	total = sum(sum(cm))
	acc = accuracy_score(y,preds)
	print ('accuracy: ', acc)
	sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
	print ('sensitivity: ', sensitivity)
	specificity = cm[1,1]/(cm[1,0]+cm[1,1])
	print ('specificity:', specificity)

	return acc, sensitivity, specificity
 

def test(
		checkpoint_dir=config_train["checkpoint_dir"],
        hop_length=config_db["HOP"],
		bands = config_db["BAND"],
		window = config_db["WINDOW"],
		batch_size=config_train["batch_size"],
		split_id=config_db["split_id"],
		participants=config_db["test"],
		sources=config_db["allowedSources"],
		db_root_dir=config_db["DB_ROOT_DIR"]
        ):
	print ('read checkpoints: %s'%checkpoint_dir)
	checkpoint_dir = checkpoint_dir+'/cv%d'%split_id


	#TODO restore any checkpoint
	latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)	
	if not latest_ckpt:
		raise IOError('Invalid checkpoint path: %s! It is not possible to evaluate the model.'%checkpoint_dir)
	print ('restore checkpoint:%s'%latest_ckpt)



	# Create the session that we'll use to execute the model
	sess_config = tf.ConfigProto(
	    log_device_placement=False,
	    allow_soft_placement = True
	)
	sess = tf.Session(config=sess_config)

	saver = tf.train.import_meta_graph(latest_ckpt+'.meta')
	saver.restore(sess, latest_ckpt)
	
	graph = tf.get_default_graph()
	graph.finalize()

	# Get the input and output operations
	input_op = graph.get_operation_by_name('Input') 
	input_tensor = input_op.outputs[0]
	output_op = graph.get_operation_by_name('Prediction') 
	output_tensor = output_op.outputs[0]

    #get data and predict        
	X_cough, X_other, _, _ = get_imgs(split_id=split_id,	
					  db_root_dir = db_root_dir,
        			  listOfParticipantsInTestset=participants,
					  listOfAllowedSources=sources
				  	)

	print('nr of samples coughing (test): %d' % len(X_cough))
	print('nr of samples NOT coughing (test): %d' % len(X_other))

	X = X_cough + X_other
	y = [1]*len(X_cough)+[0]*len(X_other)


	predictions=[]
	for x in X:  #make_batches(X, batch_size): 
		x = preprocess(x, bands=bands, hop_length=hop_length, window=window)
		x = np.expand_dims(x, 0)
		predictions.append(sess.run(output_tensor, {input_tensor: x}))
	

	print ()
	print ('********************************************************************************')
	print ('Evaluate over Everything:')
	acc, sen, spe = classification_report(predictions,y)


	X = list(zip(X, y, predictions))
	sources = ["studio", "iphone", "samsung", "htc", "tablet", "audio track"]
	for mic in sources:
		Xlittle = [x for x in X if mic in get_device(x[0])]
		if len(Xlittle) > 0:
			Xlittle, ylittle, Plittle = zip(*Xlittle)
			print ()
			print ('********************************************************************************')
			print ('Evaluate '+mic)
			classification_report(Plittle, ylittle)

	return acc, sen, spe



test()









