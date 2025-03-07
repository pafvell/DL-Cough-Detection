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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score



#******************************************************************************************************************


tf.set_random_seed(0)


#loading config file
parser = argparse.ArgumentParser(description='Script to evaluate a given Model')
parser.add_argument('--config', 
                     type=str,
                     default='./config.json',
                     help='path to the file config.json which stores all the necessary parameters')
parser.add_argument('--ckpt_dir', 
                     type=str,
                     default='',
                     help='path to the file config.json which stores all the necessary parameters')
args = parser.parse_args()

#loading configuration
with open(args.config) as json_data_file:
           config = json.load(json_data_file)
control_config = config["controller"] # reads the config for the controller file
config_db = config["dataset"] 
config_train = control_config["training_parameter"]

if args.ckpt_dir:
	CKPT_DIR = args.ckpt_dir
else:
	CKPT_DIR = config_train["checkpoint_dir"]

if "NFFT" in config_db:
	NFFT = config_db["NFFT"]
else:
	NFFT = 2048		

#******************************************************************************************************************

def classification_report(y_true, y_pred, sanity_check=False, print_report=True):
	cm = confusion_matrix(y_true,y_pred)
	total = sum(sum(cm))
	acc = accuracy_score(y_true,y_pred)
	specificity = cm[0,0]/(cm[0,0]+cm[0,1])  
	sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
	precision = cm[1,1]/(cm[0,1]+cm[1,1]) 

	if print_report:
		print ('Confusion Matrix: \n', cm)
		print ('accuracy: ', acc)
		print ('sensitivity (recall): ', sensitivity)
		print ('specificity:', specificity)
		print ('precision: ', precision)

	if sanity_check:
		print ('(SANITY CHECK - our precision: %f vs sklearn precision: %f)'%(precision, precision_score(y_true, y_pred)))
		print ('(SANITY CHECK - our sensitivity: %f vs sklearn recall: %f)'%(sensitivity, recall_score(y_true, y_pred)))


	return acc, sensitivity, specificity, precision
 

def test(
	 	model_name=control_config["model"],
        	hop_length=config_db["HOP"],
		bands = config_db["BAND"],
		window = config_db["WINDOW"],
	 	size_cub=control_config["spec_size"],
		batch_size=config_train["batch_size"],
        	num_estimator=config_train["num_estimator"],
         	num_filter=config_train["num_filter"],
		split_id=config_db["split_id"],
		participants=config_db["test"],
		sources=config_db["allowedSources"],
		db_root_dir=config_db["DB_ROOT_DIR"],
		checkpoint_dir=CKPT_DIR,
                nfft = NFFT, 	
        ):



	print ('read checkpoints: %s'%checkpoint_dir)
	checkpoint_dir = checkpoint_dir+'/cv%d'%split_id

	print ('evaluate model:'+model_name)
	model = importlib.import_module(model_name) 

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
	
	input_tensor = tf.placeholder(tf.float32, shape=[bands,size_cub], name='Input')
	x = tf.expand_dims(input_tensor, 0) 
	_, output_tensor = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter, is_training=False)


	saver = tf.train.Saver()
	saver.restore(sess, latest_ckpt)		

    	#get data and predict        
	X_cough, X_other, _, _ = get_imgs(	split_id=config_db["split_id"],
						db_root_dir = config_db["DB_ROOT_DIR"],
    						listOfParticipantsInTestset=config_db["test"],
    						listOfParticipantsInValidationset=config_db["validation"],
						listOfAllowedSources=config_db["allowedSources"]
					)
        

	print('nr of samples coughing (test): %d' % len(X_cough))
	print('nr of samples NOT coughing (test): %d' % len(X_other))

	X = X_cough + X_other
	y = [1]*len(X_cough)+[0]*len(X_other)


	predictions=[]
	for x in X:  #make_batches(X, batch_size): 
		x = preprocess(x, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
		#x = np.expand_dims(x, 0)
		predictions.append(sess.run(output_tensor, {input_tensor: x}))
	

	print ()
	print ('********************************************************************************')
	print ('Evaluate over Everything:')
	acc, sen, spe, prec = classification_report(y, predictions)


	X = list(zip(X, y, predictions))
	sources = ["studio", "iphone", "samsung", "htc", "tablet", "audio track" ]
	for mic in sources:
		Xlittle = [x for x in X if mic in get_device(x[0])]
		if len(Xlittle) > 0:
			path, y_true, y_pred = zip(*Xlittle)
			print ()
			print ('********************************************************************************')
			print ('Evaluate '+mic)
			classification_report(y_true, y_pred)

	kinds = ["Close (cc)", "Distant (cd)", "01_Throat Clearing", "02_Laughing", "03_Speaking", "04_Spirometer"]

	for kind in kinds:
		Xlittle = [x for x in X if kind in x[0]]
		if len(Xlittle) > 0:
			path, y_true, y_pred = zip(*Xlittle)
			print ()
			print ('********************************************************************************')
			print ('Evaluate '+kind)
			cm = confusion_matrix(y_true,y_pred)
			acc = accuracy_score(y_true,y_pred)
			print ('Confusion Matrix: \n', cm)
			print ('accuracy: ', acc)

	return acc, sen, spe, prec



test()






