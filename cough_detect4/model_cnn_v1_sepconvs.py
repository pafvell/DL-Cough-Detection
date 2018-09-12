#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

from utils import softmax_cross_entropy_v2 as softmax_cross_entropy
#from tensorflow.losses import softmax_cross_entropy


"""
	small conv net - train all
"""



def classify(x, 
	     num_classes,
             dropout_keep_prob=0.5,
             weight_decay = None,
	     scope='model_v1',
	     reuse=None,
             is_training=True 
	):
	"""
	 model used to make predictions
	 input: x -> shape=[None,bands,frames,num_channels]
	 output: logits -> shape=[None,num_labels]
	"""
	with slim.arg_scope(simple_arg_scope(weight_decay=weight_decay)): 
		    	with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
                             with tf.variable_scope(scope, [x], reuse=reuse) as sc:

                                        net = tf.expand_dims(x, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension
                                        print ('model input shape: %s'%net.get_shape())

                                        # net = slim.conv2d(net, 16, [3, 9], scope='conv1')
                                        net = slim.separable_conv2d(net, 16, [3, 9], depth_multiplier=1, scope='conv1')
                                        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool1')
                                        net = slim.separable_conv2d(net, 16, [3, 5], depth_multiplier=1, scope='conv2')
                                        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
                                        net = slim.flatten(net)
                                        net = slim.fully_connected(net, 256, scope='fc1')
                                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                                        net = slim.fully_connected(net, 256, scope='fc2')
                                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout2')
                                        logits = slim.fully_connected(net, num_classes, scope='fc3', activation_fn=None)
                                        return logits


def build_model(x, 
		y,
	        num_classes=2,
                num_estimator=None,
                num_filter=None,
                is_training=True,
		reuse=None
		):
	"""
	 handle model. calculate the loss and the prediction for some input x and the corresponding labels y
	 input: x shape=[None,bands,frames,num_channels], y shape=[None]
	 output: loss shape=(1), prediction shape=[None]

	CAUTION! controller.py uses a function whith this name and arguments.
	"""
	#preprocess
	y = slim.one_hot_encoding(y, num_classes)

	#model
	logits = classify(x, num_classes=num_classes, is_training=is_training, reuse=reuse)	

	#results
	loss = tf.reduce_mean(softmax_cross_entropy(logits = logits, onehot_labels = y)) 
	predictions = tf.argmax(slim.softmax(logits),1)

	return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = None #all weights are trainable



