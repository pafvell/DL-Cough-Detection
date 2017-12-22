#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

"""
	resnet 18 like in [https://arxiv.org/pdf/1512.03385.pdf] - train all
"""

def classify(x, 
	     num_classes,
             num_blocks=4,
	     scope='model_v1',
	     reuse=None,
             is_training=True 
	):
	"""
	 model used to make predictions
	 input: x -> shape=[None,bands,frames,num_channels]
	 output: logits -> shape=[None,num_labels]
	"""
	with slim.arg_scope(simple_arg_scope()): 
        	with slim.arg_scope(batchnorm_arg_scope()): 
		    	with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
  				with slim.arg_scope([slim.conv2d], weights_initializer= slim.variance_scaling_initializer(seed=seed)):
		                     with tf.variable_scope(scope, [x], reuse=reuse) as sc:
						net = tf.expand_dims(x, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension
						for i in range(num_blocks):
							with tf.variable_scope("block%d"%i):
								depth = 2**(i+6)
								residual = slim.conv2d(x, depth, [3,3], scope='convA%d'%i)
								residual = slim.conv2d(residual, depth, [3,3], stride=stride, activation_fn=None, scope='convB%d'%i)
								x = tf.nn.relu(x + residual)
                                                                if i < num_blocks-1:
									x = slim.max_pool2d(x, [1, 2], stride=2, scope='pool%d'%i)

						x = tf.reduce_mean(x, [1, 2], name='global_pool')
						#x = slim.flatten(x)
						logits = slim.fully_connected(net, num_classes, scope='fc', activation_fn=None)
						return logits


def build_model(x, 
		y,
	        num_classes=2,
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
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y)) 
	predictions = tf.argmax(slim.softmax(logits),1)

	return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = None #all weights are trainable



