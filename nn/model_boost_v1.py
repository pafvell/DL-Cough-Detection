#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope


"""
	larger conv net - Train top + bottom + keep the rest at random 
"""


def classify(inputs, 
             num_estimator,
	     num_classes,
             dropout_keep_prob=0.5,
             middle_size=4,
	     scope=None,
	     reuse=None,
             is_training=True 
	):
        """
	 model used to make predictions
	 input: x -> shape=[None,bands,frames,num_channels]
	 output: logits -> shape=[None,num_labels]
        """
        with slim.arg_scope(simple_arg_scope()): 
        	#with slim.arg_scope(batchnorm_arg_scope()): 
                      with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
                             with tf.variable_scope(scope, 'model_v1', [inputs], reuse=reuse) as scope:

                                      net = tf.expand_dims(inputs, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension

                                      with tf.variable_scope('bottom'):
                                                net = slim.conv2d(net, 64, [5, 3], rate=2, scope='convB1')
                                                net = slim.max_pool2d(net, [2, 1], scope='poolB1')

				      #random block
                                      with tf.variable_scope('middle'):
                                                for i in range(middle_size):
                                                	net = slim.conv2d(net, 64, [3, 3], scope='convM', reuse=i>0)
                                                net = slim.max_pool2d(net, [2, 1], scope='poolM')

				      # Use conv2d instead of fully_connected layers.
                                      with tf.variable_scope('top'):
                                                net = slim.flatten(net)
                                                net = slim.fully_connected(net, 128, scope='fc1')
                                                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                                                logits = slim.fully_connected(net, num_classes, scope='fc2', activation_fn=None)
                                                gamma = tf.Variable(1./(num_estimator), name='gamma')

                                      return logits, gamma


def build_model(x, 
		y,
	        num_classes=2,
		num_estimator=10,
                is_training=True,
		reuse=None
		):
        """
	 handle model. calculate the loss and the prediction for some input x and the corresponding labels y
	 input: x shape=[None,bands,frames,num_channels], y shape=[None]
	 output: loss shape=(1), prediction shape=[None]

	CAUTION! controller.py uses a function whith this name and arguments.

	here we do boosting without additive training

        """
        #preprocess
        y = slim.one_hot_encoding(y, num_classes)

        #model	
        logits = 0 
        for i in range(num_estimator):
                predictions, gamma = classify(x, num_estimator=num_estimator, num_classes=num_classes, is_training=is_training, reuse=reuse, scope='c%d'%i)
                zeta = 2 / (i+1) * gamma
                logits = (1-zeta) * logits + zeta * predictions
    

        #results
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y, label_smoothing=0.05)) 
        predictions = tf.argmax(slim.softmax(logits),1)

        return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = ['bottom', 'top'] #bottom + top are trainable



