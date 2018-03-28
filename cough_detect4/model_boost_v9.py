#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

from utils import softmax_cross_entropy_v2 as softmax_cross_entropy
#from tensorflow.losses import softmax_cross_entropy



def classify(inputs, 
	     num_classes,
             dropout_keep_prob=0.5,
             weight_decay = 1e-3,
             num_filter=32,
	     scope=None,
	     reuse=None,
             is_training=True,
             route=3
	):
        """
	 model used to make predictions
	 input: x -> shape=[None,bands,frames,num_channels]
	 output: logits -> shape=[None,num_labels]
        """
        with slim.arg_scope(simple_arg_scope(weight_decay=weight_decay)): 
                      with slim.arg_scope([slim.batch_norm, slim.dropout],
		                	is_training=is_training):
                             with tf.variable_scope(scope, 'weak_learner', [inputs], reuse=reuse) as scope:

                                      net = tf.expand_dims(inputs, -1) #input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension

                                      with tf.variable_scope('stump'):
                                                net = slim.conv2d(net, num_filter, [1, 7], scope='conv1x7')
                                                net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool1')
                                                net = slim.conv2d(net, num_filter, [1, 5], scope='conv1x5')

                                      with tf.variable_scope('middle'):

                                                for i in range(route):
                                                      net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool%d'%(i+2))
                                                      net += slim.conv2d(net, num_filter, [3, 3], scope='conv3x3_%d'%(i+2))

                                                net = tf.reduce_max(net, 2) 

                                      with tf.variable_scope('top'):
                                                net = slim.flatten(net)
                                                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                                                logits = slim.fully_connected(net, num_classes, scope='fc2', activation_fn=None) 

                                      return logits


def loss_fkt(logits, y):
        return tf.reduce_mean(softmax_cross_entropy(logits = logits, onehot_labels = y, label_smoothing=0.05)) 



def build_model(x, 
		y,
	        num_classes=2,
		num_estimator=64,
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
        with tf.variable_scope('model_v1'):
             logits = classify(x, num_classes=num_classes, is_training=is_training, reuse=reuse, scope='H0', route=0)
             loss = loss_fkt(logits, y)

             for i in range(1,num_estimator):
                logits += classify(x, num_classes=num_classes, is_training=is_training, reuse=reuse, scope='H%d'%(i+1))
                loss += loss_fkt(logits, y)

   
        #results
        predictions = tf.argmax(slim.softmax(logits),1)
        return loss, predictions 	




#Parameters
TRAINABLE_SCOPES = None #everything is trainable



