#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

from utils import softmax_cross_entropy_v2 as softmax_cross_entropy
#from tensorflow.losses import softmax_cross_entropy

"""
	RNN
"""
def preprocess_input(inputs):
            """
            split into 16 patches with shape [16,8]
            """
            inputs = tf.expand_dims(inputs, -1)
            print ('input shape: ', inputs.get_shape())
            inputs = tf.extract_image_patches(images=inputs, ksizes=[1, 16, 8, 1], strides=[1, 16, 8, 1], rates=[1, 1, 1, 1], padding='VALID')
            inputs = tf.squeeze(inputs, 1)
            print ('patches shape: ', inputs.get_shape())
            return inputs

def preprocess_input_mean(inputs):
            inputs = preprocess_input(inputs)
            shape = tf.shape(inputs)
            inputs = tf.reshape([shape[0], shape[1], 16,8])
            inputs = tf.reduce_mean(inputs, 2)
            return inputs



def RNN_unidir(inputs, 
	num_outputs,
	num_cells=1,
	scope=None,
	reuse=None, 
        is_training=True
	):
  with tf.variable_scope('rnn_multicell', [inputs],
      			reuse=reuse) as sc:
       with slim.arg_scope(batchnorm_arg_scope(is_training=is_training)):         
            inputs = preprocess_input(inputs)

            cell1 = tf.nn.rnn_cell.GRUCell(128)
            cell2 = tf.nn.rnn_cell.GRUCell(64)
            cell3 = tf.nn.rnn_cell.GRUCell(32)
            cell4 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple = True)
            cell4 = tf.contrib.rnn.AttentionCellWrapper(cell4, 16)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3, cell4])
            output, state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
            print ('rnn out shape: ', output.get_shape())
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            print ('transf. rnn out shape: ', last.get_shape())
            last = slim.fully_connected(last,256, 
					activation_fn=tf.nn.relu)

            return slim.fully_connected(last,2, activation_fn=None)


def RNN_bidir(inputs, 
	num_outputs,
	num_cells=1,
	scope=None,
	reuse=None, 
        is_training=True
	):
  with tf.variable_scope('rnn_multicell', [inputs],
      			reuse=reuse) as sc:
       with slim.arg_scope(batchnorm_arg_scope(is_training=is_training)):         
            inputs = preprocess_input(inputs)

            cellF1 = tf.nn.rnn_cell.GRUCell(128)
            cellF2 = tf.nn.rnn_cell.GRUCell(64)
            cellF3 = tf.nn.rnn_cell.GRUCell(32)
            cellF4 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple = True)
            cellF4 = tf.contrib.rnn.AttentionCellWrapper(cellF4, 16)

            cellB1 = tf.nn.rnn_cell.GRUCell(128)
            cellB2 = tf.nn.rnn_cell.GRUCell(64)
            cellB3 = tf.nn.rnn_cell.GRUCell(32)
            cellB4 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple = True)
            cellB4 = tf.contrib.rnn.AttentionCellWrapper(cellB4, 16)
            
            output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn ([cellF1, cellF2, cellF3, cellF4], [cellB1, cellB2, cellB3, cellB4], inputs, dtype = tf.float32)
            print ('rnn out: ', output.get_shape())

            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)

            print ('transf. rnn out shape: ', last.get_shape())
            last = slim.fully_connected(last,256, 
					activation_fn=tf.nn.relu)

            return slim.fully_connected(last,2, activation_fn=None)


def RNN_deepcough(inputs, 
	num_outputs,
	num_cells=1,
	scope=None,
	reuse=None, 
        is_training=True
	):
  with tf.variable_scope('rnn_multicell', [inputs],
      			reuse=reuse) as sc:
       with slim.arg_scope(batchnorm_arg_scope(is_training=is_training)):         
            #inputs = preprocess_input(inputs)

            cellF1 = tf.nn.rnn_cell.GRUCell(128)
            cellF2 = tf.nn.rnn_cell.GRUCell(64)

            cellB1 = tf.nn.rnn_cell.GRUCell(128)
            cellB2 = tf.nn.rnn_cell.GRUCell(64)
            
            output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn ([cellF1, cellF2], [cellB1, cellB2], inputs, dtype = tf.float32)
            print ('rnn bidir out: ', output.get_shape())

            cell3 = tf.nn.rnn_cell.GRUCell(32)
            cell4 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple = True)
            cell4 = tf.contrib.rnn.AttentionCellWrapper(cell4, 16)

            cell = tf.nn.rnn_cell.MultiRNNCell([cell3, cell4])
            output, _ = tf.nn.dynamic_rnn(cell, output, dtype = tf.float32)

            print ('rnn out shape: ', output.get_shape())
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            print ('transf. rnn out shape: ', last.get_shape())
            last = slim.fully_connected(last,256, 
					activation_fn=tf.nn.relu)

            return slim.fully_connected(last,2, activation_fn=None)


def build_model(x, 
		y,
	        num_classes=2,
                is_training=True,
                num_estimator=None,
                num_filter=None,
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
        print ('input: ', x.get_shape())
        #model
        logits = RNN_deepcough(x, num_outputs=num_classes, reuse=reuse, is_training=is_training)	

        #results
        loss = tf.reduce_mean(softmax_cross_entropy(logits = logits, onehot_labels = y)) 
        predictions = tf.argmax(slim.softmax(logits),1)

        return loss, predictions 	





#Parameters
TRAINABLE_SCOPES = None #all weights are trainable




