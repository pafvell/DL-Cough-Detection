#Author: Kevin Kipfer

import tensorflow as tf
import os, fnmatch


def simple_arg_scope(weight_decay=0.0005, 
	seed=0,
	activation_fn=tf.nn.relu ):
	"""Defines a simple arg scope.
	relu, xavier, 0 bias, conv2d padding Same, weight decay
	Args:
	weight_decay: The l2 regularization coefficient.
	Returns:
	An arg_scope.
	"""
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
		weights_initializer= tf.contrib.layers.xavier_initializer(seed=seed),# this is actually not needed
		activation_fn=activation_fn,
		weights_regularizer=slim.l2_regularizer(weight_decay),
		biases_initializer=tf.zeros_initializer()):
		with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
			return arg_sc


def simple_arg_scope(weight_decay=0.0005,
	seed=0,
	activation_fbn=tf.nn.relu):

	return None


def leaky_relu(features, 
	alpha=0.05):

	return tf.nn.relu(features) - alpha*tf.nn.relu(-features)


def get_variables_to_train(trainable_scopes=None):

	trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	return trainable_variables


def clip_grads(grads_and_vars, clipper=5.):
    with tf.name_scope('clip_gradients'):
         gvs = [(tf.clip_by_norm(grad, clipper), val) for grad,val in grads_and_vars]
         return gvs


def add_grad_noise(grads_and_vars, grad_noise=0.):
	with tf.name_scope('add_gradients_noise'):
		gvs = [(tf.add(grad, tf.random_normal(tf.shape(grad), stddev=grad_noise)), val) for grad, val in grads_and_vars]
		return gvs

def find_files(root, fntype, recursively=False):
	
	fntype = '*.'+fntype

	if not recursively:
		return glob.glob(os.path.join(root, fntype))

	matches = []
	for dirname, subdirnames, filenames in os.walk(root):
		for filename in fnmatch.filter(filenames, fntype):
			matches.append(os.path.join(dirname, filename))
	
	return matches

