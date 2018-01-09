#Author: Maurice Weber

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from utils import *




def compress_decompress2(x,
	scope='model_cae',
	reuse=None,
	is_training=True):
	
	with slim.arg_scope(simple_arg_scope()):
		with slim.arg_scope([slim.batch_norm], is_training=is_training):
			with tf.variable_scope(scope, [x], reuse=reuse) as sc:

				net = tf.expand_dims(x,-1)
				print('model input_shape: %s'%net.get_shape())

				# encoder
				net = slim.conv2d(net, 64, [3,9], scope='conv1')
				net = slim.conv2d(net, 128, [3,5], scope='conv2')

				# decoder
				net = slim.conv2d(net, 128, [3,5], scope='conv2')
				net = slim.conv2d(net, 64, [3,9], scope='conv1')

				y = slim.conv2d(net, 1, [3,9], scope='conv_final')

				return y


def build_model2(x,
	is_training=True):
	
	# build model
	x_approx = compress_decompress2(x)

	# cost function measures pixelwise difference
	# loss = tf.reduce_sum(tf.square(x_approx - x))
	
	return x_approx




def compress_decompress(x,
	scope='model_cae',
	reuse=None,
	is_training=True,
	n_filters = [1, 10, 10, 10],
	filter_sizes = [3, 3, 3, 3],
	strides = [1, 2, 2, 1]):

	print("DEBUGGING SHAPE SHIT: ", x.get_shape().as_list())
	
	# ensure 2-D is converted to square tensor
	if len(x.get_shape()) == 2:
		x_dim = np.sqrt(x.get_shape().as_list()[1])
		if x_dim != int(x_dim):
			raise ValueError('NSUPPORTED INPUT DIMENSIONS')
		x_dim = int(x_dim)
		x_tensor = tf.reshape(x, [-1, x_dim, x_dim, n_filters[0]])
	elif len(x.get_shape()) == 4:
		x_tensor = x
	else:
		raise ValueError('UNSUPPORTED INPUT DIMENSIONS')
	current_input = x_tensor


	# optional
	if augment:
		# do some data augmenation stuff
		pass

	
	# build encoder
	encoder = []
	shapes = []
	for layer_i, n_output in enumerate(n_filters[1:]):
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())
		
		# weights
		W = tf.Variable(tf.random_uniform([
			filter_sizes[layer_i],
			filter_sizes[layer_i],
			n_input, n_output],
			-1.0 / math.sqrt(n_input),
			1.0 / math.sqrt(n_input)))

		# biases
		b = tf.Variable(tf.zeros([n_output]))

		encoder.append(W)
		output = leaky_relu(
			features=tf.add(tf.nn.conv2d(
			current_input, W, strides=strides, padding='SAME'), b),
			alpha=0.1)
		current_input = output


	# store the latent representation
	z = current_input
	encoder.reverse()
	shapes.reverse()


	# build the decoder using the same weights
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

		output = leaky_relu(
			features=tf.add(tf.nn.conv2d_transpose(
				current_input, W, tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
				strides=strides, padding='SAME'), b),
			alpha=0.5)
		current_input = output


	# reconstruction through network
	y = current_input

	return y

	
def build_model(x,
	is_training=True):
	
	# build model
	x_approx = compress_decompress(x)

	# cost function measures pixelwise difference
	loss = tf.reduce_sum(tf.square(x_approx - x))
	
	return loss, x_approx



TRAINABLE_SCOPES = None



































































































































