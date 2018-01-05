"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.
Parag K. Mital, Jan 2016

source: https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py

store and restore models:
https://stackoverflow.com/questions/35486961/how-to-store-trained-weights-and-biases-in-model-a-for-use-in-model-b-in-tensorf

"""


import librosa
import tensorflow as tf
import numpy as np
import math


def leaky_relu(features, 
	alpha=0.05):

	return tf.nn.relu(features) - alpha*tf.nn.relu(-features)


def autoencoder(input_shape = [None, 784],
	n_filters = [1, 15, 10, 5],
	filter_sizes = [3, 3, 3, 3],
	strides = [1, 2, 2, 1],
	augment = False):
	'''
	
	building deep denoising autoencoder with [optionally] tied weights

	'''


	# input to the network
	x = tf.placeholder(tf.float32, input_shape, name='x')

	# ensure 2-D is converted to square tensor
	if len(x.get_shape()) == 2:
		x_dim = np.sqrt(x.get_shape().as_list()[1])
		if x_dim != int(x_dim):
			raise ValueError('Error: UNSUPPORTED INPUT DIMENSIONS')
		x_dim = int(x_dim)
		x_tensor = tf.reshape(x, [-1, x_dim, x_dim, n_filters[0]])
	elif len(x.get_shape()) == 4:
		x_tensor = x
	else:
		raise ValueError('Error: UNSUPPORTED INPUT DIMENSIONS')
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

	# cost function measures pixelwise difference
	cost = tf.reduce_sum(tf.square(y - x_tensor))
	
	return {'x': x, 'z': z, 'y': y, 'cost': cost}




def test_mnist(do_plots = True):
	'''
	
	use cae on mnist data set
	
	'''

	import tensorflow as tf
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	import matplotlib.pyplot as plt


	# load MNIST
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	mean_img = np.mean(mnist.train.images, axis=0)
	ae = autoencoder()

	# some hyperparams
	learning_rate = 1e-2
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

	# create session to use graph
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# fit all training data
	batch_size = 100
	n_epochs = 20
	print('Start learning')
	for epoch_i in range(n_epochs):
		for batch_i in range(mnist.train.num_examples // batch_size):
			batch_xs, _ = mnist.train.next_batch(batch_size)
			train = np.array([img - mean_img for img in batch_xs])
			sess.run(optimizer, feed_dict={ae['x']: train})
		print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))


	# plot some reconstructions
	if do_plots:
		n_examples = 10
		test_xs, _ = mnist.train.next_batch(n_examples)
		test_xs_norm = np.array([img - mean_img for img in test_xs])
		recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
		print(recon.shape)
		fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
		for example_i in range(n_examples):
			axs[0][example_i].imshow(
				np.reshape(test_xs[example_i, :], (28, 28)))
			axs[1][example_i].imshow(
				np.reshape(
					np.reshape(recon[example_i, ...], (784,)) + mean_img,
					(28, 28)))
		fig.show()
		plt.draw()
		plt.waitforbuttonpress()


	return None


if __name__ == "__main__":
	test_mnist()



































































































