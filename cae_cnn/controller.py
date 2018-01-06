#Authors: Kevin Kipfer, Filipe Barata, Maurice Weber

import tensorflow as tf
import numpy as np

from preprocessing import *
from utils import *
from input_pipeline import *

#******************************************************************************************************************

from model_cae import *

#******************************************************************************************************************


ROOT_DIR = './AudioData'


def train_autoencoder(train_data,
	test_data,
	eta=1e-2, #learning rate
	grad_noise=1e-3,
	batch_size=64,
	train_capacity=1500,
	trainable_scopes=TRAINABLE_SCOPES):

	
	graph = tf.Graph()
	with graph.as_default():
		
		# load training data
		with tf.device('/cpu:0'):
			train_runner = CustomRunner_ae(train_data, batch_size=batch_size, capacity=train_capacity)
			train_batch = train_runner.get_inputs()

		# initialize
		global_step = tf.Variable(0, name='global_step', trainable=False)
		eta = tf.train.exponential_decay(eta, global_step, 100000, 0.96, staircase=False)
		train_op = tf.train.AdamOptimizer(learning_rate=eta)

		train_loss, train_approx = build_model2(train_batch)

		# specify which variables should be trained
		params = get_variables_to_train(trainable_scopes)
		print('nr of trainable variables: %d'%len(params))

		# control dependencies for batchnorm, ema, etc + update update global step
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			# calculate the gradients for the batch of data
			grads = train_op.compute_gradients(train_loss, var_list=params)
			# gradient clipping
			grads = clip_grads(grads)
			# add noise
			if grad_noise > 0:
				grad_noise = tf.train.exponential_decay(grad_noise, global_step, 10000, 0.96, staircase=False)
				grads = add_grad_noise(grads, grad_noise)
			# minimize
			train_op = train_op.apply_gradients(grads, global_step=global_step)

		# some summaries
		# DO THIS

		# Load test data
		with tf.device('/cpu:0'):
			test_runner = CustomRunner_ae(train_data, is_training=False, batch_size=batch_size, capacity=test_capacity)
			test_batch = test_runner.get_inputs()

		# Evaluation
		test_loss, test_approx = build_model2(test_batch)
		current_cost = tf.reduce_sum(tf.square(test_batch - test_approx))

		# Initialize
		sess = tf.Session(graph=graph)
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with sess.as_default():
			sess.run(init)

			print('start learning')

			try:
				i = 0
				while True:

					i+= 1
					# training
					_, step, train_loss_ = sess.run([train_op, global_step, train_loss])

			except KeyboardInterrupt:
				print("Manual interrupt occurred.")
				train_runner.close()
				#finally:
				current_cost_ = sess.run(current_cost)

				print ('################################################################################')
				print ('Results - Cost:%f'%(current_cost_))
				print ('################################################################################')

				sess.close()


def train_cnn():

	return None


def main(unused_args):

	listOfParticipantsToExcludeInTrainset = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"] #participants used in the test-set

	list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
	'04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

	##
	# READING COUGH DATA
	#
	#

	print ('use data from root path %s'%ROOT_DIR)

	coughAll = find_files(ROOT_DIR + "/04_Coughing", "wav", recursively=True)
	assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'

	#remove broken files
	for broken_file in list_of_broken_files:
		broken_file = os.path.join(ROOT_DIR, broken_file)
		if broken_file in coughAll:
			print ( 'file ignored: %s'%broken_file )
			coughAll.remove(broken_file)

	#split cough files into test- and training-set
	testListCough = []
	trainListCough = coughAll
	for name in coughAll:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListCough.append(name)
				trainListCough.remove(name)

	print('nr of samples coughing: %d' % len(testListCough))

	##
	# READING OTHER DATA
	#
	#

	other = find_files(ROOT_DIR + "/05_Other Control Sounds", "wav", recursively=True)

	testListOther = []
	trainListOther = other
	for name in other:
		for nameToExclude in listOfParticipantsToExcludeInTrainset:
			if nameToExclude in name:
				testListOther.append(name)
				trainListOther.remove(name)

	print('nr of samples NOT coughing: %d' % len(testListOther))


	train_data = (trainListCough, trainListOther)
	test_data = (testListCough, testListOther)

	##
	# START TRAINING
	#
	#

	print('Kick off training procedure')

	tf.set_random_seed(0)
	train_autoencoder(train_data, test_data)


if __name__ == '__main__':
	tf.app.run()
































