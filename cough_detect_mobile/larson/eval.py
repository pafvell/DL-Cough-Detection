# Author: Maurice Weber

import json
import h5py
import numpy as np
import random

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# set random seed
tf.set_random_seed(1)

# read params from config file
db_root_dir = "./Audio_Data"
db_version = "larson"

num_features = 14
num_trees = 500
batch_size = 512

db_filename = '/data_%s.h5' % db_version

latest_ckpt = tf.train.latest_checkpoint('./checkpoints')

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])

# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
params = dict(num_classes=2,
              num_features=num_features,
              num_trees=num_trees,
              # defaults:
              max_nodes=10000,
              bagging_fraction=1.0,
              valid_leaf_threshold=10,
              dominate_method='bootstrap',
              dominate_fraction=0.99,
              prune_every_samples=0)

hparams = tensor_forest.ForestHParams(**params).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Measure accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# builder
builder = tf.profiler.ProfileOptionBuilder

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

# load data
with h5py.File(db_root_dir + db_filename, 'r') as hf:
    train_devices = hf['train_devices'][:].astype('U13').tolist()
    train_labels = hf['train_data'][:, 0]
    train_features = hf['train_data'][:, 1:]
    print("shape of train features: {train_shape}".format(train_shape=np.shape(train_features)))

    test_devices = hf['test_devices'][:].astype('U13').tolist()
    test_labels = hf['test_data'][:, 0]
    test_features = hf['test_data'][:, 1:]
    print("shape of test features: {test_shape}".format(test_shape=np.shape(test_features)))

saver = tf.train.Saver()
run_metadata = tf.RunMetadata()
with tf.contrib.tfprof.ProfileContext('./tmp/train_dir',
                                      trace_steps=[],
                                      dump_steps=[]) as pctx:
    with tf.Session() as sess:
        sess.run([init_vars])
        saver.restore(sess, latest_ckpt)

        pctx.trace_next_step()

        batch_x = [train_features[1, :]]

        _ = sess.run([infer_op], feed_dict={X: batch_x}, run_metadata=run_metadata,
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
        pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.float_operation())
        pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.time_and_memory())
        pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
