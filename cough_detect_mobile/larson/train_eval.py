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
batch_size = 16
train_steps = 500

db_filename = '/data_%s.h5' % db_version

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])

# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
params = dict(num_classes=2,
              num_features=num_features,
              num_trees=num_trees,
              # defaults:
              max_nodes=1500,
              bagging_fraction=1.0,
              valid_leaf_threshold=5,
              dominate_method='bootstrap',
              dominate_fraction=0.99,
              prune_every_samples=0)

hparams = tensor_forest.ForestHParams(**params).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# adds tree variables to resources.shared_resources -> also to graph?
for tree_in_forest in forest_graph.trees:
    _ = tensor_forest.TreeVariables(hparams,
                                    tree_num=tree_in_forest.tree_num,
                                    training=True)
forest_size = forest_graph.average_size()

# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# builder
builder = tf.profiler.ProfileOptionBuilder

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

# saver = tf.train.Saver(var_list=[tf.GraphKeys.GLOBAL_VARIABLES, resources.shared_resources()])

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

run_metadata = tf.RunMetadata()
with tf.Session() as sess:
    sess.run(init_vars)

    # Training
    num_samples = len(train_labels)
    try:
        for i in range(1, train_steps + 1):
            # Prepare Data
            random_indices = random.sample(range(num_samples), batch_size)
            batch_x = train_features[random_indices, :]
            batch_y = train_labels[random_indices]

            # run training op
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y},
                               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            if i % 10 == 0 or i == 1:
                acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
                print("==========")
                print(sess.run([forest_size], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                               run_metadata=run_metadata))
                print("==========")
                print('Step %i, Loss: %f, Acc: %f' % (i, loss, acc))

        # Test Model
        print("Test Accuracy:",
              sess.run(accuracy_op, feed_dict={X: test_features, Y: test_labels},
                       options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata))

    except KeyboardInterrupt:
        pass

    infer_graph = forest_graph.inference_graph(input_data=X)

    random_indices = random.sample(range(num_samples), batch_size)
    batch_x = train_features[random_indices, :]
    batch_y = train_labels[random_indices]

    with tf.contrib.tfprof.ProfileContext('./tmp/train_dir',
                                          trace_steps=[],
                                          dump_steps=[]) as pctx:
        pctx.trace_next_step()
        sess.run([infer_op], feed_dict={X: batch_x}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata)
        pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.time_and_memory())
        pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.float_operation())

# # Test Model
# print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_features, Y: test_labels}))
#
# random_indices = random.sample(range(num_samples), batch_size)
# batch_x = train_features[random_indices, :]train
# batch_y = train_labels[random_indices]
#
#
#
# tf.reset_default_graph()
#
#     with
#
#     with tf.contrib.tfprof.ProfileContext('./tmp/train_dir',
#                                           trace_steps=[],
#                                           dump_steps=[]) as pctx:
#         pctx.trace_next_step()
#         pctx.dump_next_step()
#         _ = sess.run([infer_op], feed_dict={X: batch_x}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
#                      run_metadata=run_metadata)
#         pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.time_and_memory())
#         pctx.profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.float_operation())
