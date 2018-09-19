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
            max_nodes=10000,
            bagging_fraction=1.0,
            valid_leaf_threshold=10,
            dominate_method='bootstrap',
            dominate_fraction=0.99,
            prune_every_samples=0)

hparams = tensor_forest.ForestHParams(**params).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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


with tf.Session() as sess:
    # Run the initializer
    sess.run(init_vars)

    # Training
    num_samples = len(train_labels)
    for i in range(1, train_steps + 1):
        # Prepare Data
        random_indices = random.sample(range(num_samples), batch_size)
        batch_x = train_features[random_indices, :]
        batch_y = train_labels[random_indices]

        # run training op
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, loss, acc))

    tf.train.write_graph(sess.graph_def, 'graphs', 'random_forest.pbtxt')

    # Test Model
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_features, Y: test_labels}))
