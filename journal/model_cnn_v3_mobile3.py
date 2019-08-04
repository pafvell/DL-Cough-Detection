# Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import simple_arg_scope, batchnorm_arg_scope

from utils import softmax_cross_entropy_v2 as softmax_cross_entropy

# from tensorflow.losses import softmax_cross_entropy


"""
small conv net - train all
"""


def classify(x,
             num_classes,
             num_filter,
             dropout_keep_prob=0.5,
             weight_decay=5e-4,
             scope='model_v1',
             reuse=None,
             route=3,
             is_training=True
             ):
    """
     model used to make predictions
     input: x -> shape=[None,bands,frames,num_channels]
     output: logits -> shape=[None,num_labels]
    """
    with slim.arg_scope(simple_arg_scope(weight_decay=weight_decay)):
        # with slim.arg_scope(batchnorm_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with tf.variable_scope(scope, [x], reuse=reuse) as sc:
                # input needs to be in the format NHWC!! if there is only one channel, expand it by 1 dimension
                net = tf.expand_dims(x, -1)
                print("input X", str(net.get_shape()))
                with tf.variable_scope('stump'):
                    net = slim.separable_conv2d(net, 16, [1, 7], depth_multiplier=1, scope='conv1x7')
                    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool1')
                    net = slim.separable_conv2d(net, num_filter, [1, 5], depth_multiplier=1, scope='conv1x5')

                with tf.variable_scope('middle'):
                    for i in range(route):
                        print('input shape' + str(net.get_shape()))
                        net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool%d' % (i + 2))
                        net = slim.separable_conv2d(net, num_filter, [3, 3], depth_multiplier=1,
                                                    scope='conv3x3_%d' % (i + 2))

                    net = tf.reduce_max(net, 2)

                with tf.variable_scope('top'):
                    net = slim.flatten(net)
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
                    logits = slim.fully_connected(net, num_classes, scope='fc2', activation_fn=None)

                return logits


def build_model(x,
                y,
                num_classes=2,
                num_estimator=None,  # we missuse num_estimator for the number of convolutions
                num_filter=16,
                is_training=True,
                reuse=None,
                include_logits=False,
                loss_algo=None,
                beta=0.6
                ):
    """
     handle model. calculate the loss and the prediction for some input x and the corresponding labels y
     input: x shape=[None,bands,frames,num_channels], y shape=[None]
     output: loss shape=(1), prediction shape=[None]

    CAUTION! controller.py uses a function whith this name and arguments.
    """
    # preprocess)
    y = slim.one_hot_encoding(y, num_classes)
    # model
    logits = classify(x, num_classes=num_classes, num_filter=num_filter, route=num_estimator, is_training=is_training,
                      reuse=reuse)

    print("input y", str(y.get_shape()))
    print("logits", str(logits.get_shape()))
    # results
    if loss_algo is None or loss_algo == "cross_entropy":
        loss = softmax_cross_entropy(logits=logits, onehot_labels=y)
        loss = tf.reduce_mean(loss)
    elif loss_algo == "weighted":
        loss = weighted_cross_entropy(y, logits, beta)
        loss = tf.reduce_mean(loss)
    elif loss_algo == "balanced":
        loss = balanced_cross_entropy(y, logits, beta)
        loss = tf.reduce_mean(loss)
    elif loss_algo == "focal":
        loss = focal_loss(y, logits)
        loss = tf.reduce_mean(loss)
    elif loss_algo == "dice":
        loss = dice_loss(y, logits)
    elif loss_algo == "tversky":
        loss = tversky_loss(y, logits, beta)
    else:
        raise Exception("unknown loss %s" % loss_algo)

    predictions = tf.argmax(slim.softmax(logits), 1)

    if include_logits:
        return loss, predictions, logits

    return loss, predictions


# Parameters
# TRAINABLE_SCOPES = None #all weights are trainable


def weighted_cross_entropy(y_true, y_pred, beta):
    # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    y_pred = tf.log(y_pred / (1 - y_pred))
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    return loss


def balanced_cross_entropy(y_true, y_pred, beta):
    # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    y_pred = tf.log(y_pred / (1 - y_pred))
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    return loss * (1 - beta)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.log(y_pred / (1 - y_pred))

    weight_a = alpha * (1 - y_pred) ** gamma * y_true
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - y_true)

    return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    # some implementations don't square y_pred
    denominator = tf.reduce_sum(y_true + tf.square(y_pred))

    return numerator / (denominator + tf.keras.backend.epsilon())


def focal_loss1(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) \
           - tf.sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))


def tversky_loss(y_true, y_pred, beta):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return numerator / (tf.reduce_sum(denominator) + tf.keras.backend.epsilon())
