#!/usr/bin/python
# Authors: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading
import importlib
import json
import argparse
from utils import *

# ******************************************************************************************************************
# possible models
# from model_cnn_weak import *
# from model_cnn_v1 import *
# from model_cnn_v3_3 import *
# from model_cnn_v4 import *
# from model_resnet_v1 import *
# from model_densenet_v1 import *
# from model_boost_v9 import *
# from model_boost_v6 import *
# from model_bag_v1 import *
# from model_boost_v1 import *
# from model_rnn_v2 import *
# from model_rnn_v1 import *

# ******************************************************************************************************************

# loading config file
parser = argparse.ArgumentParser()
parser.add_argument('-config',
                    type=str,
                    default='config.json',
                    help='store a json file with all the necessary parameters')
args = parser.parse_args()

# loading configuration
with open(args.config) as json_data_file:
    config = json.load(json_data_file)
control_config = config["controller"]  # reads the config for the controller file
config_db = config["dataset"]
config_train = control_config["training_parameter"]


# ******************************************************************************************************************


def get_imgs(db_name,
             batch_size,
             num_epochs,
             buffer_size,
             size_cub,
             db_version,
             root_dir,
             prefetch_batchs=3000,
             device=""):
    if device:
        filename = os.path.join(root_dir, '%s%s%s.tfrecords' % (db_name, db_version, device))
    else:
        filename = os.path.join(root_dir, '%s%s.tfrecords' % (db_name, db_version))
    print ('use dataset location: ' + filename)
    dataset = tf.data.TFRecordDataset([filename])

    def parse(record):
        keys_to_features = {
            'data': tf.VarLenFeature(tf.float32),
            'label': tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.sparse_tensor_to_dense(parsed['data'], default_value=0)
        image.set_shape([16 * size_cub])
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [16, size_cub], name='shape_image')
        label = tf.cast(parsed["label"], tf.int32)
        return image, label

    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(prefetch_batchs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels, iterator


def train(
        model_name=control_config["model"],
        split_id=config_db["split_id"],
        checkpoint_dir=config_train["checkpoint_dir"],
        db_version=config["DB_version"],
        root_dir=config["ROOT_DIR"],
        eta=config_train["eta"],  # learning rate
        grad_noise=config_train["grad_noise"],
        clipper=config_train["clipper"],
        batch_size=config_train["batch_size"],
        num_classes=control_config["num_classes"],
        num_estimator=config_train["num_estimator"],
        num_filter=config_train["num_filter"],
        trainable_scopes=config_train["trainable_scopes"],
        size_cub=control_config["spec_size"],
        train_capacity=config_train["train_capacity"],
        test_capacity=config_train["test_capacity"],
        num_epochs=config_train["num_epochs"],
        max_num_steps=config_train["max_num_steps"],
        # if None it trains until num_epochs is reached. Otherwise whatever is reached first
        gpu_fraction=config_train["gpu_fraction"],
        log_every_n_steps=config_train["log_every_n_steps"],
        eval_every_n_steps=config_train["eval_every_n_steps"],
        save_every_n_steps=config_train["save_every_n_steps"],
        save_checkpoint=config_train["save_checkpoint"],
        device_name_for_db="",
        device_name_for_test=""):
    if device_name_for_db:
        checkpoint_dir = checkpoint_dir + device_name_for_db

    tf.set_random_seed(0)

    model = importlib.import_module(model_name)  # loads the model specified in the config file
    print ('save checkpoints to: %s' % checkpoint_dir)
    print ('train model:' + model_name)

    graph = tf.Graph()
    with graph.as_default():
        # load training data
        with tf.device("/cpu:0"):
            train_batch, train_labels, train_op_init = get_imgs('train_%d' % split_id,
                                                                batch_size=batch_size,
                                                                buffer_size=train_capacity,
                                                                num_epochs=num_epochs,
                                                                size_cub=size_cub,
                                                                db_version=db_version,
                                                                root_dir=root_dir,
                                                                device=device_name_for_db
                                                                )

        # initialize
        global_step = tf.Variable(0, name='global_step', trainable=False)
        eta = tf.train.exponential_decay(eta, global_step, 80000, 0.96, staircase=False)
        # Adam calculates the learning rate based on the initial learning rate as an upper limit. Decreasing this
        # limit might help to reduce loss during the latest training steps, when the computed loss with the
        # previously associated lambda(initial learning rate) parameter has stopped to decrease.
        train_op = tf.train.AdamOptimizer(learning_rate=eta)  # , epsilon=1e-5
        eta = train_op._lr

        train_loss, preds = model.build_model(train_batch, train_labels, num_estimator=num_estimator,
                                              num_filter=num_filter)
        tf.summary.scalar('training/loss', train_loss)
        train_acc, train_acc_update = tf.metrics.accuracy(predictions=preds, labels=train_labels)
        tf.summary.scalar('training/accuracy', train_acc)

        # add regularization
        regularization_loss = tf.losses.get_regularization_losses()
        if regularization_loss:
            train_loss += tf.add_n(regularization_loss)
            tf.summary.scalar('training/total_loss', train_loss)

        # specify what parameters should be trained
        params = get_variables_to_train(trainable_scopes)
        print ('nr trainable vars: %d' % len(params))

        # control depenencies for batchnorm, ema, etc. + update global step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Calculate the gradients for the batch of data.
            grads = train_op.compute_gradients(train_loss, var_list=params)
            # gradient clipping
            grads = clip_grads(grads, clipper=clipper)
            # add noise
            if grad_noise > 0:
                grad_noise = tf.train.exponential_decay(grad_noise, global_step, 10000, 0.96, staircase=False)
                grads = add_grad_noise(grads, grad_noise)
            # minimize
            train_op = train_op.apply_gradients(grads, global_step=global_step)

        # some summaries
        tf.summary.scalar('other/learning_rate', eta)
        tf.summary.scalar('other/gradient_noise', grad_noise)

        with tf.variable_scope('gradients'):
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name, grad)

                    # collect summaries
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Merge all train summaries.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # load Test Data
        with tf.device("/cpu:0"):
            test_batch, test_labels, test_op_init = get_imgs('test_%d' % split_id,
                                                             batch_size=batch_size,
                                                             buffer_size=test_capacity,
                                                             num_epochs=num_epochs,
                                                             size_cub=size_cub,
                                                             db_version=db_version,
                                                             root_dir=root_dir,
                                                             device=device_name_for_test)

        # Evaluation
        test_batch = tf.placeholder_with_default(test_batch, shape=test_batch.get_shape(), name='Input')
        test_loss, predictions = model.build_model(test_batch, test_labels, num_estimator=num_estimator,
                                                   num_filter=num_filter, is_training=False, reuse=True)
        predictions = tf.identity(predictions, 'Prediction')

        # Collect test summaries
        with tf.name_scope('evaluation') as eval_scope:
            tf.summary.scalar('loss', test_loss)

            mpc, mpc_update = tf.metrics.mean_per_class_accuracy(predictions=predictions, labels=test_labels,
                                                                 num_classes=num_classes)
            tf.summary.scalar('mpc_accuracy', mpc)

            accuracy, acc_update = tf.metrics.accuracy(predictions=predictions, labels=test_labels)
            tf.summary.scalar('accuracy', accuracy)

            auc, auc_update = tf.metrics.auc(labels=test_labels, predictions=predictions)
            tf.summary.scalar('AUC', auc)

            precision, prec_update = tf.metrics.precision(labels=test_labels, predictions=predictions)
            tf.summary.scalar('precision', precision)

            recall, rec_update = tf.metrics.recall(labels=test_labels, predictions=predictions)
            tf.summary.scalar('recall', recall)

            # tf.summary.image('test_batch', tf.expand_dims(test_batch, -1))

        test_summary_op = tf.summary.merge(list(tf.get_collection(tf.GraphKeys.SUMMARIES, eval_scope)),
                                           name='test_summary_op')
        test_summary_update = tf.group(train_acc_update, acc_update, mpc_update, auc_update, prec_update, rec_update)

        # initialize
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=8, gpu_options=gpu_options))
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with sess.as_default():
            sess.run(init)

            # checkpoints
            saver = load_model(sess, checkpoint_dir + '/cv%d' % split_id, checkpoint_dir, args.config)  # 'config.json')

            # wait for the queues to be filled
            time.sleep(20)

            train_writer = tf.summary.FileWriter(checkpoint_dir + '/train_%d' % split_id)  # , sess.graph)
            test_writer = tf.summary.FileWriter(checkpoint_dir + '/test_%d' % split_id)

            # assert that no new tensors get added to the graph after this steps
            sess.graph.finalize()

            print ('start learning')
            try:
                i = 0
                while not max_num_steps or i <= max_num_steps:
                    i += 1
                    # training
                    _, step, train_loss_ = sess.run([train_op, global_step, train_loss])
                    # logging: update training summary
                    if i >= 500 and i % log_every_n_steps == 0:
                        summary = sess.run([summary_op])[0]
                        train_writer.add_summary(summary, step)

                    # logging: update testing summary
                    if i >= 500 and i % eval_every_n_steps == 0:
                        summary, auc_, accuracy_, loss_, prec_, rec_, _ = sess.run(
                            [test_summary_op, auc, accuracy, test_loss, precision, recall, test_summary_update])
                        print ('EVAL: step: %d, idx: %d, auc: %f, accuracy: %f' % (step, i, auc_, accuracy_))
                        test_writer.add_summary(summary, step)

                    # save checkpoint
                    if save_checkpoint and i % save_every_n_steps == save_every_n_steps - 1:
                        print ('save model (step %d)' % step)
                        saver.save(sess, checkpoint_dir + '/cv%d/checkpoint' % split_id, global_step=step)

            except KeyboardInterrupt:
                print("Manual interrupt occurred.")
            except tf.errors.OutOfRangeError:
                print("End of Dataset: it ran for %d epochs" % num_epochs)

            print ('################################################################################')
            print ('Results - AUC:%f, accuracy:%f, precision:%f, recall:%f, loss:%f' % (
            auc_, accuracy_, prec_, rec_, loss_))
            print ('################################################################################')
            saver.save(sess, checkpoint_dir + '/cv%d/checkpoint' % split_id, global_step=step)
            sess.close()

            return auc_, accuracy_, prec_, rec_


# ******************************************************************************************************************


def main(unused_args):
    '''
          if DO_CV:
                results = []
                for i in range(5):
                     out = train(cv_partition_id=i)
                     results.append(out)
             
                results = np.array(results)
                mean_ = np.mean(results, axis=0)
                std_ = np.std(results, axis=0)
                min_ = np.min(results, axis=0)
                max_ = np.max(results, axis=0)
                measures = ['auc ', 'acc ', 'prec', 'rec ']
                assert mean_.shape[0] == 4, 'theres a bug in the mean calculation. 4 != %d'%mean_.shape[0]


                print ()
                print ('***********************************************************************************************')
                print ('-------------------------------CROSS VALIDATION-------------------------------------')
                print ('***********************************************************************************************')
                for j in range(4):
                     print ('%s - mean: %f, std:%f, min:%f, max%f' %(measures[j], mean_[j], std_[j], min_[j], max_[j]))
                print ('***********************************************************************************************')
          else: 
                '''
    train()


# combinations
# "studio", "iphone", "samsung", "htc", "tablet",

# studio**
# studioiphone
# studiosamsung
# studiohtc
# studiotablet

# iphone**
# iphonesamsung
# iphonehtc
# iphonetablet

# samsung**
# samsunghtc
# samsungtablet

# tablet**
# tablethtc


def main_device_cv(device_name_for_db, device_name_for_test):
    train(device_name_for_db=device_name_for_db, device_name_for_test=device_name_for_test)


if __name__ == '__main__':
    if config["DEVICE_CV"]:
        # for device in config["dataset"]["allowedSources"]:
        #
        #         if device == "audio track" :
        #             continue

        main_device_cv(device_name_for_db="samsungtablet", device_name_for_test="samsung")

    else:
        tf.app.run()
