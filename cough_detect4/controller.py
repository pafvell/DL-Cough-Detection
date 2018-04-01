#!/usr/bin/python
#Authors: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading
import importlib
import json
from utils import *


#******************************************************************************************************************
#possible models
#from model_cnn_weak import *
#from model_cnn_v1 import *
#from model_cnn_v3_3 import *
#from model_cnn_v4 import *
#from model_resnet_v1 import *
#from model_densenet_v1 import *
#from model_boost_v9 import *
#from model_boost_v6 import *
#from model_bag_v1 import *
#from model_boost_v1 import *
#from model_rnn_v2 import *
#from model_rnn_v1 import *

#******************************************************************************************************************
#loading configuration
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

control_config = config["controller"] # reads the config for the controller file
MODEL = importlib.import_module(control_config["model"]) #loads the model specified in the config file
config_train = control_config["train"]

ROOT_DIR = config["ROOT_DIR"]
NUM_CLASSES = control_config["num_classes"]
SPEC_SIZE= control_config["spec_size"]
DB_VERSION  = config["DB_version"] #'175_61'


def get_imgs(db_name, batch_size, buffer_size=10000, prefetch_batchs=3000, num_epochs=1000):
  size_cub=SPEC_SIZE

  filename = os.path.join(ROOT_DIR, '%s%s.tfrecords'%(db_name, DB_VERSION))
  print ('use dataset location: '+filename)
  dataset = tf.data.TFRecordDataset([filename])

  def parse (record):
            keys_to_features = {
			'data': tf.VarLenFeature(tf.float32), 
                        'label': tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            image = tf.sparse_tensor_to_dense(parsed['data'], default_value=0) 
            image.set_shape([16* size_cub])
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, [16,size_cub], name='shape_image')
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

         eta= config_train["eta"], #learning rate
         grad_noise = config_train["grad_noise"],
         clipper=config_train["clipper"],
         checkpoint_dir=config_train["checkpoint_dir"],
         batch_size=config_train["batch_size"],
         trainable_scopes=config_train["trainable_scopes"],
         train_capacity=config_train["train_capacity"],
         test_capacity=config_train["test_capacity"],
         num_epochs = config_train["num_epochs"],
         gpu_fraction=config_train["gpu_fraction"],
         log_every_n_steps=config_train["log_every_n_steps"],
         eval_every_n_steps=config_train["eval_every_n_steps"],
         save_every_n_steps=config_train["save_every_n_steps"],
         save_checkpoint=config_train["save_checkpoint"]):


       print ('save checkpoints to: %s'%checkpoint_dir)

       graph = tf.Graph() 
       with graph.as_default():
              #load training data
              with tf.device("/cpu:0"):
                    train_batch, train_labels, train_op_init = get_imgs('train', batch_size=batch_size, buffer_size=train_capacity, num_epochs=num_epochs)

              #initialize
              global_step = tf.Variable(0, name='global_step', trainable=False)
              eta = tf.train.exponential_decay(eta, global_step, 80000, 0.96, staircase=False) 
              #Adam calculates the learning rate based on the initial learning rate as an upper limit. Decreasing this limit might help to reduce loss 
              #during the latest training steps, when the computed loss with the previously associated lambda(initial learning rate) parameter has stopped to decrease.
              train_op = tf.train.AdamOptimizer(learning_rate=eta) #, epsilon=1e-5 
              eta = train_op._lr

              train_loss, preds = MODEL.build_model(train_batch, train_labels)
              tf.summary.scalar('training/loss', train_loss )
              train_acc, train_acc_update = tf.metrics.accuracy(predictions=preds, labels=train_labels)
              tf.summary.scalar('training/accuracy', train_acc )
	
              #add regularization
              regularization_loss = tf.losses.get_regularization_losses() 
              if regularization_loss: 
                        train_loss += tf.add_n(regularization_loss)
                        tf.summary.scalar('training/total_loss', train_loss )
             
              #specify what parameters should be trained
              params = get_variables_to_train(trainable_scopes) 
              print ('nr trainable vars: %d'%len(params))  

              #control depenencies for batchnorm, ema, etc. + update global step
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
              with tf.control_dependencies(update_ops):
                        # Calculate the gradients for the batch of data.
                        grads = train_op.compute_gradients(train_loss, var_list = params)   
                        # gradient clipping
                        grads = clip_grads(grads, clipper=clipper)
                        # add noise
                        if grad_noise > 0:
                                grad_noise = tf.train.exponential_decay(grad_noise, global_step, 10000, 0.96, staircase=False) 
                                grads = add_grad_noise(grads, grad_noise)
                        # minimize
                        train_op = train_op.apply_gradients(grads, global_step=global_step)
                      
              #some summaries
              tf.summary.scalar('other/learning_rate', eta  )
              tf.summary.scalar('other/gradient_noise', grad_noise  )
              
              with tf.variable_scope('gradients'):
              	for grad, var in grads:
              		if grad is not None:
              			tf.summary.histogram(var.op.name, grad)    
  		       
              #collect summaries
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

              #Merge all train summaries.
              summary_op = tf.summary.merge(list(summaries), name='summary_op')

              #load Test Data
              with tf.device("/cpu:0"):
                   test_batch, test_labels, test_op_init = get_imgs('test', batch_size=batch_size, buffer_size=test_capacity, num_epochs=num_epochs)


              #Evaluation
              test_loss, predictions = MODEL.build_model(test_batch, test_labels, is_training=False, reuse=True)

              #Collect test summaries
              with tf.name_scope('evaluation' ) as eval_scope:
                      tf.summary.scalar('loss', test_loss )

                      mpc, mpc_update = tf.metrics.mean_per_class_accuracy(predictions=predictions, labels=test_labels, num_classes=NUM_CLASSES)
                      tf.summary.scalar('mpc_accuracy', mpc )

                      accuracy, acc_update = tf.metrics.accuracy(predictions=predictions, labels=test_labels)
                      tf.summary.scalar('accuracy', accuracy )

                      auc, auc_update = tf.metrics.auc(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('AUC', auc )
                      
                      precision, prec_update = tf.metrics.precision(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('precision', precision )
                      
                      recall, rec_update = tf.metrics.recall(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('recall', recall )
                      
                      #tf.summary.image('test_batch', tf.expand_dims(test_batch, -1))

              test_summary_op = tf.summary.merge(list(tf.get_collection(tf.GraphKeys.SUMMARIES, eval_scope)), name='test_summary_op')
              test_summary_update = tf.group(train_acc_update, acc_update, mpc_update, auc_update, prec_update, rec_update)

              #initialize
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
              sess = tf.Session(graph=graph,config=tf.ConfigProto(inter_op_parallelism_threads=8, gpu_options=gpu_options))
              init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

              with sess.as_default():
                sess.run(init)

              	#checkpoints              
                saver = load_model(sess, checkpoint_dir, 'controller.py')

                #wait for the queues to be filled
                time.sleep(20) 
              		    
                train_writer = tf.summary.FileWriter(checkpoint_dir+"/train", sess.graph)
                test_writer = tf.summary.FileWriter(checkpoint_dir+"/test")

                #assert that no new tensors get added to the graph after this steps 
                sess.graph.finalize()

                print ('start learning')
                try:
              	        i=0
              	        while True:
                                i+=1
              		        #training
                                _, step, train_loss_ = sess.run([train_op, global_step, train_loss])
              			#logging: update training summary
                                if i >= 300 and i%(log_every_n_steps) == 0:
                                        summary = sess.run([summary_op])[0]
                                        train_writer.add_summary(summary, step)
                           
              			#logging: update testing summary
                                if i >= 300 and i%(eval_every_n_steps) == 0:
                                        summary, mpc_, accuracy_, _ = sess.run([test_summary_op, mpc, accuracy, test_summary_update])
                                        print ('EVAL: step: %d, idx: %d, mpc: %f, accuracy: %f'% (step, i,  mpc_, accuracy_))
                                        test_writer.add_summary(summary, step)
                           
                                #save checkpoint
                                if i%(save_every_n_steps) == save_every_n_steps-1 and save_checkpoint:
                                        print ('save model (step %d)'%step)
                                        saver.save(sess,checkpoint_dir+'/checkpoints', global_step=step)

                except KeyboardInterrupt:
                      	        print("Manual interrupt occurred.")
                except tf.errors.OutOfRangeError:
                                print("End of Dataset: it ran for %d epochs"%num_epochs)

                mpc_, accuracy_, loss_ = sess.run([mpc, accuracy, test_loss])

                print ('################################################################################')
                print ('Results - mpca:%f, accuracy:%f, loss:%f'%(mpc_,accuracy_,loss_))
                print ('################################################################################')
                saver.save(sess,checkpoint_dir+'/checkpoints', global_step=step)
                sess.close()


    
def main(unused_args):


       ##
       # START TRAINING
       #
       #

       tf.set_random_seed(0)
       train()
    



if __name__ == '__main__':
       tf.app.run()    


