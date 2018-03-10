#!/usr/bin/python
#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import *
#from input_pipeline import *
import argparse

#******************************************************************************************************************

#from model_cnn_weak import *
from model_cnn_v2 import *
#from model_cnn_v3_3 import *
#from model_cnn_v4 import *
#from model_resnet_v1 import *
#from model_densenet_v1 import *
#from model_boost_v4 import *
#from model_boost_v4_5 import *
#from model_bag_v1 import *
#from model_boost_v1 import *
#from model_rnn_v2 import *
#from model_rnn_v1 import *

#******************************************************************************************************************


parser = argparse.ArgumentParser()
parser.add_argument('--eta',
                    default=3e-3,
                    type=float,
                    help='learning rate')
parser.add_argument('--grad_noise',
                    default=1e-3,
                    type=float,
                    help='amount of gradient noise')
parser.add_argument('--clipper',
                    default=10.,
                    type=float,
                    help='amount of gradient clipping')
parser.add_argument('--num_classes',
                    default=2,
                    type=int,
                    help='number of used classes')
parser.add_argument('--checkpoint_dir',
                    default='./checkpoints/default',
                    help='path where the new checkpoints should be saved to')
parser.add_argument('--restore_model_path',
                    default=None,
                    help='path where the system will find the checkpoints, that it shall use to restore the model')
parser.add_argument('--dataset_dir',
                    default=None,
                    help='path where the dataset is stored')
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    help='size of the mini-batch')
parser.add_argument('--trainable_scopes',
                    default=TRAINABLE_SCOPES,
                    help='which scopes of the model should all be trainable')
parser.add_argument('--num_epochs',
                    default=100000,
                    type=int,
                    help='max number of epochs - how many times does it train with the entire dataset. \
                          The training stops when either num_epochs or num_steps is reached')
parser.add_argument('--num_steps',
                    default=None,
                    type=int,
                    help='max number of training steps - how many minibatches are used to train the model? \
                          If num_steps is set to None it uses num_epochs as a boundary; \
                          otherwise it stops when either num_epochs or num_steps is reached')
parser.add_argument('--train_capacity',
                    default=6500,
                    type=int,
                    help='buffer size for the training input queue')
parser.add_argument('--test_capacity',
                    default=3000,
                    type=int,
                    help='buffer size for the testing input queue')
parser.add_argument('--gpu_fraction',
                    default=1.0,
                    type=float,
                    help='what percentage of the gpu shall be used?')
parser.add_argument('--log_every_n_steps',
                    default=1000,
                    type=int,
                    help='how frequent (how many steps) shall it log the training summaries?')
parser.add_argument('--eval_every_n_steps',
                    default=1000,
                    type=int,
                    help='how frequent (how many steps) shall it log the testing/evaluation summaries?')
parser.add_argument('--save_every_n_steps',
                    default=10000,
                    type=int,
                    help='how frequent (how many steps) shall it save the model checkpoints? this flag is only used if save_checkpoints is set to True (default)')
parser.add_argument('--no_checkpoints', dest='save_checkpoint', action='store_false',
                    help='shall the model checkpoints be saved at all?')
parser.set_defaults(save_checkpoint=True)
parser.add_argument('--no_training', dest='do_training', action='store_false',
                    help='if this flag is set the model is only evaluated')
parser.set_defaults(do_training=True)



SPEC_SIZE=64

def get_imgs(db_name, batch_size, buffer_size=10000, prefetch_batchs=1000, num_epochs=1000):
  #c_dict={56:64, 112:32, 224:16}          
  size_cub=SPEC_SIZE#c_dict[HOP_LENGTH]

  print ('use dataset location: '+db_name)
  dataset = tf.data.TFRecordDataset([db_name])

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

  #iterator = dataset.make_initializable_iterator()
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  return features, labels, iterator


def train(args):

       print ('save checkpoints to: %s'%args.checkpoint_dir)
       print type(args.batch_size)
       graph = tf.Graph() 
       with graph.as_default():
              #load training data
              with tf.device("/cpu:0"):
                    train_batch, train_labels, train_op_init = get_imgs('%s_%s.tfrecords'%(args.dataset_dir, 'train'), 
                                                                        batch_size=args.batch_size, 
                                                                        buffer_size=args.train_capacity, 
                                                                        num_epochs=args.num_epochs)

              #initialize
              global_step = tf.Variable(0, name='global_step', trainable=False)
              eta = tf.train.exponential_decay(args.eta, global_step, 80000, 0.96, staircase=False) 
              train_op = tf.train.AdamOptimizer(learning_rate=eta) 

              train_loss, preds = build_model(train_batch, train_labels)
              tf.summary.scalar('training/train_loss', train_loss )
	
              #add regularization
              regularization_loss = tf.losses.get_regularization_losses() #use tf.losses.get_regularization_loss instead?
              if regularization_loss: 
                        train_loss += tf.add_n(regularization_loss)
                        tf.summary.scalar('training/total_loss', train_loss )
             
              #specify what parameters should be trained
              params = get_variables_to_train(args.trainable_scopes) 
              print ('nr trainable vars: %d'%len(params))  

              #control depenencies for batchnorm, ema, etc. + update global step
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
              with tf.control_dependencies(update_ops):
                        # Calculate the gradients for the batch of data.
                        grads = train_op.compute_gradients(train_loss, var_list = params)   
                        # gradient clipping
                        grads = clip_grads(grads, clipper=args.clipper)
                        # add noise
                        if args.grad_noise > 0:
                                grad_noise = tf.train.exponential_decay(args.grad_noise, global_step, 10000, 0.96, staircase=False) 
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
                   test_batch, test_labels, test_op_init = get_imgs('%s_%s.tfrecords'%(args.dataset_dir, 'test'),
									batch_size=args.batch_size, 
									buffer_size=args.test_capacity, 
									num_epochs=args.num_epochs)


              #Evaluation
              test_loss, predictions = build_model(test_batch, test_labels, is_training=False, reuse=True)	

              #Collect test summaries
              with tf.name_scope('evaluation' ) as eval_scope:
                      tf.summary.scalar('test_loss', test_loss )

                      mpc, mpc_update = tf.metrics.mean_per_class_accuracy(predictions=predictions, labels=test_labels, num_classes=args.num_classes)
                      tf.summary.scalar('mpc_accuracy', mpc )

                      accuracy, acc_update = tf.metrics.accuracy(predictions=predictions, labels=test_labels)
                      tf.summary.scalar('accuracy', accuracy )

                      auc, auc_update = tf.metrics.auc(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('AUC', auc )
                      
                      precision, prec_update = tf.metrics.precision(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('precision', precision )
                      
                      recall, rec_update = tf.metrics.recall(labels=test_labels, predictions=predictions)
                      tf.summary.scalar('recall', recall )
                      

              test_summary_op = tf.summary.merge(list(tf.get_collection(tf.GraphKeys.SUMMARIES, eval_scope)), name='test_summary_op')
              test_summary_update = tf.group(acc_update, mpc_update, auc_update, prec_update, rec_update)

              #initialize
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
              sess = tf.Session(graph=graph,config=tf.ConfigProto(inter_op_parallelism_threads=8, gpu_options=gpu_options))
              init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

              with sess.as_default():
                sess.run(init)

              	#checkpoints              
                saver = load_model(sess, args.checkpoint_dir, restore_model_path=args.restore_model_path, root_files=['controller.py'])

                #wait for the queues to be filled
                time.sleep(20) 
              		    
                train_writer = tf.summary.FileWriter(args.checkpoint_dir+"/train", sess.graph)
                test_writer = tf.summary.FileWriter(args.checkpoint_dir+"/test")

                #assert that no new tensors get added to the graph after this steps 
                sess.graph.finalize()

                print ('start learning')
                try:
              	        i=0
              	        while not args.num_steps or i<=args.num_steps:
                                i+=1
              		        #training
                                _, step, train_loss_ = sess.run([train_op, global_step, train_loss])
                                #print ('step: %d, idx: %d, train_loss: %f'% (step, i, train_loss_))
              			#logging: update training summary
                                if i >= 300 and i%(args.log_every_n_steps) == 0:
                                        summary = sess.run([summary_op])[0]
                                        #print ('step: %d, idx: %d'% (step, i))
                                        train_writer.add_summary(summary, step)
                           
              			#logging: update testing summary
                                if i >= 300 and i%(args.eval_every_n_steps) == 0:
                                        summary, mpc_, accuracy_, _ = sess.run([test_summary_op, mpc, accuracy, test_summary_update])
                                        print ('EVAL: step: %d, idx: %d, mpc: %f, accuracy: %f'% (step, i,  mpc_, accuracy_))
                                        test_writer.add_summary(summary, step)
                           
                                #save checkpoint
                                if args.save_checkpoint and i%(args.save_every_n_steps) == args.save_every_n_steps-1:
                                        print ('save model (step %d)'%step)
                                        saver.save(sess,args.checkpoint_dir+'/checkpoint', global_step=step)

                except KeyboardInterrupt:
                      	        print("Manual interrupt occurred.")
                except tf.errors.OutOfRangeError:
                                print("End of Dataset: it ran for %d epochs"%args.num_epochs)

                mpc_, accuracy_, loss_ = sess.run([mpc, accuracy, test_loss])

                print ('################################################################################')
                print ('Results - mpca:%f, accuracy:%f, loss:%f'%(mpc_,accuracy_,loss_))
                print ('################################################################################')
                saver.save(sess,args.checkpoint_dir+'/checkpoints', global_step=step)
                sess.close()



def evaluate(args):

       graph = tf.Graph() 
       with graph.as_default():
              #load Test Data
              with tf.device("/cpu:0"):
                   test_batch, test_labels, test_op_init = get_imgs('%s_%s.tfrecords'%(args.dataset_dir, 'test'),
									batch_size=args.batch_size, 
									buffer_size=args.test_capacity, 
									num_epochs=args.num_epochs)
              #Evaluation
              test_loss, predictions = build_model(test_batch, test_labels, is_training=False, reuse=False)	

              #Collect test summaries
              with tf.name_scope('evaluation' ) as eval_scope:
                      tf.summary.scalar('test_loss', test_loss )

                      mpc, mpc_update = tf.metrics.mean_per_class_accuracy(predictions=predictions, labels=test_labels, num_classes=args.num_classes)
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
                      #tf.summary.histogram('labels', test_labels)

              test_summary_op = tf.summary.merge(list(tf.get_collection(tf.GraphKeys.SUMMARIES, eval_scope)), name='test_summary_op')
              test_summary_update = tf.group(acc_update, mpc_update, auc_update, prec_update, rec_update)

              #initialize
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
              sess = tf.Session(graph=graph,config=tf.ConfigProto(inter_op_parallelism_threads=8, gpu_options=gpu_options))
              init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

              with sess.as_default():
                sess.run(init)

                #saver = load_model(sess, args.checkpoint_dir, restore_model_path=args.restore_model_path, root_files=['controller.py'])
                load_model(sess, None, restore_model_path=args.restore_model_path, root_files=['controller.py'])

                #wait for the queues to be filled
                time.sleep(2) 
              		    
                #test_writer = tf.summary.FileWriter(args.checkpoint_dir+"/test")

                print ('start evalutation')
                try:
              	        i=0
              	        while True:
                                i+=1
                                summary, mpc_, accuracy_, precision_, recall_, loss_, _ = sess.run([test_summary_op, mpc, accuracy, precision, recall, test_loss, test_summary_update])
                                if i >= 300 and i%(100) == 0:
                                        print ('EVAL: idx: %d, mpc: %f, accuracy: %f'% (i,  mpc_, accuracy_))
                                        #test_writer.add_summary(summary, step)
                           
                except KeyboardInterrupt:
                      	        print("Manual interrupt occurred.")
                except tf.errors.OutOfRangeError:
                                print("End of Dataset")


                print ('################################################################################')
                print ('Results - mpca:%f, accuracy:%f, precision:%f, recall:%f, loss:%f'%(mpc_,accuracy_,precision_, recall_, loss_))
                print ('################################################################################')
                sess.close()




    
def main(unused_args):

       args = parser.parse_args()

       tf.set_random_seed(0)

       if args.do_training:
          train(args)
       else:
          evaluate(args)




if __name__ == '__main__':
       tf.app.run()    


