#!/usr/bin/python
#Author: Kevin Kipfer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, math, shutil, time, threading

from utils import *
#from input_pipeline import *


#******************************************************************************************************************

#from model_cnn_weak import *
#from model_cnn_v1 import *
#from model_cnn_v3_3 import *
#from model_cnn_v4 import *
#from model_resnet_v1 import *
#from model_densenet_v1 import *
#from model_boost_v4 import *
from model_boost_v4_5 import *
#from model_bag_v1 import *
#from model_boost_v1 import *
#from model_rnn_v2 import *
#from model_rnn_v1 import *

#******************************************************************************************************************


ROOT_DIR = './Audio_Data'
NUM_CLASSES = 2
HOP_LENGTH  = 56
DB_VERSION  = ''

def get_imgs(db_name, batch_size, buffer_size=10000, prefetch_batchs=1000, num_epochs=1000):
  c_dict={56:64, 112:32, 224:16}          
  size_cub=c_dict[HOP_LENGTH]

  filename = os.path.join(ROOT_DIR, '%s%s_%d.tfrecords'%(db_name, DB_VERSION, HOP_LENGTH))
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

  #iterator = dataset.make_initializable_iterator()
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  return features, labels, iterator


def train(
         eta=2e-3, #learning rate
         grad_noise=1e-3,
         clipper=10.,
         #checkpoint_dir='./checkpoints/test',
         #checkpoint_dir='./checkpoints/cnn_v1.64s',
         #checkpoint_dir='./checkpoints/cnn_v2.9',
         #checkpoint_dir='./checkpoints/cnn_v3.31',
         #checkpoint_dir='./checkpoints/rnn_v1.03',
         #checkpoint_dir='./checkpoints/rnn_v2.01',
         #checkpoint_dir='./checkpoints/resnet_v1.0',
         #checkpoint_dir='./checkpoints/dense_v1.0',
         checkpoint_dir='./checkpoints/boost_v4.5l2',
         #checkpoint_dir='./checkpoints/boost_v1.0',
         batch_size=64,
         #n_producer_threads=8,
         trainable_scopes=TRAINABLE_SCOPES,
         train_capacity=6500,
         test_capacity=3000,
         num_epochs = 100000,
         gpu_fraction=1.0,
         log_every_n_steps=1000,
         eval_every_n_steps=1000,
         save_every_n_steps=10000,
         save_checkpoint=True):


       print ('save checkpoints to: %s'%checkpoint_dir)

       graph = tf.Graph() 
       with graph.as_default():
              #load training data
              with tf.device("/cpu:0"):
                    train_batch, train_labels, train_op_init = get_imgs('train', batch_size=batch_size, buffer_size=train_capacity, num_epochs=num_epochs)

              #initialize
              global_step = tf.Variable(0, name='global_step', trainable=False)
              eta = tf.train.exponential_decay(eta, global_step, 80000, 0.96, staircase=False) 
              train_op = tf.train.AdamOptimizer(learning_rate=eta) 

              train_loss, preds = build_model(train_batch, train_labels)
              tf.summary.scalar('training/train_loss', train_loss )
	
              #add regularization
              regularization_loss = tf.losses.get_regularization_losses() #use tf.losses.get_regularization_loss instead?
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
                   #test_runner = CustomRunner(test_data, is_training = False, batch_size=batch_size, capacity=test_capacity)
                   #test_batch, test_labels = test_runner.get_inputs()
                   test_batch, test_labels, test_op_init = get_imgs('test', batch_size=batch_size, buffer_size=test_capacity, num_epochs=num_epochs)


              #Evaluation
              test_loss, predictions = build_model(test_batch, test_labels, is_training=False, reuse=True)	

              #Collect test summaries
              with tf.name_scope('evaluation' ) as eval_scope:
                      tf.summary.scalar('test_loss', test_loss )

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
                      

                      tf.summary.image('test_batch', tf.expand_dims(test_batch, -1))
                      #tf.summary.histogram('predictions', predictions)
                      tf.summary.histogram('labels', test_labels)

              test_summary_op = tf.summary.merge(list(tf.get_collection(tf.GraphKeys.SUMMARIES, eval_scope)), name='test_summary_op')
              test_summary_update = tf.group(acc_update, mpc_update, auc_update, prec_update, rec_update)

              #initialize
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
              sess = tf.Session(graph=graph,config=tf.ConfigProto(inter_op_parallelism_threads=8, gpu_options=gpu_options))
              init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

              with sess.as_default():
                sess.run(init)

              	#checkpoints              
                saver = load_model(sess, checkpoint_dir, 'controller.py')

                # start the tensorflow QueueRunner's
                #tf.train.start_queue_runners(sess=sess)

                # start our custom queue runner's threads
                #train_runner.start_threads(sess, n_threads=n_producer_threads)
                #test_runner.start_threads(sess, n_threads=1)
                #sess.run([train_op_init.initializer, test_op_init.initializer])

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
                                #print ('step: %d, idx: %d, train_loss: %f'% (step, i, train_loss_))
              			#logging: update training summary
                                if i >= 300 and i%(log_every_n_steps) == 0:
                                        summary = sess.run([summary_op])[0]
                                        #print ('step: %d, idx: %d'% (step, i))
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

                #train_runner.close()
                mpc_, accuracy_, loss_ = sess.run([mpc, accuracy, test_loss])

                print ('################################################################################')
                print ('Results - mpca:%f, accuracy:%f, loss:%f'%(mpc_,accuracy_,loss_))
                print ('################################################################################')
        
                #test_runner.close()
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


