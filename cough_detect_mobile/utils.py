#Author: Kevin Kipfer, Filipe Barata

import librosa
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, fnmatch, sys, random, glob

from shutil import copyfile




def simple_arg_scope(weight_decay=0.0005, 
       	             seed=0,
                     activation_fn=tf.nn.relu ):
  """Defines a simple arg scope.
       relu, xavier, 0 bias, conv2d padding Same, weight decay
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
             	      weights_initializer= tf.contrib.layers.xavier_initializer(seed=seed),# this is actually not needed
             	      activation_fn=activation_fn,
                      weights_regularizer= slim.l2_regularizer(weight_decay) if weight_decay is not None else None,
                      biases_initializer=tf.zeros_initializer()):
           with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
             	return arg_sc
'''
def better_arg_scope(weight_decay=0.0005, 
       	             seed=0,
                     activation_fn=tf.nn.elu ):
  """Defines a simple arg scope.
       elu, variance scaling, 0 bias, conv2d padding Same, weight decay
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
             	      weights_initializer= slim.variance_scaling_initializer(seed=seed),
             	      #weights_initializer= tf.contrib.layers.xavier_initializer(seed=seed),
             	      activation_fn=activation_fn,
                      weights_regularizer= slim.l2_regularizer(weight_decay) if weight_decay is not None else None,
                      biases_initializer=tf.zeros_initializer()):
           with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
             	return arg_sc
'''

def batchnorm_arg_scope(
                       is_training,
                       batch_norm_decay=0.997,
                       batch_norm_epsilon=1e-5,
                       batch_norm_scale=True):
  """Defines an arg_scope that initializes all the necessary parameters for the batch_norm
     add this if you want to use the batch norm 
  Returns:
    An arg_scope.
  """

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      normalizer_fn=slim.batch_norm, 
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
        return arg_sc


def clip_grads(grads_and_vars, clipper=5.):
    with tf.name_scope('clip_gradients'):
         gvs = [(tf.clip_by_norm(grad, clipper), val) for grad,val in grads_and_vars]
         return gvs

def add_grad_noise(grads_and_vars, grad_noise=0.):
    with tf.name_scope('add_gradients_noise'):
         gvs = [(tf.add(grad, tf.random_normal(tf.shape(grad),stddev=grad_noise)), val) for grad,val in grads_and_vars]
         return gvs


def softmax_cross_entropy_v2(onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None):
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import math_ops, nn, array_ops
  from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss, Reduction
  loss_collection=ops.GraphKeys.LOSSES
  reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
  if onehot_labels is None:
    raise ValueError("onehot_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "softmax_cross_entropy_loss",
                      (logits, onehot_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    losses = nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                  logits=logits,
                                                  name="xentropy")
    return compute_weighted_loss(
losses, weights, scope, loss_collection, reduction=reduction)


class HiddenPrints:
    """
       hide console outputs
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

 
def get_variables_to_train(trainable_scopes=None, show_variables=False, sample_rate=-1):
    """Returns a list of variables to train.

      Returns:
        A list of variables to train by the optimizer.
    """
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 

    if sample_rate>0:
       trainable_variables = random.sample(trainable_variables, int(round(len(trainable_variables)*sample_rate)))

    if trainable_scopes is None:
        return trainable_variables

    variables_to_train = []

    if show_variables:
           print ('*********************************************************************')
           print ('trainable variables: ')
    for s in trainable_scopes:
           for v in trainable_variables:
              if s in v.name:
                        variables_to_train.append(v)
                        if show_variables:
                               print (v.name)

    print ('*********************************************************************')

    return variables_to_train


def load_model(sess, 
       	checkpoint_path, 
       	copy_path = None,
        root_file = None,
	max_to_keep=10,
       	show_cp_content=False, 
       	ignore_missing_vars=False):
        """warm-start the training.
        """
       
        if not os.path.exists(checkpoint_path):
       		os.makedirs(checkpoint_path)  
       		if root_file:
                   if not copy_path:
                      copy_path=checkpoint_path
                   copyfile(root_file, copy_path+'/config.json')#+root_file)

        latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)	
        if not latest_ckpt:
               return tf.train.Saver(max_to_keep=max_to_keep)
	
        print ( 'restore from checkpoint: '+checkpoint_path )

        with HiddenPrints():
                variables = slim.get_model_variables() # slim.get_variables_to_restore()
        
        if show_cp_content:
                print ()
                print ('------------------------------------------------------------------------------')
                print ('variables stored in checkpoint:')
                from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
                print_tensors_in_checkpoint_file(latest_ckpt, '', False, False)
                print ('------------------------------------------------------------------------------')
       	
        if ignore_missing_vars:
       		reader = tf.train.NewCheckpointReader(latest_ckpt)
	       	saved_shapes = reader.get_variable_to_shape_map()

	       	var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables
	       	                            if var.name.split(':')[0] in saved_shapes])

	       	print ('nr available vars in the checkpoint: %d'%len(var_names))
	       	restore_vars = []
	       	name2var = dict(zip(map(lambda x:x.name.split(':')[0], variables), variables))
	       	with tf.variable_scope('', reuse=True):
	       	            for var_name, saved_var_name in var_names:
	       	                curr_var = name2var[saved_var_name]
	       	                var_shape = curr_var.get_shape().as_list()
	       	                if var_shape == saved_shapes[saved_var_name]:
	       	                    restore_vars.append(curr_var)

	       	print ('nr vars restored: %d'%len(restore_vars))    
	       	saver = tf.train.Saver(restore_vars, max_to_keep=max_to_keep)
        else:
                saver = tf.train.Saver(variables, max_to_keep=max_to_keep)

        saver.restore(sess,latest_ckpt)
        return saver



def find_files(root, fntype, recursively=False):
       fntype = '*.'+fntype
       
       if not recursively:
       		return glob.glob(os.path.join(root, fntype))
       
       matches = []
       for dirname, subdirnames, filenames in os.walk(root):
           for filename in fnmatch.filter(filenames, fntype):
                matches.append(os.path.join(dirname, filename))
       return matches


def remove_broken_files(root, list_of_broken_files, files):
       for broken_file in list_of_broken_files:
           broken_file = os.path.join(root, broken_file)
           if broken_file in files:
                 print ( 'file ignored: %s'+broken_file )
                 files.remove(broken_file)
       return files


def get_device(filename):

  devices_dict = {
        'Rode': "studio",
        'SamsungS5': "samsung",
        'SamsunsgS5': "samsung",
        'audio track': "audio track",
        'htc': "htc",
        'iPhone': "iphone",
        'iphone': "iphone",
        'rode': "studio",
        's5': "samsung",
        's6': "samsung",
        'samsung': "samsung",
        'samsungS5': "samsung",
        'tablet': "tablet",
        'tablet2': "tablet"
        }

  try:
    device = devices_dict[filename.split("/")[-1].split("_")[1].split("-")[0]]
    return device
  except:
    raise KeyError("unknown device for file %s"%filename)


def get_imgs(	split_id,
		db_root_dir,
    		listOfParticipantsInTestset,
    		listOfParticipantsInValidationset,
		listOfAllowedSources,
        device_cv = False,
        device = ""
	    ):
       '''
	possible experiment splits:
	1)	standard test/training a la Felipe:  split files into test- and training-set by excluding all for the testset selected persons (stored in config.json /test)
	2)	standard validation a la Felipe: randomly split off a validation set out of the training_set and train on the rest
	3)	standard test/training a la Maurice: like 1 but  only consider certain types of microphones (defined in config.json /allowedSources)
	4)	standard validation a la Maurice: like 2 but  only consider certain types of microphones (defined in config.json /allowedSources)
	5)	standard test/training a la Kevin: like 3 but instead of splitting by selected persons only split randomly 80:20 

       '''

       list_of_broken_files = ['04_Coughing/Distant (cd)/p17_rode-108.wav', '04_Coughing/Distant (cd)/p17_htc-108.wav', \
                               '04_Coughing/Distant (cd)/p17_tablet-108.wav', \
                               '04_Coughing/Distant (cd)/p17_iphone-108.wav',  '04_Coughing/Distant (cd)/p17_samsung-108.wav']

       ##
       # READING DATA
       #
       #

       print ('use data from root path %s'%db_root_dir)
       coughAll = find_files(db_root_dir + "/04_Coughing", "wav", recursively=True)
       assert len(coughAll) > 0, 'no cough files found. did you set the correct root path to the data in line 22?'
       coughAll = remove_broken_files(db_root_dir, list_of_broken_files, coughAll)
       other = find_files(db_root_dir + "/05_Other Control Sounds", "wav", recursively=True)

       #print( 'CoughAll: %d'%len(coughAll))
       #print( 'Other: %d'%len(other))
       #print( 'Total: %d'%(len(coughAll)+len(other)))

       #3+4) additional choosable tests: only consider certain types of microphones 
       if split_id > 2: 
                coughAll = [c for c in coughAll for allowedMic in listOfAllowedSources if allowedMic in get_device(c)]
                other	 = [c for c in other for allowedMic  in listOfAllowedSources if allowedMic in get_device(c)]

       #5) additional choosable tests: when only considering certain types of microphones 
       #if split_id==5:
       #         trainListOther, testListOther = train_test_split(other, test_size=0.20, random_state=42) 
       #         trainListCough, testListCough = train_test_split(coughAll, test_size=0.20, random_state=42) 
       #         return testListCough, testListOther, trainListCough, trainListOther

       ##
       # Make Sets
       #
       #



       testListOther, testListCough = [], []
       trainListCough = list(coughAll)
       trainListOther = list(other)

       #1) split files into test- and training-set by excluding all for the testset selected persons
       for name in coughAll:
                for nameToExclude in listOfParticipantsInTestset:
                        if nameToExclude in name:
                              testListCough.append(name)
                              trainListCough.remove(name)

       for name in other:
                for nameToExclude in listOfParticipantsInTestset:
                        if nameToExclude in name:
                              testListOther.append(name)
                              trainListOther.remove(name)

       #2) randomly split off a validation set out of the training_set
       if split_id%2==0:
                #trainListOther, testListOther = train_test_split(trainListOther, test_size=0.10, random_state=42) 
                #trainListCough, testListCough = train_test_split(trainListCough, test_size=0.10, random_state=42) 

                testListOther, testListCough = [], []
                coughAll = list(trainListCough)
                other = list(trainListOther) 

                for name in coughAll:
                        for nameToExclude in listOfParticipantsInValidationset:
                              if nameToExclude in name:
                                     testListCough.append(name)
                                     trainListCough.remove(name)

                for name in other:
                        for nameToExclude in listOfParticipantsInValidationset:
                              if nameToExclude in name:
                                     testListOther.append(name)
                                     trainListOther.remove(name)


       if device_cv:
           trainListCough = [kk for kk in trainListCough if get_device(kk) != device]
           trainListOther = [kk for kk in trainListOther if get_device(kk) != device]

           testListCough = [ll for ll in testListCough if get_device(ll) == device]
           testListOther = [ll for ll in testListOther if get_device(ll) == device]

       return testListCough, testListOther, trainListCough, trainListOther


def standardize(timeSignal):
         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)
         timeSignal = (timeSignal - minValue)/(maxValue - minValue) 
         return timeSignal


def extract_Signal_Of_Importance(signal, window, sample_rate ):
        """
	extract a window around the maximum of the signal
	input: 	signal
                window -> size of a window
		sample_rate 
        """

        window_size = int(window * sample_rate)			

        start = max(0, np.argmax(np.abs(signal)) - (window_size // 2))
        end = min(np.size(signal), start + window_size)
        signal = signal[start:end]

        length = np.size(signal)
        assert length <= window_size, 'extracted signal is longer than the allowed window size'
        if length < window_size:
                #pad zeros to the signal if too short
                signal = np.concatenate((signal, np.zeros(window_size-length))) 
        return signal


def preprocess(	sound_file,
                bands,
                hop_length,
                window,
                nfft
                ):

                try:
                       time_signal, sample_rate = librosa.load(sound_file, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!')
                       raise e

                time_signal = extract_Signal_Of_Importance(time_signal, window, sample_rate)
                time_signal = standardize(time_signal)
                mfcc = librosa.feature.melspectrogram(y=time_signal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length, n_fft=nfft)
                
                return mfcc


def computeLocalHuMoments(stardardized_time_signal, sample_rate, hop_length=512):
    import skimage.util
    import skimage.measure
    import scipy.fftpack

    w = 5
    mfcc = librosa.feature.melspectrogram(y=stardardized_time_signal, sr=sample_rate, n_mels=75, power=1, hop_length=hop_length,n_fft=4096)
    log_mfcc = librosa.logamplitude(mfcc)
    (n,m) = np.shape(mfcc)
    if m%w != 0:
        mzero = m + w - m%w
    else:
        mzero = m

    log_mfcc_resized = np.zeros((n,mzero))
    log_mfcc_resized[:n, :m] = log_mfcc
    energymatrix = skimage.util.view_as_blocks(log_mfcc_resized, block_shape=(w, w))

    nrow = np.shape(energymatrix)[0]
    ncol = np.shape(energymatrix)[1]
    humatrix = np.zeros(nrow,ncol)
    for i in range(0,nrow):
        for j in range(0,ncol):
            M = skimage.measure.moments(energymatrix[i,j])
            cr = M[1, 0] / M[0, 0]
            cc = M[0, 1] / M[0, 0]
            momentscentral = skimage.measure.moments_central(energymatrix[i,j], (cr, cc))
            normalizedCentralmoments= skimage.measure.moments_normalized(momentscentral)
            humoments = skimage.measure.moments_hu(normalizedCentralmoments)
            humatrix[i,j] = humoments[0]

    TQ = scipy.fftpack.dct(humatrix,axis=1)
    result = TQ[1:,:]

    return result


def make_batches(iterable, n=1):
	l=len(iterable)
	for ndx in range(0,l, n):
		yield iterable[ndx:min(ndx+n,l)]

	
def random_choice(a, axis, samples_shape=None):
    if samples_shape is None:
        samples_shape = (1,)
    shape = tuple(a.get_shape().as_list())
    dim = shape[axis]
    choice_indices = tf.random_uniform(samples_shape, minval=0, maxval=dim, dtype=tf.int32)
    samples = tf.gather(a, choice_indices, axis=axis)
    return samples

