#!/usr/bin/python


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

def preprocess(sound_file,
               bands,
               hop_length,
               window,
               nfft
               ):
  try:
    time_signal, sample_rate = librosa.load(sound_file, mono=True, res_type='kaiser_fast')
  except ValueError as e:
    print('!!!!!!! librosa failed to load file: !!!!!!!!!')
    raise e

  time_signal = extract_Signal_Of_Importance(time_signal, window, sample_rate)
  time_signal = standardize(time_signal)
  mfcc = librosa.feature.melspectrogram(y=time_signal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length,
                                        n_fft=nfft)
  return mfcc




if __name__ == "__main__":


  import tensorflow as tf
  import librosa
  import numpy as np
  import importlib
  import json


  with open('UbiComp-last/wk236/config.json') as json_data_file:
    config = json.load(json_data_file)
    control_config = config["controller"]  # reads the config for the controller file
    config_db = config["dataset"]
    config_train = control_config["training_parameter"]

  model_name = control_config["model"]
  hop_length = config_db["HOP"]
  bands = config_db["BAND"]
  window = config_db["WINDOW"]
  size_cub = control_config["spec_size"]
  batch_size = config_train["batch_size"]
  num_estimator = config_train["num_estimator"]
  num_filter = config_train["num_filter"]
  nfft = 2048


  latest_ckpt = tf.train.latest_checkpoint('UbiComp-last/wk236/cv1/')

  model = importlib.import_module(model_name)

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True
  )
  sess = tf.Session(config=sess_config)

  input_tensor = tf.placeholder(tf.float32, shape=[bands, size_cub], name='Input')
  x = tf.expand_dims(input_tensor, 0)
  _, output_tensor = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter, is_training=False)

  saver = tf.train.Saver()
  saver.restore(sess, latest_ckpt)
  #save the model
  saver.save(sess, "tmp/model.ckpt")

  #save the graph
  tf.train.write_graph(sess.graph_def, 'models', 'modelx.pbtxt')

