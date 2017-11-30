import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.style.use('ggplot')


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(fileNames, bands=60, frames=2):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []

    for fn in fileNames:
        sound_clip, s = extract_Signal_Of_Importance(fn)
        label = '0'
        if 'Coughing' in fn:
            label = '1'

        for (start, end) in windows(sound_clip, window_size):
            start = int(start)
            end = int(end)
            if (len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                logspec = librosa.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)

    data = np.asarray(log_specgrams)
    log_specgrams = data.swapaxes(1, 2).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def extract_feature(file_name):
    signal, sample_rate = extract_Signal_Of_Importance(file_name)
    stft = np.abs(librosa.stft(signal))
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(signal, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz



def extract_Signal_Of_Importance(file_name):
    X, sample_rate = librosa.load(file_name)
    maxValue = np.max(np.abs(X))
    absX = np.abs(X)
    indMax = absX.tolist().index(maxValue)
    numberOfSamples = np.ceil(sample_rate * 0.160) #average time of half a cough
    startInd = int(np.max(indMax - numberOfSamples,0))
    maxLeng = np.size(X)
    if startInd + 2*numberOfSamples > maxLeng - 1:
        endInd = int(maxLeng - 1)
        startInd = int(endInd - 2 * numberOfSamples)
    else:
        endInd = int(startInd + 2*numberOfSamples)

    signal = X[startInd:endInd]
    return signal, sample_rate



def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')