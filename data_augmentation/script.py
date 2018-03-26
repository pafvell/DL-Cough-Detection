import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# load params
WINDOW_SIZE = 0.16
BANDS = 16
HOP = 56
N_FFT = 256

# augmenting params
AUGM_LIST = ['add_noise', 'pitch_shift', 'time_stretch']
AUGM_METHOD = AUGM_LIST[2]

NOISE_STDEV = 2e-1
N_STEPS_PITCH_SHIFT = 5
STRETCH_FACTOR = 0.8


def extract_Signal_Of_Importance(signal, sample_rate, window=WINDOW_SIZE):
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


def standardize(timeSignal):

	       #TODO
         maxValue = np.max(timeSignal)
         minValue = np.min(timeSignal)

         #maxValue = 1.7
         #minValue = -1.8

         timeSignal = (timeSignal - minValue)/(maxValue - minValue) 

         #but since timeSignal is in [-1.8,1.7]
         #timeSignal /= 1.8
         return timeSignal


def add_noise(signal, sigma=NOISE_STDEV):
        '''
        Input:
        sound signal; time series vector, standardized
        Output:
        sound signal + gaussian noise
        '''
        std = sigma * np.max(signal)
        noise_mat = np.random.randn(signal.shape[0])*std
        return signal + noise_mat


def pitch_shift(signal, sample_rate, n_steps=N_STEPS_PITCH_SHIFT):

        # as in https://librosa.github.io/librosa/generated/librosa.effects.pitch_shift.html#librosa.effects.pitch_shift

        return librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=n_steps)


def time_stretch(signal, sample_rate, window_size=WINDOW_SIZE, stretch_factor=STRETCH_FACTOR):

        # as in https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html#librosa.effects.time_stretch

        signal = librosa.effects.time_stretch(y=signal, rate=stretch_factor)

        return extract_Signal_Of_Importance(signal=signal, window=window_size, sample_rate=sample_rate)


def apply_augment(signal, sample_rate, method=None):

      if method == None:
        return signal

      elif method == "add_noise":
        return add_noise(signal=signal)

      elif method == "pitch_shift":
        return pitch_shift(signal=signal, sample_rate=sample_rate)

      elif method == "time_stretch":
        return time_stretch(signal=signal, sample_rate=sample_rate)

      else:
      	raise NotImplementedError()


def load_audio_file(f):
	
	signal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')

	return extract_Signal_Of_Importance(signal=signal, window = WINDOW_SIZE, sample_rate=sample_rate), sample_rate


coughing_file = 'coughing_p05_htc-50.wav'
laughing_file = 'laughing_p05_htc-83.wav'
speaking_file = 'speaking_p05_iPhone-106.wav'

files = [coughing_file, laughing_file, speaking_file]

# f = laughing_file

for f in files:

	data, sr = load_audio_file(f='./input/' + f)
	data = standardize(data)

	mfcc = librosa.feature.melspectrogram(y=data, 
										sr=sr,
										n_mels=BANDS,
										power=1,
										hop_length=HOP,
										n_fft=N_FFT)


	data_augm = apply_augment(data, sample_rate=sr, method=AUGM_METHOD)
	mfcc_augm = librosa.feature.melspectrogram(y=data_augm, 
										sr=sr,
										n_mels=BANDS,
										power=1,
										hop_length=HOP,
										n_fft=N_FFT)



	# plots
	fig = plt.figure(figsize=(14, 7))

	# plot signal without data augm
	plt.subplot(411)
	plt.title(f.split("_")[0])
	plt.ylabel('Amplitude')
	plt.plot(np.linspace(0, 1, len(data)), data)

	# plot original mfcc
	plt.subplot(412)
	plt.imshow(mfcc)

	# plot data augm signal
	plt.subplot(413)
	plt.title("augmentation method: %s, %f"%(AUGM_METHOD, N_STEPS_PITCH_SHIFT))
	plt.ylabel('Amplitude')
	plt.plot(np.linspace(0, 1, len(data)), data)

	#plot augmented mfcc
	plt.subplot(414)
	plt.imshow(mfcc_augm)

	plt.tight_layout()

	plt.show()





























