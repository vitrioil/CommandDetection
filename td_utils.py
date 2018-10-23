import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
	rate, data = get_wav_info(wav_file)
	nfft = 200 # Length of each window segment
	fs = 8000 # Sampling frequencies
	noverlap = 120 # Overlap between windows
	nchannels = data.ndim
	if nchannels == 1:
		pxx = plt.specgram(data, nfft, fs, noverlap = noverlap)[0]
	elif nchannels == 2:
		pxx = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)[0]
	return pxx[:,:5511]


# Load a wav file
def get_wav_info(wav_file):
	rate, data = wavfile.read(wav_file)
	return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
	change_in_dBFS = target_dBFS - sound.dBFS
	return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(pos_list,neg_list,bg_list):
	activates = []
	backgrounds = []
	negatives = []
	for pos in pos_list:
		print(os.listdir(pos))
		for filename in os.listdir(pos):
			if filename.endswith("wav"):
				activate = AudioSegment.from_wav(pos+filename)
				activates.append(activate)

	for bg in bg_list:
		print(os.listdir(bg))
		for filename in os.listdir(bg):
			if filename.endswith("wav"):
				background = AudioSegment.from_wav(bg+filename)			
				backgrounds.append(background)
		
	for neg in neg_list:
		print(os.listdir(neg))
		for filename in os.listdir(neg):
			if filename.endswith("wav"):
				negative = AudioSegment.from_wav(neg+filename)
				negatives.append(negative)
	return activates, negatives, backgrounds

#print(graph_spectrogram("../Data/Training/raw/train0.wav").shape)
#print(graph_spectrogram("../Data/BG/BG3.wav").shape)
#print(graph_spectrogram("../Data/BG/BG4.wav").shape)
