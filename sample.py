import os
import gc
import sys
import h5py
import time
import psutil
import operator
import librosa
import numpy as np
from pympler import tracker
from scipy.io import wavfile
from pydub import AudioSegment
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

class Augment:
	'''
		https://www.kaggle.com/huseinzol05/sound-augmentation-librosa
	'''

	def pitch_speed(sample, sample_rate):
		'''
			Changes pitch and speed of audio
		'''
		if len(sample.shape) > 1 and  sample.shape[1] != 1:
			print("Only mono audio supported")
			return
		y_pitch_speed = sample.copy()
		# you can change low and high here
		length_change = np.random.uniform(low=0.5,high=1.5)
		speed_fac = 1.0  / length_change
		tmp = np.interp(np.arange(len(y_pitch_speed),step = speed_fac),np.arange(len(y_pitch_speed)),y_pitch_speed)
		minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
		y_pitch_speed *= 0
		y_pitch_speed[0:minlen] = tmp[0:minlen]
		
		return y_pitch_speed

	def pitch(sample, sample_rate):
		'''
			Changes pitch
		'''
		y_pitch = sample
		bins_per_octave = 24
		pitch_pm = 4
		pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)   
		y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
		return y_pitch
	
	def distribution_noise(sample, sample_rate):
		'''
			Adds noise
		'''
		mode = 1 if len(sample.shape) == 1 else sample.shape[1]
		y_noise = sample.copy()
		if mode == 1:
			y_noise = y_noise[...,np.newaxis]
		# you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
		noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)
		try:
			y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=(y_noise.shape[0], mode))
		except MemoryError as e:
			print(str(e))
			return 
		return y_noise

class Sample:
	memory_limit = 90
	Tx = 1999
	coeff = 20
	Ty = 999
	classes = 2
	bg_len = 10_000
	logdir = "../data/"
	locTrain = f"{logdir}/Training/array/"
	locTest = f"{logdir}/Test/array/"
	def __init__(self, posPath, negPath, bgPath, pos_answer_len=40,
			train_size=1000, test_size=300, saved=False):
		self.posPath = posPath
		self.negPath = negPath
		self.bgPath = bgPath
		if not saved:
			self.activates,self.negatives,self.bg = Sample.load_raw_audio(
									self.posPath,self.negPath,self.bgPath
								)
			
			self.activates.sort(key = lambda x: len(x))
			self.negatives.sort(key = lambda x: len(x))
			assert np.all(np.diff([len(i) for i in self.activates])) >= 0
			assert np.all(np.diff([len(i) for i in self.negatives])) >= 0
			self.train_size = train_size + 1
			self.test_size = test_size + 1
			self.init_dataset()
			self.pos_answer_len = pos_answer_len
	
	def __enter__(self, *args, **kwargs):
		pass

	def __exit__(self, *args, **kwargs):
		self.h5_file.close()

	def match_target_amplitude(sound: AudioSegment, target_dBFS:int) -> AudioSegment:
		change_in_dBFS = target_dBFS - sound.dBFS
		return sound.apply_gain(change_in_dBFS)
	
	def load_raw_audio(pos_list: list, neg_list:list, bg_list:list) -> (list, list, list):
		activates = []
		backgrounds = []
		negatives = []
		for pos in pos_list:
			for filename in os.listdir(pos):
				if filename.endswith("wav"):
					activate = AudioSegment.from_wav(pos+filename)
					activates.append(activate)
	
		for bg in bg_list:
			for filename in os.listdir(bg):
				if filename.endswith("wav"):
					background = AudioSegment.from_wav(bg+filename)                 
					backgrounds.append(background)
         
		for neg in neg_list:
			for filename in os.listdir(neg):
				if filename.endswith("wav"):
					negative = AudioSegment.from_wav(neg+filename)
					negatives.append(negative)
		return activates, negatives, backgrounds

	def calc_mfcc(file_name: str, nfft = 2048) -> np.ndarray:
		(rate,sig) = wav.read(file_name)
		#if the audio is mono and not stereo
		if len(sig.shape) == 1:
			print("converting to stereo")
			sig = np.repeat(sig[..., np.newaxis], 2, axis = 1)
			print(sig.shape)
		mfcc_feat = mfcc(sig,rate,nfft=nfft)
		d_mfcc_feat = delta(mfcc_feat, 2)
		fbank_feat = logfbank(sig,rate,nfft=nfft)
		fbank_feat = fbank_feat[..., :20]
		return fbank_feat
	
	def init_dataset(self, flag='w'):
		self.h5_file = h5py.File("dataset.h5", flag)
		self.train_group = self.h5_file.create_group("training")
		self.test_group = self.h5_file.create_group("test") 
		self.X_train = self.train_group.create_dataset("X_train", shape = (self.train_size, self.Tx, self.coeff), dtype = np.float32)
		self.X_test = self.test_group.create_dataset("X_test", shape = (self.test_size, self.Tx, self.coeff), dtype = np.float32)
		self.Y_train = self.train_group.create_dataset("Y_train", shape = (self.train_size, self.Ty, self.classes), dtype = np.float32)
		self.Y_test = self.test_group.create_dataset("Y_test", shape = (self.test_size, self.Ty, self.classes), dtype = np.float32)

	def assign_seg(self,interval_length: int) -> (int, int):
		'''
			Assigns a random segment of time in the audio file

			input: interval
			interval_length: int, interval length of the segment

			returns a tuple of segment (start_segment, end_segment)
		'''

		random_len = np.random.randint(0,self.bg_len - interval_length)
		return (random_len,random_len+interval_length)

	def is_overlapping(self, segments: list, interval: tuple) -> bool:
		'''
			checks if given segment doesn't overlap with list
			of segment

			input: segments, interval

			interval: tuple, (start_time, end_time) of a given segment
			segments: list of tuple of (start_time, end_time)
		
			returns a boolean implying if the given interval overlaps with 
			list of segments
		'''
		if segments == []:
			return False
		for seg in segments:
			if (interval[0] <= seg[1] + self.pos_answer_len and interval[1] + self.pos_answer_len >= seg[0]):
				return True
		return False

	def insert_audio(self, background: AudioSegment , segments: list, audio: AudioSegment) -> (AudioSegment, list, list, bool):
		'''
			Finds an empty interval to insert audio 
			into a background audio file, if no such
			interval exists don't add it to background file

			input: background, segments, audio

			background: AudioSegment, An audio file that contains background noise
			segments: list, List of intervals that contains the given audio file
			audio: AudioSegment, Positive or Negative audio file 

			returns a tuple of 
			background: AudioSegment, An audio file overlayed with the given audio file
			interval: list, List of given intervals of audio file
			segments: list, updated input
			bool: A boolean implying the given audio is added into background 
				file or it couldn't find an interval 
		'''
		interval = self.assign_seg(len(audio))
		count = 0 
		while(self.is_overlapping(segments,interval)):
			interval = self.assign_seg(len(audio)) 
			count += 1
			if count == 200:
				return background,interval,segments,True
		segments.append(interval)
		background = background.overlay(audio - 15, position = interval[0])
		return background,interval,segments,False

	def insert_ones(self, y: list, end_pos: int) -> list:
		'''
			Insert ones at the position where trigger word was said

			input: y, end_pos

			y: list, The label for the neural network
			end_pos: int, The position where the label should
					be one upto self.pos_answer_len

			returns updated list with ones added to the location
		'''
		end_pos = int(self.Ty*end_pos/self.bg_len)# Convert time domain to freq

		for i in range(end_pos+1,end_pos+self.pos_answer_len+1):
			if i < y.shape[1]:
				y[0,i] = 1
		return y

	def create_one_example(self, background: AudioSegment, num: int, train = True) -> (np.ndarray, np.ndarray):
		'''
			Creates one training sample from a given background audio
			This function samples audio files from recordings 
			Inserts into background segment and inserts ones into 
			labels whenever it samples from a positive audio file

			input: background, num, train

			background: AudioSegment, Background audio file
			n#um: int, Indicating the training sample number
			train: bool, Indicating trainig or testing set

			returns training sample and label
		'''
		decrease = 5
		background = background - decrease
		total_pos,total_neg = np.random.randint(2,3),np.random.randint(2,3)
		pos_data,neg_data = [],[]
		y = np.zeros((1,self.Ty),dtype=np.int)
		segments = []
		time_out_check = 5
		start = time.time()
		random_index = np.random.randint(0,len(self.activates))
		while total_pos >= 0:
			background,interval,segments,timed_out = self.insert_audio(background,segments,self.activates[random_index])
			if timed_out:
				total_pos += 1
				if random_index:
					random_index -= 1
				if time.time() - start > time_out_check:
					break
			else:
				pos_data.append(self.activates[random_index])
				y = self.insert_ones(y,interval[1])
				random_index = np.random.randint(0,len(self.activates))
			total_pos -= 1
		
		start = time.time()
		random_index = np.random.randint(0,len(self.negatives))
		while total_neg >= 0:
			background,interval,segments,timed_out = self.insert_audio(background,segments,self.negatives[random_index])
			if timed_out:
				total_neg += 1
				if random_index:
					random_index -= 1
				if time.time() - start > time_out_check:
					break
			else:
				neg_data.append(self.negatives[random_index])
				random_index = np.random.randint(0,len(self.negatives))
			total_neg -= 1

		background = Sample.match_target_amplitude(background,-decrease)
		folder = "Training" if train else "Test"
		name = "train" if train else "test"
		location = f"{self.logdir}/{folder}/raw/{name}{num}.wav"#f"../Data/{folder}/raw/{name}{num}.wav"
		background.export(location,format="wav")
		print(f"{location} saved!",end="\r")
		#x = graph_spectrogram("LOCKK.wav")
		#x = x.transpose([1,0])
		x = Sample.calc_mfcc(f"{location}")
		return x,y


	def make_data_set(self, save=True) -> None:
		'''
			Makes training and testing data

			inputs: save

			save: boolean, saves the training data
		'''
		num = 0
		try:
			for bg in self.bg:
				for i in range(self.train_size//len(self.bg)):
					x,y = self.create_one_example(bg,num)
					self.X_train[num, :, :]  = x
					self.Y_train[num, :]  = to_categorical(y, self.classes)
					num += 1
				assert psutil.virtual_memory().percent < self.memory_limit, "Memory usage high"
		except KeyboardInterrupt as e:
			print(str(e))
		gc.collect()
		self.X_train.flush()
		self.Y_train.flush()
		print()
		print("Training set was made")
		num = 0
		try:
			for bg in self.bg:
				for i in range(self.test_size//len(self.bg)):
					x,y = self.create_one_example(bg,num,train=False)
					self.X_test[num, :, :] = x
					self.Y_test[num, :] = to_categorical(y, self.classes)
					num += 1
				assert psutil.virtual_memory().percent < self.memory_limit, "Memory usage high"
		except KeyboardInterrupt as e:
			print(str(e))
		print()
		print("Testing set was made")

if __name__ == '__main__':
	names = ["Ameya/", "Prem/", "Raj/", "Vinit/", "Vivek/"]
	pos = [os.path.join("../data/Pos/",name) for name in names]
	neg = [os.path.join("../data/Neg/"+name) for name in names]
	bg = ["../data/BG/"]
	s = Sample(pos,neg,bg,train_size=800,test_size=200)
	with s:
		s.make_data_set()

