import time
import numpy as np
from td_utils import *
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt

class Sample:
	Tx = 5511
	freq = 101
	Ty = 1375 
	bg_len = 10_000
	logdir = "../data/"
	locTrain = "{logdir}/Training/array/"
	locTest = "{logdir}/Test/array/"
	def __init__(self, posPath, negPath, bgPath, pos_answer_len=50, train_size=100, test_size=30, saved=False):
		self.posPath = posPath
		self.negPath = negPath
		self.bgPath = bgPath
		if not saved:
			self.activates,self.negatives,self.bg = load_raw_audio(self.posPath,self.negPath,self.bgPath)
			self.pos_answer_len = pos_answer_len
			self.train_size = train_size//len(self.bg)
			self.test_size = test_size//len(self.bg)

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
			if data[0] <= seg[1] and data[1] >= seg[0]: 
				return True
		return False

	def insert_audio(self, background: AudioSegment , segments: list, audio: AudioSegment) -> (AudioSegment, , list, bool):
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
			bool: A boolean implying the given audio is added into background 
				file or it couldn't find an interval 
		'''
		interval = self.assign_seg(len(audio))
		timeout = time.time() + 0.1*60
		t = time.time()
		while(self.is_overlapping(segments,interval)):
			interval = self.assign_seg(len(audio)) 
			if time.time() > t:
				return background,interval,segments,True
		segments.append(interval)
		background = background.overlay(audio-10,position = interval[0])
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

	def create_one_example(self, background: AudioSegment, num: int, train = True) -> np.ndarray, np.ndarray:
		'''
			Creates one training sample from a given background audio
			This function samples audio files from recordings 
			Inserts into background segment and inserts ones into 
			labels whenever it samples from a positive audio file

			input: background, num, train

			background: AudioSegment, Background audio file
			num: int, Indicating the training sample number
			train: bool, Indicating trainig or testing set

			returns training sample and label
		'''
		background = background - 20

		total_pos,total_neg = np.random.randint(1,5),np.random.randint(1,5)
		pos_data,neg_data = [],[]
		y = np.zeros((1,self.Ty),dtype=np.int)
		segments = []
		for i in range(total_pos):
			random_index = np.random.randint(0,len(self.activates))
			background,interval,segments,timed = self.insert_audio(background,segments,self.activates[random_index])
			if timed:
				i -= 1
			else:
				pos_data.append(self.activates[random_index])
				y = self.insert_ones(y,interval[1])

		for i in range(total_neg):
			random_index = np.random.randint(0,len(self.negatives))
			background,interval,segments,timed = self.insert_audio(background,segments,self.negatives[random_index])
			if timed:
				i -= 1
			else:
				neg_data.append(self.negatives[random_index])
			
		background = match_target_amplitude(background,-20)
		folder = "Training" if train else "Test"
		name = "train" if train else "test"
		location = f"{logdir}/{folder}/raw/{name}{num}.wav"#f"../Data/{folder}/raw/{name}{num}.wav"
		_ = background.export(location,format="wav")
		print(f"{location} saved!",end="\r")
		x = graph_spectrogram(location)
		x = x.transpose([1,0])
		return x,y


	def make_data_set(self, save=True) -> None:
		'''
			Makes training and testing data

			inputs: save

			save: boolean, saves the training data
		'''
		X_train,Y_train = self.create_one_example(self.bg[np.random.randint(0,len(self.bg))],0)
		X_train = X_train[np.newaxis,...]
		num = 1
		for bg in self.bg:
			for i in range(self.train_size):
				x,y = self.create_one_example(bg,num)
				X_train = np.concatenate((X_train,x[np.newaxis,...]),axis=0)
				Y_train = np.concatenate((Y_train,y),axis=0)
				num += 1
		print()
		print("Training set was made")
		X_test,Y_test = self.create_one_example(self.bg[np.random.randint(0,len(self.bg))],0)
		X_test = X_test[np.newaxis,...]
		num = 1
		for bg in self.bg:
			for i in range(self.test_size):
				x,y = self.create_one_example(bg,num,train=False)
				X_test = np.concatenate((X_test,x[np.newaxis,...]),axis=0)
				Y_test = np.concatenate((Y_test,y),axis=0)
				num += 1
		print()
		print("Testing set was made")
		if save:
			np.save(self.locTrain+"X_train.npy",X_train)
			np.save(self.locTrain+"Y_train.npy",Y_train[...,np.newaxis])
			np.save(self.locTest+"X_test.npy",X_test)
			np.save(self.locTest+"Y_test.npy",Y_test[...,np.newaxis])
			print("Arrays were saved")
		return X_train,Y_train,X_test,Y_test

	def load_dataset(self):
		X_train = np.load(self.locTrain+"X_train.npy")
		Y_train = np.load(self.locTrain+"Y_train.npy")
		X_test  = np.load(self.locTest+"X_test.npy")
		Y_test  = np.load(self.locTest+"Y_test.npy")
		return X_train,Y_train,X_test,Y_test

if __name__ == '__main__':
	pos = "../data/Pos/"
	neg = "../data/Neg/"
	bg =  "../data/BG/"
	s = Sample(pos,neg,bg,train_size=200,test_size=60)
	s.make_data_set()
