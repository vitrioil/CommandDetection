from td_utils import *
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import time
class Sample:
	Tx = 5511
	freq = 101
	Ty = 1375 
	bg_len = 10_000
	def __init__(self,posPath,negPath,bgPath,pos_answer_len=50,train_size=100,test_size=30,saved=False):
		self.posPath = posPath
		self.negPath = negPath
		self.bgPath = bgPath
		if not saved:
			self.activates,self.negatives,self.bg = load_raw_audio(self.posPath,self.negPath,self.bgPath)
			self.pos_answer_len = pos_answer_len
			self.train_size = train_size//len(self.bg)
			self.test_size = test_size//len(self.bg)
		

	def assign_seg(self,interval):

		random_len = np.random.randint(0,self.bg_len - interval)
		return (random_len,random_len+interval)

	def is_overlapping(self,segments,data):
		if segments == []:
			return False
		for seg in segments:
			if data[0] <= seg[1] and data[1] >= seg[0]: 
				return True
		return False

	def insert_audio(self,background,segments,audio):
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

	def insert_ones(self,y,end_pos):
		end_pos = int(self.Ty*end_pos/self.bg_len)# Convert time domain to freq

		for i in range(end_pos+1,end_pos+self.pos_answer_len+1):
			if i < y.shape[1]:
				y[0,i] = 1
		return y

	def create_one_example(self,background,num,train= True):
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
		location = f"../Data/{folder}/raw/{name}{num}.wav"
		_ = background.export(location,format="wav")
		print(f"{location} saved!",end="\r")
		x = graph_spectrogram(location)
		x = x.transpose([1,0])
		return x,y


	def make_data_set(self,save=True):
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
			locTrain = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/Training/array/"
			locTest = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/Test/array/"
			np.save(locTrain+"X_train.npy",X_train)
			np.save(locTrain+"Y_train.npy",Y_train[...,np.newaxis])
			np.save(locTest+"X_test.npy",X_test)
			np.save(locTest+"Y_test.npy",Y_test[...,np.newaxis])
			print("Arrays were saved")
		return X_train,Y_train,X_test,Y_test

	def load_dataset(self):
		locTrain = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/Training/array/"
		locTest = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/Test/array/"
		X_train = np.load(locTrain+"X_train.npy")
		Y_train = np.load(locTrain+"Y_train.npy")
		X_test = np.load(locTest+"X_test.npy")
		Y_test = np.load(locTest+"Y_test.npy")
		return X_train,Y_train,X_test,Y_test

if __name__ == '__main__':
	pos = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/Pos/"
	neg = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/Neg/"
	bg = "/home/vitrioil/Documents/Scripts/Python/DL/Keras_Tut/LockApp/Data/BG/"
	s = Sample(pos,neg,bg,train_size=200,test_size=60)
	s.make_data_set()
