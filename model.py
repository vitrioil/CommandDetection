import yaml
import h5py
import keras
import numpy as np
#from td_utils import *
import tensorflow as tf
from sample import Sample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,Nadam,RMSprop
from keras.models import Model, load_model, Sequential
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, LSTM, Conv1D, SimpleRNN

with open("hype.yaml") as f:
	hype = yaml.load(f)

hyper_param = hype["hyper"]

class TModel:
	def __init__(self, Tx, coeff, path, load_model_flag = True, load_data_flag = True):
		self.load_hyper_parameters()
		self.Tx = Tx
		self.coeff = coeff
		self.load_model_flag = load_model_flag
		self.load_data_flag = load_data_flag
		if self.load_model:
			self.load_model()
		if self.load_data_flag:
			self.load_data(path)

	def load_data(self, path):
		self.file = h5py.File(path)
		self.training = self.file["training"]
		self.test = self.file["test"]
		self.X_train = self.training["X_train"]
		self.Y_train = self.training["Y_train"]
		self.X_test = self.test["X_test"]
		self.Y_test = self.test["Y_test"]
		self.X_val = self.X_train[int(self.val_split*self.X_train.shape[0]):,...]
		self.Y_val = self.Y_train[int(self.val_split*self.Y_train.shape[0]):,...]
		self.X_train = self.X_train[:int(self.val_split*self.X_train.shape[0]), ...]
		self.Y_train = self.Y_train[:int(self.val_split*self.Y_train.shape[0]), ...]
		if self.Y_train.shape[-1] != 2:
			self.Y_train = keras.utils.to_categorical(self.Y_train,num_classes=2)
		if self.Y_test.shape[-1] != 2:
			self.Y_test = keras.utils.to_categorical(self.Y_test,num_classes=2)
		self.check_skewness()
		

	def load_hyper_parameters(self):
		self.val_split = hyper_param["val_split"]
		self.lr = float(hyper_param["lr"])
		self.batch_size = hyper_param["batch_size"]
		self.epochs = hyper_param["epochs"]
		self.logdir = hyper_param["logdir"]
		self.optimizer = hyper_param["optimizer"]
		self.loss = hyper_param["loss"]
		self.metrics = hyper_param["metrics"]

	def check_skewness(self):
		y = np.argmax(self.Y_train, axis = 2)
		val = np.sum(y)
		print("Training", val / (self.Y_train.shape[0] * self.Y_train.shape[1]))
		y = np.argmax(self.Y_test, axis = 2)
		val = np.sum(y)
		print("Testing", val / (self.Y_test.shape[0] * self.Y_test.shape[1]))

	def makeModel(self,shape):

		X_input = Input(shape = shape)

		X = Conv1D(filters=128,kernel_size=2,strides=2)(X_input)
		X = BatchNormalization()(X)
		X = Activation("relu")(X)
		X = Dropout(0.15)(X)

		X = LSTM(10, return_sequences=True)(X)
		X = BatchNormalization()(X)
		X = Activation("relu")(X)

		X = LSTM(10, return_sequences=True)(X)
		X = BatchNormalization()(X)
		X = Activation("relu")(X)

		X = LSTM(10, return_sequences=True)(X)
		X = BatchNormalization()(X)
		X = Activation("relu")(X)

		X = TimeDistributed(Dense(2,activation="softmax"))(X)

		model = Model(inputs=X_input,outputs=X)
		return model

	def train(self):
		if self.load_model_flag:
			self.load_model()
		else:
			self.model = self.makeModel((self.Tx,self.coeff))
		self.graph = tf.get_default_graph()
		print(self.model.summary())
		opt = keras.optimizers.__dict__[self.optimizer]
		self.model.compile(opt(lr=self.lr),loss=self.loss,metrics=[m for m in self.metrics])
		try:
			self.model.fit(self.X_train,self.Y_train,batch_size=self.batch_size,epochs=self.epochs,
					verbose=1,shuffle="batch", validation_data = (self.X_val, self.Y_val), 
					callbacks = [keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq = 1)])
		except KeyboardInterrupt as e:
			print(str(e))
		self.model.save("../data/Model/model3.h5")
		loss,acc = self.model.evaluate(self.X_test,self.Y_test,batch_size=self.batch_size)
		print(f"Testing accuracy {acc}")
		print(f"Testing loss {loss}")
		self.detect_triggerword(0, "../data/Training/raw/train0.wav", plot_graph = True)

	def load_model(self):
		print("Loading model")
		self.model = keras.models.load_model("../data/Model/model3.h5")
		self.graph = tf.get_default_graph()
		print("Loaded")
	
	def get_wav_info(wav_file): 
		rate, data = wavfile.read(wav_file) 
		return rate, data 	
	
	def graph_spectrogram(wav_file):
		rate, data = TModel.get_wav_info(wav_file)
		nfft = 200 # Length of each window segment
		fs = 8000 # Sampling frequencies
		noverlap = 120 # Overlap between windows
		nchannels = data.ndim
		if nchannels == 1:
		  pxx = plt.specgram(data, nfft, fs, noverlap = noverlap)[0]
		elif nchannels == 2:
		  pxx = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)[0]
		return pxx[:,:5511]
	
	def detect_triggerword(self, i, filename, plot_graph=False):
		#x = Sample.calc_mfcc(filename)
		x =  self.X_train[i]
		if x.shape[0] >= 1999:
			x = x[:1999, :]
		else:
			print(x.shape, "Quitting not enough info")
			return
		if len(x.shape) < 3:
			x = x[np.newaxis,...]
		with self.graph.as_default():
			y = self.model.predict(x)
		y = np.argmax(np.squeeze(y), axis = 1)
		if plot_graph:
			plt.subplot(3, 1, 1)
			_ = TModel.graph_spectrogram(filename)
			plt.subplot(3, 1, 2)
			plt.plot(y)
			plt.subplot(3, 1, 3)
			plt.plot(np.argmax(np.squeeze(self.Y_train[i]), axis = 1))
			plt.show()
		return y
