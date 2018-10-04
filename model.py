import keras
import numpy as np
from td_utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, LSTM, Conv1D

class TModel:

	def __init__(self,Tx,freq,X_train,Y_train,X_test,Y_test):
		self.Tx = Tx
		self.freq = freq
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
		if self.Y_train.shape[2] != 2:
			self.Y_train = keras.utils.to_categorical(self.Y_train,num_classes=2)
		if self.Y_test.shape[2] != 2:
			self.Y_test = keras.utils.to_categorical(self.Y_test,num_classes=2)

	def makeModel(self,shape):

		X_input = Input(shape = shape)

		X = Conv1D(filters=128,kernel_size=13,strides=4)(X_input)
		X = BatchNormalization()(X)
		X = Activation("relu")(X)
		X = Dropout(0.15)(X)

		X = LSTM(128,return_sequences=True)(X)
		X = Dropout(0.15)(X)

		X = LSTM(128,return_sequences=True)(X)
		#X = Dropout(0.1)(X)
		X = BatchNormalization()(X)
		X = Dropout(0.15)(X)

		X = TimeDistributed(Dense(2,activation="softmax"))(X)

		model = Model(inputs=X_input,outputs=X)
		return model

	def train(self,lr=1e-4,batch_size=1,epochs=2,saved=False):
		if saved:
			self.load_model()
		else:
			self.model = self.makeModel((self.Tx,self.freq))
		print(self.model.summary())
		self.model.compile(Adam(lr=lr),loss="categorical_crossentropy",metrics=["categorical_accuracy"])
		try:
			self.model.fit(self.X_train,self.Y_train,batch_size=batch_size,epochs=epochs,verbose=1,shuffle=True)
		except KeyboardInterrupt as e:
			print(str(e))
		self.model.save("../Data/Model/modelNew.h5")
		loss,acc = self.model.evaluate(self.X_test,self.Y_test,batch_size=batch_size)
		print(f"Testing accuracy {acc}")
		print(f"Testing loss {loss}")

	def load_model(self):
		self.model = keras.models.load_model("../Data/Model/model2.h5")
		self.graph = tf.get_default_graph()

	def detect_triggerword(self,filename,plot_graph=False):
		plt.subplot(2,1,1)
		x = graph_spectrogram(filename)
		# the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
		x  = x.swapaxes(0,1)
		x = np.expand_dims(x, axis=0)
		if x.shape[1] < 5511:
			print(x.shape)
			return None
		x = x[:,:5511,:]
		with self.graph.as_default():
			predictions = self.model.predict(x)
		predictions = np.argmax(predictions,axis=2)
		if plot_graph:
			plt.subplot(2, 1, 2)
			plt.plot(predictions[0,:])
			plt.ylabel('probability')
			plt.show()
		return predictions
