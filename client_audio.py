
'''



self.frames : list -> deque

'''


import wave
import time
import socket
import pyaudio
import numpy as np
from queue import Queue
from model import TModel
from sample import Sample
from threading import Thread

frames = []
ip_port = ("127.0.0.1", 30002)

class Listen:
	host = ""
	port = 30002
	def __init__(self,p,stream,form=pyaudio.paInt16,chunk=1024*2,channels=2,
			rate=44100,shift_bytes=275, rate=16000, 
			threshold=2500, silence_limit=1, prev_audio_limit = 0.5):
		self.p = p
		self.stream = stream
		self.form = form
		self.chunk = chunk
		self.channels = channels
		self.rate = rate
		self.rel = self.rate/self.chunk
		self.threshold = threshold
		self.silence_limit = silence_limit
		self.prev_audio_limit = prev_audio_limit
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.s.bind((self.host, self.port))
		self.q = Queue()
		self.frames = []
		self.saved = False
		self.shift_bytes = shift_bytes
		self.closed = False

	def _close(self):
		print("Closing the socket")
		self.closed = True
		self.s.close()
	
	def _send(msg):
		self.con.send(msg)

	def record(self):    
	    while True:
	        self.frames.append(stream.read(self.chunk))


	def save(self,remove=True,filename="test.wav"):
		'''
		Save the wav file. 
		'''
		if self.saved or len(self.frames) == 0:
			return
		with wave.open(filename, 'wb') as wf:
			wf.setnchannels(self.channels)
			wf.setsampwidth(self.p.get_sample_size(self.form))
			wf.setframerate(self.rate)
			print("Saving bytes {}".format(len(self.frames)),end="\r")
			wf.writeframes(b''.join(self.frames))
		if remove:
			self.frames = self.frames[self.shift_bytes:]
		print(f"Frames length is now {len(self.frames)}")
		self.saved=True

	def reset_save(self):
		self.saved = False
		
	def stop(self):
		'''
		To stop the server with ^C
		To do: Improvise
		'''
		last_time = time.time()
		while True:
			try:
				if time.time() - last_time > 10:
					last_time = time.time()
					self.reset_save()
				time.sleep(8)
			except KeyboardInterrupt as e:
				print("Exception in stop()",str(e))
				self.close()
				self.save()
				self.s.close()

	def send_command(self):
		'''
			Send a confirmation message that the 
			trigger word has been detected 
		'''
		msg = "RPi"
		self.send(msg.encode())
		#Now listen for the command and send the audio raw bytes
		self.listen_for_command()
	
	def listen_for_command(self):
		audio2send = []
		check_thresh = deque(maxlen=self.rel*self.silence_limit)
		prev_audio = deque(maxlen=self.rel*self.prev_audio_limit)
		started = False
		while True:
			current_audio = self.stream.read(chunk)
			check_thresh.append(np.sqrt(np.abs(audioop.avg(current_audio, 4))))
			if sum((i>self.threshold for i in check_thresh) > 0):
				audio2send.append(current_audio)
				started = True
			elif started:
				msg = "Command "
				for i in list(prev_audio)+audio2send:
					msg += i
				self.send(msg)
				break
			else:
				prev_audio.append(current_audio)


class Evaluate:
	def __init__(self,listen_object):
		self.sample = Sample(None,None,None,saved=True)
		Tx,freq = 5511,101 
		self.model = TModel(Tx,freq,*self.sample.load_dataset())
		self.model.load_model()
		self.listen_object = listen_object

	def continuously_analyze(self):
		while len(self.listen_object.frames) != 0:
			predictions = self.model.detect_triggerword("test.wav")
			if predictions is None:
				self.listen_object.reset_save()
				self.listen_object.save(remove=self.listen_object.closed)
			else:
				print("Got enough data resetting save",end="\r")
				self.listen_object.reset_save()
				self.listen_object.save()
				predictions = np.squeeze(predictions)
				continuous = 0
				print("Total ones",np.sum(predictions))
				for i in predictions:
					'''
					To do hyperparameter this to 50/75 contiguous ones to predict
					'''
					if i == 1:
						continuous += 1
					else:
						continuous = 0
					if continuous >= 50:
						print("Did you say lock?",end="\r")
						#pyautogui.press(["ctrl","c"])#(["win","shift","x"])
						self.listen_object.send_command()
						continuous = 0
			time.sleep(2)

def start_threads(l,e):
	print("Creating and starting threads")
	tPlay = Thread(target = l.play)
	tStop = Thread(target=l.stop)
	tAnalyze = Thread(target=e.continuously_analyze)
	tPlay.start()
	tStop.start()
	tAnalyze.start()

if __name__ == "__main__":
	
	form,channels,rate,chunk = pyaudio.paInt16,2,44100,1024*2
	p = pyaudio.PyAudio()

	stream = p.open(format=form,
					channels = channels,
					rate = rate,
					output = True,
					frames_per_buffer = chunk,
					)
	print("Initializing",end="\r")
	l = Listen(p,stream,form,chunk,channels,rate)
	e = Evaluate(l) 
	print("Done!	  ")
	start_threads(l,e)
