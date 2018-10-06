'''



self.frames : list -> deque

'''
import wave
import time
import socket
import audioop
import pyaudio
import collections
import numpy as np
from queue import Queue
#from model import TModel
#from sample import Sample
from threading import Thread

class Listen:
	host = "192.168.0.108"
	port = 30001
	def __init__(self, form=pyaudio.paInt16,chunk=1024,channels=1,
			shift_bytes=275, rate=16000, 
			threshold=2000, silence_limit=1, prev_audio_limit = 0.5):
		self.form = form
		self.chunk = chunk
		self.channels = channels
		self.rate = rate
		self.rel = self.rate//self.chunk
		self.threshold = threshold
		self.silence_limit = silence_limit
		self.prev_audio_limit = prev_audio_limit
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.s.connect((self.host, self.port))
		self.q = Queue()
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format=self.form,
			channels = self.channels,
			rate = self.rate,
			input = True,
			frames_per_buffer = self.chunk,
			)
		self.save_second = 10
		self.frames = collections.deque(maxlen=self.rel*self.save_second)
		self.saved = False
		self.shift_bytes = shift_bytes
		self.closed = False

	def __exit__(self):
		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()

	def _send(self, msg):
		if isinstance(msg, str):
			msg = msg.encode()
		self.s.send(msg)

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
		print(f"Frames length is now {len(self.frames)}")
		self.saved=True

	def reset_save(self):
		self.saved = False
	
	def stream_record(self):
		return self.stream.read(self.chunk)

	def record(self):
		while True:
			self.frames.append(self.stream_record())
			print(f"Recording {len(self.frames)}", end="\r")

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
				self.save()
				self.s.close()
	
	def detected(self):
		while True:
			self.send_command()
			time.sleep(5)

	def send_command(self):
		'''
			Send a confirmation message that the 
			trigger word has been detected 
		'''
		msg = "RPi"
		self._send(msg)
		time.sleep(2)
		#Now listen for the command and send the audio raw bytes
		self.listen_for_command()
	
	def listen_for_command(self):
		audio2send = []
		check_thresh = collections.deque(maxlen=self.rel*self.silence_limit)
		prev_audio = collections.deque(maxlen=int(self.rel*self.prev_audio_limit))
		print(self.rel*self.silence_limit,"||||", self.rel*self.prev_audio_limit)
		started = False
		while True:
			while len(self.frames) == 0:
				pass
			current_audio = self.frames.pop()#self.stream_record()
			check_thresh.append(np.sqrt(np.abs(audioop.avg(current_audio, 4))))
			if sum([i>self.threshold for i in check_thresh]) > 0:
				audio2send.append(current_audio)
				started = True
			elif started:
				print("Sending {} bytes".format(len(list(prev_audio)+audio2send)))
				msg = b"Command " + b"".join(list(prev_audio) + audio2send)
				self._send(msg)
				time.sleep(1)
				self._send(b"End")
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
	tRecord = Thread(target=l.record)
	tStop = Thread(target=l.stop)
	tSend = Thread(target=l.detected)
	#tAnalyze = Thread(target=e.continuously_analyze)
	tRecord.start()
	tStop.start()
	tSend.start()
	#tAnalyze.start()

li = list(dir(Listen)) + list(dir(Evaluate))
def hook(f, *_):
	global li
	if True:#f.f_code.co_name in li and f.f_code.co_name != "__init__": 
		print(f.f_code.co_name)
import sys
#sys.setprofile(hook)
if __name__ == "__main__":
	
	form,channels,rate,chunk = pyaudio.paInt16, 1, 16000, 1024*2
	print("Initializing",end="\r")
	l = Listen(form,chunk,channels,rate)
	e = None#Evaluate(l) 
	print("Done!	  ")
	start_threads(l,e)
	#l.send_command()
