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
from model import TModel
from sample import Sample
from threading import Thread, Lock
import speech_recognition as speech

class Listen:
	host = "192.168.0.108"
	port = 30001
	def __init__(self, form=pyaudio.paInt16,chunk=1024,channels=1,
			shift_bytes=275, rate=16000, 
			threshold=10000, silence_limit=1, prev_audio_limit = 0.5):
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
		self.saving = Lock()
		self.stream = self.p.open(format=self.form,
					channels = self.channels,
					rate = self.rate,
					input = True,
					frames_per_buffer = self.chunk,
					)
		self.save_second = 1000
		self.frames = collections.deque(maxlen=self.rel*self.save_second)
		self.saved = False
		self.shift_bytes = shift_bytes
		self.closed = False
		self.recognizer = speech.Recognizer()

	def __exit__(self):
		self.close_client()
	def _send(self, msg):
		if isinstance(msg, str):
			msg = msg.encode()
		self.s.send(msg)
	
	def _receive(self, byte=1024):
		msg = self.s.recv(byte)
		if isinstance(msg, bytes):
			msg = msg.decode()
		return msg

	def receive(self):
		while True:
			msg = self._receive()
			if msg == "echo":
				self.trigger = True
			elif msg == "shutdown":
				self.closed = True
			print("\n", msg)

	def save(self,filename="test.wav"):
		'''
		Save the wav file. 
		'''
		while True:
			with self.saving:
				if self.saved or len(self.frames) == 0:
					continue
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
		while not self.closed:
			with self.saving:	
				data = self.stream_record()
				self.frames.append(data)
				print(f"Recording {len(self.frames)}", end="\r")
		self.close_client()

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
				time.sleep(10.1)
			except KeyboardInterrupt as e:
				print("Exception in stop()",str(e))
				self.save()
				self.s.close()
	def _convert_to_audio(self, wav_bytes: str, filename="command.wav") -> speech.AudioData:
		'''
		For now, inefficiently convert 
		raw bytes into wav file
		Then finally convert it into 
		speech.AudioFile object

		wav_bytes: string of raw bytes of audio 

		returns: speech.AudioFile compatible with speech_recognition library
		'''
		if isinstance(wav_bytes, str):
			wav_bytes = wav_bytes.encode()
		with wave.open(filename, 'wb') as wf:
			wf.setnchannels(self.channels)
			wf.setsampwidth(self.p.get_sample_size(self.form))
			wf.setframerate(self.rate)
			wf.writeframes(wav_bytes)

		audio_file = speech.AudioFile(filename)

		with audio_file as source:
			try:
				audio = self.recognizer.listen(source)
			except audioop.error as e:
				print(str(e))
				return None
		return audio
  
	def _convert_to_text(self, audio: speech.AudioData) -> str:
		'''
			audio: speech_recognition.AudioFile
  
			returns: text output from google api
		'''
		text = ""
		try:
			text = self.recognizer.recognize_google(audio)
		except speech.UnknownValueError as e:
			print(str(e))
		return  text	
	
	def convert_to_text(self, data):
		audio = self._convert_to_audio(data)
		text  = self._convert_to_text(audio)
		return text

	def detected(self):
		while True:
			data = self.listen_for_command(send = False)
			text = self.convert_to_text(data)
			print("\n", text, end='\r')
			if text.lower() == "echo":
				print("Detected a trigger word")
				self.frames.clear()
				self.send_command()
				self.trigger = False
			time.sleep(2)


	def send_command(self):
		'''
			Send a confirmation message that the 
			trigger word has been detected 
		'''
		msg = "RPi"
		self._send(msg)
		#Now listen for the command and send the audio raw bytes
		self.listen_for_command()
	
	def listen_for_command(self, send = True):
		audio2send = []
		check_thresh = collections.deque(maxlen=self.rel*self.silence_limit)
		prev_audio = collections.deque(maxlen=int(self.rel*self.prev_audio_limit))
		started = False
		while True:
			while len(self.frames) == 0:
				pass
			current_audio = self.frames.pop()#self.stream_record()
			check_thresh.append(np.sqrt(np.abs(audioop.avg(current_audio, 4))))
			val = sum([i>self.threshold for i in check_thresh]) 
			if val > 0:
				audio2send.append(current_audio)
				started = True
			elif started:
				msg = b"Command " + b"".join(list(prev_audio) + audio2send)
				if send:
					self._send(msg)
					self._send(b"End")
					break
				else:
					return msg
			else:
				prev_audio.append(current_audio)

	def close_client(self):
		try:
			print("Alright goodbye!")
			self._send("End")
			self.stream.stop_stream()
			self.stream.close()
			self.p.terminate()
			self.s.close()
		except Exception as e:
			print(str(e))


class Evaluate:
	def __init__(self,listen_object):
		self.sample = Sample(None,None,None,saved=True)
		Tx,coeff = self.sample.Tx, self.sample.coeff
		#self.model = TModel(Tx,coeff, "../dataset.h5")
		self.listen_object = listen_object

	def continuously_analyze(self):
		while len(self.listen_object.frames) != 0:
			continue
			predictions = self.model.detect_triggerword("test.wav")
			if predictions is None:
				self.listen_object.reset_save()
			else:
				print("Got enough data resetting save",end="\r")
				self.listen_object.reset_save()
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
	tReceive = Thread(target=l.receive) 

	tRecord.start()
	tStop.start()
	tSend.start()
	tReceive.start()

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
	e = Evaluate(l) 
	print("Done!	  ")
	start_threads(l,e)
	#l.send_command()
