import wave
import time
import socket
import pyaudio
import sqlite3
import threading
import numpy as np
from customWMD import WMD,WordNotFound
import speech_recognition as speech
from sklearn.externals.joblib import Parallel, delayed


def connect():
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind(ip_port)
	s.listen(1)
	print("Waiting for connection")
	con, addr = s.accept()
	print("Connected")
	return con, addr, s

class Notify:

	ip_port = ("", 30001)
	def __init__(self):
		self._connect()
	
	def _connect(self):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.s.bind(self.ip_port)
		self.s.listen(1)
		print("Waiting for connection")
		self.con, self.addr = self.s.accept()
		print("Connected")
		
	def _close(self):
		self.con.close()
	
	def _send(self, msg):
		self.con.send(msg)

	def _recv(self, byte=1024, wtf = "Not notify"):
		msg = self.con.recv(byte)
		if isinstance(msg, bytes):
			msg = msg.decode()
		return msg

	def acquire_and_notify(self):
		'''
			Acquire the condition variable `main_thread`
			and notify the analyze object.

			Assuming for now msg starting with `RPi`
			means RPi has detected a trigger word
		'''

		try:
			with main_thread:
				main_thread.acquire()
				print("Acquired back again")
				msg = self._recv(wtf="Notify")
				print(msg,"in acq and not")
				print(msg.startswith("RPi"))
				print("Notifying")
				main_thread.notify()
				print("Notified")
		except socket.error as se:
			print("Socket error {}".format(se))
			self._close()
			#self._connect()
		except Exception as e:
			print(str(e))
			self._close()

class Analyze:
	
	def __init__(self, notify,form=pyaudio.paInt16,chunk=1024*2,channels=2, 
			shift_bytes=275, rate=16000):
		self.notify = notify
		self.form = form
		self.chunk = chunk
		self.channels = channels
		self.shift_bytes = shift_bytes
		self.rate = rate
		self.conn = sqlite3.connect("Command.db")
		self.curr = self.conn.cursor()
		self.recognizer = speech.Recognizer()
		self.wmd = WMD()
		self.p = pyaudio.PyAudio()
		self._retrieve_commands()

	def _get_from_raspberry(self) -> str:
		'''
			After notified get the raw bytes from 
			RPi.

			returns: string of bytes received
		'''
		msg = ""
		try:
			msg = self.notify._recv(1024)
			check_str, msg = msg.split()
			if not check_str == "Command":
				return ""
		except socket.error as e:
			print(str(e))
			self.notify._close()
		except Exception as e:
			print(str(e))
			self.notify._close()
		finally:
			return msg 

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
		print(wav_bytes)
		with wave.open(filename, 'wb') as wf:
			wf.setnchannels(self.channels)
			wf.setsampwidth(self.p.get_sample_size(self.form))
			wf.setframerate(self.rate)
			wf.writeframes(wav_bytes)

		audio_file = speech.AudioFile(filename)
		with audio_file as source:
			audio = self.recognizer.listen(source)

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
		return 	text
	def _retrieve_commands(self) -> list:
		'''
		returns: all commands from database
		'''
		self.curr.execute("SELECT command from commands")
		self.commands = self.curr.fetchall()

	def _find_best(self, user_command: str, n_jobs=10) -> (np.float64, str):
		'''
		user_command: str is the command that user gave as input
		n_jobs: int for parallelizing

		returns: tuple (wmd calculated, command in the database)
		'''
		try:
			out = Parallel(n_jobs=n_jobs)(delayed(lambda x: (self.wmd.wmd(user_command, x), x)) for command in self.commands)
			out.sort(key = lambda x: x[0])
		except WordNotFound as e:
			print(str(e))
			self._send(str(e))

		return out[0]
	
	def _send(message: str):
		'''
			Send a message to RPi
		'''
		try:
			self.notify._send(message)
		except Exception as e:
			print(str(e))

	def perform(self, command: str):
		'''
			Perform the given command 
		'''
		pass

	def wait_and_check(self):
		'''
			function which wakes up when command is given to the user 
		'''
		
		while True:
			self.notify.acquire_and_notify()
			print("Acquired")
			msg = self._get_from_raspberry()
			print(f"Got {msg} from raspberry")
			audio = self._convert_to_audio(msg)
			print("Made an audio file :D")
			#text = self._convert_to_text(audio)
			print("Random text :D")
			#(distance, command) = self._find_best(text)

			#self.perform(command)

if __name__ == "__main__":
	main_thread = threading.Condition()
	main_thread.acquire()

	notify  = Notify()
	analyze = Analyze(notify)
	t = threading.Thread(target=notify.acquire_and_notify)
	t.start()
	time.sleep(1)	
	analyze.wait_and_check()
