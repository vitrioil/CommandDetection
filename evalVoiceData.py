import sqlite3
import threading
import numpy as np
from customWMD import WMD,WordNotFound
import speech_recognition as speech
from sklearn.externals.joblib import Parallel, delayed

main_thread = threading.Condition()
main_thread.acquire()

ip_port = ("", 30002)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(ip_port)
s.listen(1)

class Notify:
	
	def acquire_and_notify(self):
		'''
			Acquire the condition variable `main_thread`
			and notify the analyze object.

			Assuming for now msg starting with `RPi`
			means RPi has detected a trigger word
		'''

		while True:
			try:
				msg = s.recv(1024)
				msg = msg.decode()
				if not msg.startswith("RPi"):
					continue
				main_thread.acquire()
				main_thread.notify()
				main_thread.release()

			except Exception as e:
				print(str(e))

class Analyze:
	
	def __init__(self):
		self.conn = sqlite.connect("Command.db")
		self.curr = self.conn.cursor()
		self.recognizer = speech.Recognizer()
		self.wmd = WMD()
		self.thread = threading.Condition()

	def _get_from_raspberry(self) -> str:
		'''
			After notified get the raw bytes from 
			RPi.

			returns: string of bytes received
		'''
		msg = ""
		try:
			msg = s.recv(1024)
			msg = msg.decode()
		except Exception as e:
			print(str(e))
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
		with wave.open(filename, 'wb') as wf:
			wf.setnchannels(self.channels)
			wf.setsampwidth(self.p.get_sample_size(self.form))
			wf.setframerate(self.rate)
			print("Saving bytes {}".format(len(self.frames)),end="\r")
			wf.writeframes(b''.join(wav_bytes))

		audio_file = speech.AudioFile(filename)

		with audio_file as source:
			audio = self.recognizer.listen(source)

		return audio 
	
	def _convert_to_text(self, audio: speech.AudioData) -> str:
		'''
		audio: speech_recognition.AudioFile

		returns: text output from google api
		'''
		return self.recognizer.recognize_google(audio)
	
	def _retrieve_commands(self) -> list:
		'''
		returns: all commands from database
		'''
		self.curr.execute("SELECT command from commands")
		commands = self.curr.fetchall()

		return commands

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
			self.send(str(e))

		return out[0]
	
	def send(message: str):
		'''
			Send a message to RPi
		'''
		try:
			s.send(message)
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
			main_thread.wait()
			msg = self._get_from_raspberry()
			audio = self._convert_to_audio(msg)
			text = self._convert_to_text(audio)
			(distance, command) = self._find_best(text)

			self.perform(command)

if __name__ == "__main__":
	notify  = Notify()
	analyze = Analyze()
	t = threading.Thread(target=notify.acquire_and_notify)
	t.start()
	
	analyze.wait_and_check()

