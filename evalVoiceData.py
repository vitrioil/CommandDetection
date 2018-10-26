import wave
import time
import pickle
import socket
import audioop
import pyaudio
import sqlite3
import colorama
import threading
import importlib
import numpy as np
from customWMD import WMD,WordNotFound
import speech_recognition as speech
from sklearn.externals.joblib import Parallel, delayed

if __name__ == "__main__":
	from command import *
	import command

class StopEchoException(Exception):
	
	def __init__(self, msg, error=""):
		super().__init__(msg)
		self.msg = msg
		self.error = error

	def __str__(self):
		return str(self.msg)

def _find_command(wmd, user_command, command):
	distance = wmd.wmd(user_command, command)

	return (distance, command)
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
	def __init__(self, main_thread):
		self.main_thread = main_thread
		self._connect()
		self.closed = False

	def set_analyze(self, analyze):
		self.analyze = analyze

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
		if isinstance(msg, str):
			msg = msg.encode()
		print("Sending {}".format(msg))
		self.con.send(msg)

	def _recv(self, byte=1024):
		msg = self.con.recv(byte)
		#if isinstance(msg, bytes):
		#	msg = msg.decode()
		return msg

	def acquire_and_notify(self):
		'''
			Acquire the condition variable `main_thread`
			and notify the analyze object.

			Assuming for now msg starting with `RPi`
			means RPi has detected a trigger word
		'''
		while not self.closed:
			try:
				with self.main_thread:
					print("Acquired back again")
					msg = self._recv()
					print(msg[:3],"in acq and not")
					print("Notifying")
					self.main_thread.notify()
			except socket.error as se:
				print("Socket error {}".format(se))
				self._close()
				break
				#self._connect()
			except Exception as e:
				print(str(e))
				self._close()
			time.sleep(1)
	
	def close_server(self):
		self.closed = True

class Analyze:
	
	def __init__(self, main_thread, notify,form=pyaudio.paInt16,chunk=1024,channels=1, 
			shift_bytes=275, rate=48000):
		self.notify = notify
		self.main_thread = main_thread
		if self.notify is None:	
			print("Will not connect to RPi")
		self.form = form
		self.chunk = chunk
		self.channels = channels
		self.shift_bytes = shift_bytes
		self.rate = rate
		self.conn = sqlite3.connect("Commands.db")
		self.curr = self.conn.cursor()
		self.recognizer = speech.Recognizer()
		self.wmd = WMD()
		self.p = pyaudio.PyAudio()
		self._retrieve_commands()
		with open("words.pickle", 'rb') as f:
			self.words = pickle.load(f)

	def _get_from_raspberry(self) -> str:
		'''
			After notified get the raw bytes from 
			RPi.

			returns: string of bytes received
		'''
		audio_bytes = b""
		try:
			while True:
				msg = self.notify._recv(4096)
				if msg.startswith(b"Command "):
					check_str, msg = msg[:len(b"Command ")], msg[len(b"Command "):]
					audio_bytes += msg
				elif msg.endswith(b"End"):
					break
				else:
					audio_bytes += msg
		except socket.error as e:
			print(str(e))
			self.notify._close()
		except Exception as e:
			print(str(e))
			self.notify._close()
		finally:
			return audio_bytes 

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
		return 	text

	def _retrieve_commands(self):
		'''
		retrieves all commands from database
		'''
		self._execute("select indx, command, func from commands") 
		self.commands_table = self.curr.fetchall()

		self._execute("select indx, command, func, file_name from user_commands") 
		self.user_commands_table = self.curr.fetchall()

		self.commands = [i[1] for i in self.commands_table]
		self.commands.extend([i[1] for i in self.user_commands_table])

		self.command_to_function = {i[1]: i[2] for i in self.commands_table}
		self.user_command_to_functon = {i[1]: (i[2], i[3]) for i in self.user_commands_table}

	def _find_best(self, user_command: str, n_jobs=5) -> (np.float64, str):
		'''
		user_command: str is the command that user gave as input
		n_jobs: int for parallelizing

		returns: tuple (wmd calculated, command in the database)
		'''
		out = []
		try:
			out = Parallel(n_jobs=n_jobs)(delayed(_find_command)(self.wmd, user_command, command) for command in self.commands)
			out.sort(key = lambda x: x[0])
		except WordNotFound as e:
			print(str(e))
			self._send(str(e))
		if not isinstance(out, list) or len(out) == 0:
			return (np.inf, "")
		return out[0]
	
	def _send(self, message: str):
		'''
			Send a message to RPi
		'''
		try:
			self.notify._send(message)
		except Exception as e:
			print(str(e))

	def _execute(self, query: str):
		'''
			Execute a query
		'''
		try:
			self.curr.execute(query)
		except sqlite3.Error as e:
			print(str(e))

	def _commit(self):
		'''	
			Commit to a database
		'''
		try:
			self.conn.commit()
		except sqlite3.Error as e:
			print(str(e))

	def _check_command(self, command):
		'''
			Checks the command during insertion if it already exists
		'''
		self._execute("select command from commands where command = '{}'".format(command))
		result = [i[0] for i in self.curr.fetchall()]
		self._execute("select command from user_commands where command = '{}'".format(command)) 
		result += [i[0] for i in self.curr.fetchall()]
		if result == []:
			return False
		return True

	def _check_vocab(self, word):
		vocab_len = len(self.words)
		low, mid, high = 0, len(self.words) // 2, vocab_len 
		while low <= high:
			mid = (low + high) // 2
			if word == self.words[mid]:
				return True
			elif word < self.words[mid]:
				high = mid - 1
			else:
				low = mid + 1
			print(vocab_len, mid, end="\r")
		return False

	def perform(self, command: str, *args, **kwargs):
		'''
			Perform the given command 
		'''
		function_name = self.command_to_function.get(command)
		file_name = "command"
		if function_name is None:
			f = self.user_command_to_functon.get(command)
			if f is None: 
				print("Unexpected error, wrong input command")
				return
			function_name, file_name  = f
		print("\nLoading {} for {}".format(file_name, function_name))
		lib = importlib.import_module(file_name)
		func = lib.__dict__[function_name]
		try:
			output = func(*args, **kwargs)
			self._send(output)
		except StopEchoException as e:
			print(str(e))
			self.close_server()
		except Exception as e:
			print(str(e))
		
	def wait_and_check(self):
		'''
			function which wakes up when command is given to the user 
		'''
		while not self.notify.closed:
			with self.main_thread:
				self.main_thread.wait()
				print("Notified")
				msg = self._get_from_raspberry()
				if len(msg) == 0:
					self._send("Didn't receive any command")
					continue
				audio = self._convert_to_audio(msg)
				if audio is None:
					self._send("Sorry! Audio byte stream issue could you say that again please?")
					continue
				print("Made an audio file :D")
				text = self._convert_to_text(audio)
				print(f"You said: {text} :D")
				if len(text) == 0:
					self._send("Couldn't convert it into text! Can you repeat?")
					continue
				(distance, command) = self._find_best(text)
				print("Closest command to {} {} {} is {} {} {} at a distance of {} {} {} ".format(
						colorama.Fore.GREEN, text, colorama.Style.RESET_ALL, colorama.Fore.RED, command, 
						colorama.Style.RESET_ALL, colorama.Fore.YELLOW,  distance, colorama.Style.RESET_ALL)
				)

				self.perform(command)
	
	def command_add_command(self, function_name, file_name, command_name = ""):
		'''
		   Command: Add a new command
		
		   Special function that can add a function and a command
		'''
		if file_name.endswith(".py"):
			file_name = file_name[:-len(".py")]
		if command_name == "" or command_name is None:
			print(function_name)
			assert function_name.startswith("command_"), "Enter command name or name the function as command_{command_name}"
			command_name = function_name[len("command_"):] 
			command_name = " ".join(command_name.split('_'))
		for word in command_name.split():
			if not self._check_vocab(word):
				print("Word {} not in vocab".format(word))
				print("Failed to add the command")
				return

		print("Checking for {}".format(command_name))
		check = self._check_command(command_name)
		if check and check is not None:
			print("Command: {} is already present".format(command_name))
			return
		self._execute("insert into user_commands (command, func, file_name) values('{}', '{}', '{}')".format(command_name, function_name, file_name))
		print("executed")
		self._commit()
		self._retrieve_commands()
		print(self.user_command_to_functon)

	def command_remove_command(self, command_name):
		'''
		   Command: Add a new command
		
		   Special function that can add a function and a command
		'''
		print("Deleting {}".format(command_name))
		self._execute("delete from user_commands where command = '{}'".format(command_name))
		self._commit()
		self._retrieve_commands()
		print(self.user_command_to_functon)
	
	def close_server(self):
		self._send("shutdown")
		self.curr.close()
		self.conn.close()
		self.p.terminate()
		try:
			self.notify.close_server()
		except Exception as e:
			print(str(e))


l = dir(Notify) + dir(Analyze)
def hook(f, *_):
        if f.f_code.co_name in l and f.f_code.co_name != "__init__":
                print(f.f_code.co_name)
import sys
#sys.setprofile(hook)
if __name__ == "__main__":
	main_thread = threading.Condition()
	main_thread.acquire()

	notify  = Notify(main_thread)
	analyze = Analyze(main_thread, notify)
	t = threading.Thread(target=notify.acquire_and_notify)
	t.start()
	time.sleep(1)	
	analyze.wait_and_check()
