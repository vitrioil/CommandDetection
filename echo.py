#!/bin/python3.7
import sys
import time
import pyaudio
import argparse
import threading 
from sample import Sample
from evalVoiceData import Analyze, Notify
from client_audio import Listen, Evaluate
class StoreNameValuePair(argparse.Action):

	def __call__(self, parser, namespace, values, option_string=None):
		n, v = values.split('=')
		setattr(namespace, n, v)

parser = argparse.ArgumentParser()

parser.add_argument("--add", help="When enabled will add a command")
parser.add_argument("--remove", help="When enabled will remove a command")
parser.add_argument("--filename", help="Filename where command is located") 
parser.add_argument("--function", help="Function which is to be executed")
parser.add_argument("--command_name", help="Command name to be added or removed")
parser.add_argument("--client", help="Run as client")
parser.add_argument("--server", help="Run as server")
args = parser.parse_args()

class Echo:
	
	def __init__(self, args):
		self.args = args
		if self.args.client:
			self.client()
		elif self.args.server:
			self.notify = Notify(main_thread)
			self.analyze = Analyze(main_thread, self.notify)
			self.server()
		if self.args.add and (self.args.filename and self.args.function):
			self.analyze = Analyze(main_thread, None)
			self.analyze.command_add_command(self.args.function, self.args.filename, self.args.command_name)

		if self.args.remove and (self.args.command_name):
			self.analyze = Analyze(main_thread, None)
			self.analyze.command_remove_command(self.args.command_name)

	def client(self):
		form,channels,rate,chunk = pyaudio.paInt16, 1, 16000, 1024*2
		
		print("Initializing...")
		
		self.listen = Listen(form,chunk,channels,rate)
		self.evaluate = Evaluate(self.listen) 
		
		print("Done!")
		
		print("Loading...")
		
		tRecord = threading.Thread(target=self.listen.record)
		tStop = threading.Thread(target=self.listen.stop)
		tSend = threading.Thread(target=self.listen.detected)
		tReceive = threading.Thread(target=self.listen.receive) 
		tAnalyze = threading.Thread(target=self.evaluate.continuously_analyze)
		tSave = threading.Thread(target=self.listen.save)

		tRecord.start()
		tSave.start()
		tStop.start()
		tSend.start()
		tReceive.start()
		tAnalyze.start()


	def server(self):

		t = threading.Thread(target=self.notify.acquire_and_notify)
		t.start()
		time.sleep(1)
		self.analyze.wait_and_check()

def hook(f, *_):
	print(f.f_code.co_name)

#sys.setprofile(hook)

if __name__ == "__main__":
	main_thread = threading.Condition()
	main_thread.acquire()
	echo = Echo(args)
