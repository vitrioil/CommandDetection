#!/bin/python3.7
import sys
import argparse
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
parser.add_argument("--command-name", help="Command name to be added or removed")
parser.add_argument("--client", help="Run as client")
parser.add_argument("--server", help="Run as server")
args = parser.parse_args()

class Echo:
	
	def __init__(self, args):
		self.args = args
		if self.args.client:
			self.client()
		elif self.args.server:
			self.notify = Notify()
			self.analyze = Analyze(notify)
		self.analyze = Analyze(None)
		if self.args.add and (self.args.filename and self.args.function):
			self.analyze.command_add_command(self.args.filename, self.args.function, self.args.command-name)

		if self.args.remove and (self.args.command-name):
			self.analyze.command_add_command(self.args.command-name)

	def client(self):
		form,channels,rate,chunk = pyaudio.paInt16, 1, 16000, 1024*2
		
		print("Initializing",end="\r")
		
		self.listen = Listen(form,chunk,channels,rate)
		self.evaluate = Evaluate(self.listen) 
		
		print("Done!      ")
		
		start_threads(l,e)
		
		print("Creating and starting threads")
		
		tRecord = Thread(target=self.listen.record)
		tStop = Thread(target=self.listen.stop)
		tSend = Thread(target=self.listen.detected)
		tReceive = Thread(target=self.listen.receive) 
		tAnalyze = Thread(target=self.evaluate.continuously_analyze)
		tSave = Thread(target=self.listen.save)

		tRecord.start()
		tSave.start()
		tStop.start()
		tSend.start()
		tReceive.start()
		tAnalyze.start()


	def server(self):
		main_thread = threading.Condition()
		main_thread.acquire()

		t = threading.Thread(target=self.notify.acquire_and_notify)
		t.start()
		time.sleep(1)
		self.analyze.wait_and_check()

if __name__ == "__main__":
	echo = Echo(args)
