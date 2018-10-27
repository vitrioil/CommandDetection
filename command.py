import sqlite3
import datetime
import requests
import importlib
from lxml import html
from weather import Weather, Unit
from bs4 import BeautifulSoup as BS
from evalVoiceData import StopEchoException
news_websites = "http://timesofindia.indiatimes.com"
#os.system("weather mumbai --u c")

lib = importlib.import_module("test_dict")
def command_weather(loc = "mumbai", day = 1):
	'''
		Command: Show me the current weather

		Gets the current weather
	'''
	output = ""
	day = min(day, 5)
	w = Weather(unit = Unit.CELSIUS)
	location = w.lookup_by_location(loc)
	if location is None:
		print("Location not found")
		return
	forecast = location.forecast

	for f in forecast[:day]:
		output += "Forecast for " + str(f.date) + " is " + f.text + '\n'
		output += f.low + '\n'
		output += f.high + '\n'
		output += "="*10
		output += "\n"
	return output

def command_show_time():
	'''
		Prints the current date and time
	'''
	output = ""
	now = datetime.datetime.now()
	output += "\n\n"
	output += now.strftime("It is %A, %d %B of %Y , at %H hours and %M minutes")
	output += "\n\n"
	return output


def command_good_morning():
	output = ""
	now = datetime.datetime.now()
	if now.hour > 18:
		output += "\n\nGood evening\n\n"
	elif now.hour > 12:
		output += "\n\nGood afternoon\n\n"
	else:
		output += "\n\nGood morning\n\n"
	return output

def command_show_all_commands():
	output = ""
	try:
		con = sqlite3.connect("Commands.db")
		cur = con.cursor()
	except Exception as e:
		print(str(e))

	cur.execute("select command from commands")
	output += "Predefined commands" + '\n'
	commands = [i[0] for i in cur.fetchall()]
	for c in commands:
		output += c + '\n'

	output += '\n'
	cur.execute("select command from user_commands")
	output += "User defined commands" + '\n'
	commands = [i[0] for i in cur.fetchall()]
	for c in commands:
		output += c + '\n'
	try:
		cur.close()
		con.close()
	except Exception as e:
		print(str(e))
	return output

def command_goodbye():
	output = "\n\nGoodbye!\n\n"
	raise StopEchoException("Good bye user!")
