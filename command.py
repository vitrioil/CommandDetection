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
print(dir(lib))
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
		output += "Forecast for " + str(f.date) + " is " + f.text
		output += "\n"
		output += f.low
		output += "\n"
		output += f.high
		output += "="*10
		output += "\n"
	return output

def command_scrape_from_toi():
	'''
		Command: Show me top news of the day

		Shows top news from Times of India
	'''
	output = ""
	page = requests.get(news_websites)

	soup = BS(page.content, "html.parser")

	news = soup.find(class_ = 'top-story')
	list_of_headlines = news.find_all("li")

	for indx, news in enumerate(list_of_headlines):
		headline = news.find('a')
		output += indx + " ==> " + headline.text
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

def command_increase_volume():
	pass

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
	output += "\n\nPredefined commands are:"+"\n"
	for i in dir():
		if i.startswith("command"):
			output += i[len("command_"):]
			output += '-'*20
			output += '\n'
	return output

def command_reminders():
	pass

def command_goodbye():
	output = "\n\nGoodbye!\n\n"
	raise StopEchoException("Good bye user!")
