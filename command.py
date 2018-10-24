import datetime
import requests
import importlib
from lxml import html
from weather import Weather, Unit
from bs4 import BeautifulSoup as BS
news_websites = "http://timesofindia.indiatimes.com"
#os.system("weather mumbai --u c")

lib = importlib.import_module("test_dict")
print(dir(lib))
def command_weather(loc = "mumbai", day = 1):
	'''
		Command: Show me the current weather

		Gets the current weather
	'''
	day = min(day, 5)
	w = Weather(unit = Unit.CELSIUS)
	location = w.lookup_by_location(loc)
	if location is None:
		print("Location not found")
		return
	forecast = location.forecast

	for f in forecast[:day]:
		print("Forecast for", f.date, "is", f.text)
		print(f.low)
		print(f.high)
		print("="*10)

def command_scrape_from_toi():
	'''
		Command: Show me top news of the day

		Shows top news from Times of India
	'''
	page = requests.get(news_websites)

	soup = BS(page.content, "html.parser")

	news = soup.find(class_ = 'top-story')
	list_of_headlines = news.find_all("li")

	for indx, news in enumerate(list_of_headlines):
		headline = news.find('a')
		print(indx, "==>", headline.text)

def command_show_time():
	'''
		Prints the current date and time
	'''
	now = datetime.datetime.now()
	print("\n\n")
	print(now.strftime("It is %A, %d %B of %Y , at %H hours and %M minutes"))
	print("\n\n")
def command_increase_volume():
	pass

def command_good_morning():
	pass

def command_show_all_commands():
	print("\n\nPredefined commands are:")
	for i in dir():
		if i.startswith("command"):
			print(i[len("command_"):])
			print('-'*20)

def command_reminders():
	pass

def command_goodbye():
	pass
