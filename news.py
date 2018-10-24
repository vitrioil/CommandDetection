import requests
from lxml import html
from bs4 import BeautifulSoup as BS
news_websites = "http://timesofindia.indiatimes.com"

def command_scrape_from_toi():
	page = requests.get(news_websites)

	soup = BS(page.content, "html.parser")

	news = soup.find(class_ = 'top-story')
	list_of_headlines = news.find_all("li")
	print("\n\nTop news of the day:\n\n")
	for i in list_of_headlines:
		headline = i.find('a')
		print(headline.text)
		print('-'*50)

if __name__ == "__main__":
	scrape_from_toi()
