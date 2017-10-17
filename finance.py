import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from pandas_datareader.data import Options
from collect_news import getArticleHeadlines
from numpy import newaxis


# ================================
download_all_tickers_data = False
collect_current_headlines = False
single_ticker = '^GSPC'
tickers = ['AAPL', 'B', 'F', 'FB', 'GOOGL', 'AMZN', 'AAL', 'TSLA', 'SBUX', 'MSFT', 'NFLX', 'PYPL', 'GE', 'EBAY', 'BAC', 'JPM', 'GS', 'CCE', 'INTC', 'IBM', '^GSPC']
now = datetime.datetime.now()
start = datetime.datetime(2007, 1, 1)
end =  datetime.datetime(2017, 9, 15)
# =================================





def getCorporations(path):
	corporations = {}
	for _,corp in pd.read_csv(path).iterrows():
		corporations[corp['Ticker']] = corp['Name']
	return corporations





def getAllData(tickers, get_all = False, start=datetime.datetime(2007, 1, 1), end = datetime.datetime(2017, now.month, now.day)):
	if get_all == True:
		for ticker in tickers:
			file = 'datasets/{}.csv'.format(ticker)
			df = []
			if os.path.isfile(file):
				pass
			else:
				df =  web.DataReader(ticker, 'yahoo', start, end)
				df.to_csv('datasets/{}.csv'.format(ticker))





def getSentimentAnalysis(main_data, data):
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	import unicodedata
	sid = SentimentIntensityAnalyzer()
	for date, row in data.T.iteritems():
		try:
			sentence = unicodedata.normalize('NFKD', data.loc[date, 'Headlines']).encode('ascii','ignore')
			ss = sid.polarity_scores(sentence)
			main_data.set_value(date, 'compound', ss['compound'])
			main_data.set_value(date, 'neg', ss['neg'])
			main_data.set_value(date, 'neu', ss['neu'])
			main_data.set_value(date, 'pos', ss['pos'])
		except:
			data.loc[date, 'Headlines']
			pass
	return main_data






def getMainDataset(single_ticker=None, headlines=False, current_headlines = False):
	if single_ticker == None:
		print "Ticker symbold missing"
		exit()
	main_data = pd.DataFrame([])
	complete_file = 'datasets/{}_sa_complete.csv'.format(single_ticker)
	file = 'datasets/{}.csv'.format(single_ticker)
	if headlines == True:
		if os.path.isfile(complete_file):
			main_data = pd.read_csv(complete_file)
			main_data['Date'] = pd.to_datetime(main_data['Date'])
			main_data.set_index('Date', drop=True, inplace=True)
		else:
			df = pd.merge(pd.read_csv(file), getArticleHeadlines(collect_current=current_headlines), how='left', on=['Date'])
			df.dropna(axis=0, subset=['Headlines'], inplace=True)
			df['Date'] = pd.to_datetime(df['Date'])
			df.set_index('Date', drop=True, inplace=True)
			main_data = df.copy()
			data = df.dropna(axis=0, subset=['Headlines'])
			data['Headlines'] = data['Headlines'].apply(lambda x: unicode(x));
			main_data["compound"] = ''
			main_data["neg"] = ''
			main_data["neu"] = ''
			main_data["pos"] = ''
			main_data = getSentimentAnalysis(main_data, data)
			main_data.to_csv(complete_file)
	else:
		main_data = pd.read_csv(file)
		main_data['Date'] = pd.to_datetime(main_data['Date'])
		main_data.set_index('Date', drop=True, inplace=True)
	return main_data

 
