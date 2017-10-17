import sys, csv, json
import requests
import pandas as pd
import os
import datetime

key = '1115c367a16c43b18a532734d3d5998a'
now = datetime.datetime.now()
current_month = now.month
current_year = now.year
type_of_material_list = ['blog', 'brief', 'news', 'editorial', 'op-ed', 'list','analysis']
section_name_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health']
news_desk_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health', 'foreign']








# ====================================================
#       Format Data Frame to be date, headlines
# ====================================================
def getArticles(year, month):
	new_articles = list()
	with open('nytimes/{}-{:02}.json'.format(year, month)) as data_file: 
		data = json.load(data_file)
	articles = data["response"]["docs"]
	
	for d in range(1,32):
		date =  '{}-{:02}-{:02}'.format(year, month,d)
		headlines = list()

		for i in range(len(articles)):
			try:
				if any(substring in articles[i]['type_of_material'].lower() for substring in type_of_material_list):
					if any(substring in articles[i]['section_name'].lower() for substring in section_name_list):
						if date == articles[i]['pub_date'][:10]:
							headlines.append(str(articles[i]['headline']['main']))
			except:
				try:
					if any(substring in articles[i]['news_desk'].lower() for substring in news_desk_list):
						if date == articles[i]['pub_date'][:10]:
							headlines.append(str(articles[i]['headline']['main']))
				except:
					pass

		headlines = '. '.join(map(str,headlines))
		if len(headlines) > 0:
			new_articles.append({'Date':str(date), 'Headlines': headlines})
	return pd.DataFrame(new_articles)[['Date','Headlines']]
# ====================================================










# ====================================================
#           Download headlines from 10 years to now 
#           and create dataframes
# ====================================================
def getArticleHeadlines(collect_current=False):
	new_data = pd.DataFrame([{'Date':'', 'Headlines': ''}])
	
	if os.path.isfile('nytimes_csv/headlines.csv') and collect_current == False:
		new_data = pd.read_csv('nytimes_csv/headlines.csv')
		new_data.dropna(inplace=True)
	else:
		for year in range(2007, current_year+1):
			for month in range(1,13):
				file_str = 'nytimes/'+ str(year) + '-' + '{:02}'.format(month) + '.json'
				if os.path.isfile(file_str):
					pass
				else:
					url = 'https://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}'.format(year, month, key)
					print "getting data from {}/{}".format(month, year)
					if year == current_year and month > current_month:
						print "no more data"				
					else:
						data = requests.get(url).json()
						with open(file_str, 'w') as fout:
							json.dump(data, fout)
						fout.close()
				if year == current_year and month > current_month:
					print "no more data"
				else:	
					print "adding {}/{} to dataframe".format(year, month)
					new_data = new_data.append(getArticles(year, month), ignore_index=True)
					print "current rows {} ".format(new_data.shape[0])
		new_data = new_data[(new_data['Headlines'] != '') & (new_data['Date'] != '')]
		new_data.to_csv('nytimes_csv/headlines.csv', index=False)
	return new_data
	# ====================================================

