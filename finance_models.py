from finance import getAllData, getMainDataset
from collect_news import getArticleHeadlines
import matplotlib.pyplot as plt
import datetime, math
import numpy as np
import pandas as pd
# Machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# ================================
days_in_advanced = 5
download_all_tickers_data = False
collect_current_headlines = False
single_ticker = '^GSPC'
tickers = ['AAPL', 'B', 'F', 'FB', 'GOOGL', 'AMZN', 'AAL', 'TSLA', 'SBUX', 'MSFT', 'NFLX', 'PYPL', 'GE', 'EBAY', 'BAC', 'JPM', 'GS', 'CCE', 'INTC', 'IBM', '^GSPC']
start = datetime.datetime(2007, 1, 1)
end =  datetime.datetime(2017, 9, 15)
# =================================




def train_test(data, ratio=0.2):
	train_ratio = data.shape[0] - int(math.ceil(data.shape[0] * ratio))
	train = data.iloc[:train_ratio]
	test  = data.iloc[train_ratio:]
	return [train, test]	





def split_train_test(train, test, labels= [], features=[]):
	X_train = train[features]
	X_test = test[features] 
	y_train = train[labels]
	y_test = test[labels]
	return [X_test, y_test, X_train, y_train]




def train_test_data_model(model, data, plot=False, shuffle=True, test_size = 0.2, random_state=0, labels=[], features=[], moving_average=False):
	if shuffle == True:
		X_train, X_test, y_train, y_test = train_test_split(data[features],  data[labels], test_size=test_size, random_state=random_state)
	else:
		train, test = train_test(data, ratio=test_size)
		X_test, y_test, X_train, y_train = split_train_test(train, test, labels=labels, features=features)

	print model
	print len(name) * '-'

	model.fit(X_train, y_train['predicted_stock_price'])
	prediction = model.predict(X_test)
	
	r2 = r2_score(y_test, prediction)
	mean_absolute_error = (y_test, prediction)

	print "R^2 score: {}".format(r2)
	print "Mean Absolute Error score: {}".format(mean_absolute_error)

	prediction = pd.DataFrame(prediction, index=y_test.index.values, columns=['predicted_stock'])


	test_dates_prices = y_test.copy()
	message_label = "Model Predicted Stock Prices with unshuffled data"
	title  = "{} Unshuffled Data Prediction".format(name)

	if shuffle == True:
		prediction.sort_index(inplace=True)
		test_dates_prices.sort_index(inplace=True)
		message_label = "Model Predicted Stock Prices with shuffled data"
		title  = "{} Shuffled Data Prediction".format(name)
	if moving_average == True:
		prediction['moving_average_predicted_stock'] = prediction['predicted_stock'].rolling(window=days_in_advanced,center=False).mean()

	if plot==True:
		plt.figure(figsize=(16,10))
		plt.plot(prediction['predicted_stock'].loc['2017-01-01':], label=message_label)
		if moving_average == True:
			plt.plot(prediction['moving_average_predicted_stock'].loc['2017-01-01':], label="Predicted stock price moving average")
		plt.plot(test_dates_prices.loc['2017-01-01':], label="Actual Prices")
		plt.title(title)
		plt.ylabel("Stock Prices")
		plt.xlabel("Dates 2017")
		plt.legend(loc=0)
		plt.show()

	return model, score, prediction, test_dates_prices



# =========================================================================
# Collect all the data from the ticker list
# =========================================================================
getAllData(tickers, get_all=download_all_tickers_data, start=start, end=end)
# ==========================================================================

data = getMainDataset(single_ticker=single_ticker, headlines=True, current_headlines=collect_current_headlines)
data['high_low_volatility_pct'] = ((data['High'] - data['Adj Close']) / data['Adj Close']) * 100
data['daily_pct_change'] = ((data['Adj Close'] - data['Open']) / data['Open']) * 100
data['predicted_stock_price'] = data['Adj Close'].shift(-days_in_advanced)
data = data[:-days_in_advanced]

print data.shape[0]


# =============================================
# 				No headlines
# =============================================
chosen_cols = ['Adj Close', 'high_low_volatility_pct', 'daily_pct_change', 'Volume', 'predicted_stock_price']
df_no_headlines = data[chosen_cols]
feature_cols = ['Adj Close', 'high_low_volatility_pct', 'daily_pct_change', 'Volume']
label_cols = ['predicted_stock_price']
# =============================================

print "Data Frame with no headlines"
print df_no_headlines.head()


# # Shuffled
# RFModel = RandomForestRegressor(random_state=42)
# model, score, prediction, test_dates_prices = train_test_data_model(RFModel, df_no_headlines, plot=True, shuffle=True, test_size=0.2, labels=label_cols, features=feature_cols, moving_average=True)


# # Unshuffled
# RFModel = RandomForestRegressor(random_state=42)
# model, score, prediction, test_dates_prices = train_test_data_model(RFModel, df_no_headlines, plot=True, shuffle=False, test_size=0.2, labels=label_cols, features=feature_cols, moving_average=True)





# =============================================
# 				With just headlines
# =============================================
chosen_cols = ['Adj Close', 'compound', 'neg', 'neu', 'pos', 'predicted_stock_price']
df_just_headlines = data[chosen_cols]
feature_cols = ['Adj Close', 'compound', 'neg', 'neu', 'pos']
label_cols = ['predicted_stock_price']
# =============================================
print "Data Frame with just headlines"
print df_just_headlines.head()


# # # Suffled 
# RFModel = RandomForestRegressor(random_state=42)
# LRModel = LinearRegression()
# KNNModel = KNeighborsRegressor(n_neighbors=3)
# SVRModel = SVR(kernel='linear', max_iter=2000)
# MLPRModel = MLPRegressor(hidden_layer_sizes=(100,200,100), activation='relu', batch_size=100, random_state=42)
# model, score, prediction, test_dates_prices = train_test_data_model(MLPRModel, df_just_headlines, plot=True, shuffle=True, test_size=0.2, labels=label_cols, features=feature_cols, moving_average=True)


# # Unshuffled
# RFModel = RandomForestRegressor(random_state=42)
# LRModel = LinearRegression()
# KNNModel = KNeighborsRegressor(n_neighbors=3)
# SVRModel = SVR(kernel='linear', max_iter=2000)
# MLPRModel = MLPRegressor(hidden_layer_sizes=(100,200,100), activation='relu', batch_size=100, random_state=42)
# model, score, prediction, test_dates_prices = train_test_data_model(MLPRModel, df_just_headlines, plot=True, shuffle=False, test_size=0.2, labels=label_cols, features=feature_cols, moving_average=True)










# =============================================
# 			With healines and percent changes
# =============================================
chosen_cols = ['Adj Close', 'high_low_volatility_pct', 'daily_pct_change', 'Volume', 'compound', 'neg', 'neu', 'pos',  'predicted_stock_price']
df_with_headlines = data[chosen_cols]
feature_cols = ['Adj Close', 'high_low_volatility_pct', 'daily_pct_change', 'Volume', 'compound', 'neg', 'neu', 'pos']
label_cols = ['predicted_stock_price']
# =============================================
print "Data Frame with headlines and engineered features"
print df_with_headlines.head()


# # # # Suffled 
# RFModel = RandomForestRegressor(random_state=42)
# LRModel = LinearRegression()
# KNNModel = KNeighborsRegressor(n_neighbors=3)
# SVRModel = SVR(kernel='linear', max_iter=2000)
# MLPRModel = MLPRegressor(hidden_layer_sizes=(100,200,100), activation='relu', batch_size=100, random_state=42)
# model, score, prediction, test_dates_prices = train_test_data_model(MLPRModel, df_with_headlines, plot=True, shuffle=True, test_size=0.2, labels=label_cols, features=feature_cols, moving_average=True)




# # Unshuffled 
RFModel = RandomForestRegressor(random_state=42)
LRModel = LinearRegression()
KNNModel = KNeighborsRegressor(n_neighbors=3)
SVRModel = SVR(kernel='linear', max_iter=2000)
MLPRModel = MLPRegressor(hidden_layer_sizes=(100,200,100), activation='relu', batch_size=100, random_state=42)
model, score, prediction, test_dates_prices = train_test_data_model(MLPRModel, df_with_headlines, plot=False, shuffle=False, test_size=0.2, labels=label_cols, features=feature_cols, moving_average=True)

latest_stock_price =  test_dates_prices['predicted_stock_price'][-1:].values[0]
latest_stock_price_prediction =  prediction['predicted_stock'][-1:].values[0]


print "Latest stock price: {}".format(latest_stock_price)
print "Latest stock price prediction: {}".format(latest_stock_price_prediction)
print "Difference from predicted and actual price: {}".format((latest_stock_price_prediction) - (-1 *latest_stock_price))