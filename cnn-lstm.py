# univariate multi-step lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_excel
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import tensorflow as tf

OUT_STEPS = 2

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	# train, test = data[1:-328], data[-328:-6]
	n = len(data)
	train, test = data[0:int(n*0.7)], data[int(n*0.7):n]
	# restructure into windows of weekly data
	# train = array(split(train, len(train)))
	# test = array(split(test, len(test)))
	return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
  	
		print(actual.shape, predicted.shape)
		mse = mean_squared_error(actual[:, i], predicted[:, i])	
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	print((train.shape[0]*train.shape[1], train.shape[2]))
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 1, 10, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # # define model
    # model = Sequential([
    # tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features))
    # tf.keras.layers.Dense(100, activation='relu')
    # tf.keras.layers.Dense(n_outputs)
    # ])
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -3:, :]), 
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(3)), 
        tf.keras.layers.Dense(OUT_STEPS*n_outputs, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Dense(n_outputs*n_features),
        tf.keras.layers.Reshape([OUT_STEPS, n_features])  
      ])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')  #
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stopping])
    return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=1)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# load the new file
# dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
dataset = read_excel("input.xlsx", sheet_name=1 , header = 0, usecols=[0,1,2])
# dataset.describe().transpose()
# split into train and test
train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 2
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()