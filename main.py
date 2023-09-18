import numpy as np
import pandas as pd
import requests
import math
from scipy.stats import percentileofscore as score
import xlsxwriter
from config import IEX_Cloud_Api_Token
from config import IEX_SANDBOX_API_TOKEN
from config import Alpha_Vantage_Api_key
import json
from alpha_vantage.timeseries import TimeSeries
import time
from multiprocessing import Process
from threading import Thread
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import talib
import seaborn as sns
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor

df_columns = [
    'Ticker',
    'Daily Closes',
    'Daily Volumes',

]


df = pd.DataFrame(columns=df_columns)
pd.set_option("display.max_rows", None, "display.max_columns", None)





symbol = 'AAPL'

api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={Alpha_Vantage_Api_key}'
data_daily = requests.get(api_url).json()
#print(data_daily)

"""def getList(diction):
    return diction.keys()"""

diction = list(data_daily['Time Series (Daily)'].keys())
real_dates = diction[::-1]
#print(real_dates)
#print(diction)
#print(getList(diction))


def get_closes_daily():
    for data in data_daily['Time Series (Daily)']:
        c_data = data_daily['Time Series (Daily)'][data]['5. adjusted close']
        close_data.append(c_data)


close_data = []
get_closes_daily()

def get_volumes_daily():
    for data in data_daily['Time Series (Daily)']:
        v_data = data_daily['Time Series (Daily)'][data]['6. volume']
        volume_data.append(v_data)


volume_data = []
get_volumes_daily()


close_data.reverse()
volume_data.reverse()

close_data = close_data[-5000:]
volume_data = volume_data[-5000:]
real_dates = real_dates[-5000:]


for (close1D, volume1D) in zip(close_data, volume_data):
    df = df.append(
        pd.Series(
            [
                symbol,
                close1D,
                volume1D,

            ],
        index=df_columns),
        ignore_index=True
    )


df[['Daily Closes']] = df[['Daily Closes']].apply(pd.to_numeric)
df[['Daily Volumes']] = df[['Daily Volumes']].apply(pd.to_numeric)
df['Dates'] = real_dates

"""df['Dates'] = df['Dates'][::-1]"""
df = df[[
    'Dates',
    'Ticker',
    'Daily Closes',
    'Daily Volumes']]
df['Dates'] = pd.to_datetime(df['Dates'], infer_datetime_format=True)
df.set_index('Dates', inplace=True, drop=True)

#print(df)


"""df['Daily Closes'].plot(label='SPY', legend=True)
#plt.show()
plt.clf()

vol = df['Daily Volumes']
vol.plot.hist(bins=25)
plt.clf()
#plt.show()
df['Daily Closes'].pct_change().plot.hist(bins=15)
#df['Daily Closes PCT CHG'].plot.hist(bins=50)
plt.xlabel('adjusted close 1-day percent change')
#plt.show()
plt.clf()"""


df['10D Daily Closes Future'] = df['Daily Closes'].shift(-10)
df['10D Daily Closes Future PCT CHG'] = df['10D Daily Closes Future'].pct_change(10)
df['10D Daily Closes PCT CHG'] = df['Daily Closes'].pct_change(10)
#corr = df[['10D Daily Closes PCT CHG', '10D Daily Closes Future PCT CHG']].corr()
#print(corr)

"""plt.scatter(df['10D Daily Closes PCT CHG'], df['10D Daily Closes Future PCT CHG'])
#plt.show()
plt.clf()"""

#Features are the parameters that are used in order to predict the 'targets'
"""features = df[['Daily Closes', 'Daily Volumes']]
targets = df['Daily Closes Future']
print(type(features))
print(type(targets))"""

df['Daily Closes'] = df[['Daily Closes']].squeeze()
#print(type(df['Daily Closes']))



feature_names = ['10D Daily Closes PCT CHG']

for n in [14, 30, 50,200]:
    df['ma' + str(n)] = talib.SMA(df['Daily Closes'].values, timeperiod=n) / df['Daily Closes']
    df['rsi' + str(n)] = talib.RSI(df['Daily Closes'].values, timeperiod=n)
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

#print(feature_names)

df = df.dropna()
features = df[feature_names]
targets = df['10D Daily Closes Future PCT CHG']
feature_and_target_cols = ['10D Daily Closes Future PCT CHG'] + feature_names
feat_targ_df = df[feature_and_target_cols]
corr = feat_targ_df.corr()
print(corr)
sns.heatmap(corr, annot= True, annot_kws = {"size": 14})
plt.yticks(rotation=0, size = 14); plt.xticks(rotation=90, size = 14)  # fix ticklabel directions and size
plt.tight_layout()  # fits plot area to the plot, "tightly"
#plt.show()
plt.clf()

linear_features = sm.add_constant(features)
train_size = int(0.85 * features.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
#print(linear_features.shape, train_features.shape, test_features.shape)

model = sm.OLS(train_targets, train_features)
results = model.fit()  # fit the model
#print(results.summary())

# examine pvalues
# Features with p <= 0.05 are typically considered significantly different from 0
#print(results.pvalues)

# Make predictions from our model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

# Scatter the predictions vs the targets with 20% opacity
"""plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, test_targets, alpha=0.2, color='r', label='test')"""

# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

# Set the axis labels and show the plot
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  # show the legend
#plt.show()
plt.clf()

# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
new_features = ['Adj_Volume_1d_change', 'Adj_Volume_1d_change_SMA']
feature_names.extend(new_features)
df['Adj_Volume_1d_change'] = df['Daily Volumes'].pct_change()
df['Adj_Volume_1d_change_SMA'] = talib.SMA(df['Adj_Volume_1d_change'].values,
                                               timeperiod=5)

# Plot histogram of volume % change data
df[new_features].plot(kind='hist', sharex=False, bins=50)
#plt.show()

#df.index = pd.to_datetime(df.index)



#print(df['Dates'])

# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(df.index.dayofweek,
                              prefix='weekday',
                              drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = df.index

# Join the dataframe with the days of week dataframe
df = pd.concat([df, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend(['weekday_' + str(i) for i in range(1, 5)])
df.dropna(inplace=True)  # drop missing values in-place
#print(df.head())

# Add the weekday labels to the new_features list
new_features.extend(['weekday_' + str(i) for i in range(1, 5)])
#print(feature_names)

# Plot the correlations between the new features and the targets
sns.heatmap(df[new_features + ['10D Daily Closes Future PCT CHG']].corr(), annot=True)
plt.yticks(rotation=0)  # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90)  # ensure x-axis ticklabels are vertical
plt.tight_layout()
#plt.show()
plt.clf()



                                                                                        # Create a decision tree regression model with default arguments
decision_tree = DecisionTreeRegressor()

# Fit the model to the training features and targets
decision_tree.fit(train_features, train_targets)

# Check the score on train and test
#print(decision_tree.score(train_features, train_targets))
#print(decision_tree.score(test_features, test_targets))


# Loop through a few different max depths and check the performance
for d in [3, 5, 10]:
    # Create the tree and fit it
    decision_tree = DecisionTreeRegressor(max_depth=d)
    decision_tree.fit(train_features, train_targets)

    # Print out the scores on train and test
    #print('max_depth=', str(d))
    #print(decision_tree.score(train_features, train_targets))
    #print(decision_tree.score(test_features, test_targets), '\n')



# Use the best max_depth of 3 from last exercise to fit a decision tree
decision_tree = DecisionTreeRegressor(max_depth=3)
decision_tree.fit(train_features, train_targets)

# Predict values for train and test
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

# Scatter the predictions vs actual values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
#plt.show()
plt.clf()



from sklearn.ensemble import RandomForestRegressor

# Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)

# Look at the R^2 scores on train and test
#print(rfr.score(train_features, train_targets))
#print(rfr.score(test_features, test_targets))


from sklearn.model_selection import ParameterGrid

# Create a dictionary of hyperparameters to search
grid = {'n_estimators': [200], 'max_depth': [3], 'max_features': [4, 8], 'random_state': [42]}
test_scores = []

# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
#print(test_scores[best_idx], ParameterGrid(grid)[best_idx])



# Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=42)
rfr.fit(train_features, train_targets)

# Make predictions with our model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
#plt.show()
plt.clf()

# Get feature importances from our random forest model
importances = rfr.feature_importances_
#print(importances)

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
#plt.show()
plt.clf()


from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=42)
gbr.fit(train_features, train_targets)

#print(gbr.score(train_features, train_targets))
#print(gbr.score(test_features, test_targets))


# Extract feature importances from the fitted gradient boosting model
feature_importances = gbr.feature_importances_

# Get the indices of the largest to smallest feature importances
sorted_index = np.argsort(feature_importances)[::-1]
x = range(10)
"""print(sorted_index)
print(x)"""
# Create tick labels
labels = np.array(feature_names)[sorted_index]

plt.bar(x, feature_importances[sorted_index], tick_label=labels)

# Set the tick lables to be the feature names, according to the sorted feature_idx
plt.xticks(rotation=90)
#plt.show()
plt.clf()

from sklearn.preprocessing import scale

# Remove unimportant features (weekdays)
train_features = train_features.iloc[:, :-4]
test_features = test_features.iloc[:, :-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])

#plt.show()
plt.clf()
from sklearn.neighbors import KNeighborsRegressor

for n in range(2, 26):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)

    # Fit the model to the training data
    knn.fit(scaled_train_features, train_targets)

    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features, test_targets))
    print()  # prints a blank line

# Create the model with the best-performing n_neighbors of 5
knn = KNeighborsRegressor(25)

# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.legend()
#plt.show()
plt.clf()
#                                                                                                 NEURAL NETS
from keras.models import Sequential
from keras.layers import Dense

# Create the model
model_1 = Sequential()
model_1.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# Fit the model
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_train_features, train_targets, epochs=25)


# Plot the losses from the fit
plt.plot(history.history['loss'])

# Use the last loss as the title
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
#plt.show()
plt.clf()


from sklearn.metrics import r2_score

# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend()
#plt.show()
plt.clf()

import keras.losses
import tensorflow as tf

# Create loss function
def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true * y_pred, 0), \
                     penalty * tf.square(y_true - y_pred), \
                     tf.square(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)

keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
print(keras.losses.sign_penalty)

# Create the model
model_2 = Sequential()
model_2.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_2.add(Dense(20, activation='relu'))
model_2.add(Dense(1, activation='linear'))

# Fit the model with our custom 'sign_penalty' loss function
model_2.compile(optimizer='adam', loss='sign_penalty')
history = model_2.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
#plt.show()
plt.clf()
# Evaluate R^2 scores
train_preds = model_2.predict(scaled_train_features)
test_preds = model_2.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')  # plot test set
#plt.legend(); plt.show()
plt.clf()

from keras.layers import Dropout

# Create model with dropout
model_3 = Sequential()
model_3.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_3.add(Dropout(0.5))
model_3.add(Dense(20, activation='relu'))
model_3.add(Dense(1, activation='linear'))

# Fit model with mean squared error loss function
model_3.compile(optimizer='adam', loss='mse')
history = model_3.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
#plt.show()
plt.clf()

# Make predictions from the 3 neural net models
train_pred1 = model_1.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2 = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3 = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds = np.mean(np.hstack((test_pred1, test_pred2, test_pred3)), axis=1)
print(test_preds[-20:])

from sklearn.metrics import r2_score

# Evaluate the R^2 scores
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend()
plt.show()
