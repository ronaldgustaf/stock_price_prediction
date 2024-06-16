import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.tsa.seasonal import seasonal_decompose

import yfinance as yf
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

class Forecast:
    def __init__(self, stock, date_start, date_end):
        self.stock = stock
        self.date_start = date_start
        self.date_end = date_end
        self.sequence_length = 20

        # self.df = self.get_data()
        # self.df = self.clean_data()
        # self.df = self.add_indicators()
        # self.df = self.add_features()
        # self.df = self.add_target()

    def forecast(self):
        self.df = yf.download(self.stock, start=self.date_start, end=self.date_end)
        self.df.reset_index(inplace=True)
        self.df.drop('Close', axis=1, inplace=True)
        self.df = self.df.rename(columns={'Adj Close': f'Close'})
        
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.dayofweek

        # Adding indicators
        self.df['SMA_20'] = ta.sma(self.df['Close'], length=20, append=True)
        self.df['SMA_253'] = ta.sma(self.df['Close'], length=253, append=True)
        
        self.df['EMA_5'] = ta.ema(self.df['Close'], length=5, append=True)
        self.df['EMA_20'] = ta.ema(self.df['Close'], length=20, append=True)
        self.df['EMA_253'] = ta.ema(self.df['Close'], length=253, append=True)

        self.df['RSI_14'] = ta.rsi(self.df['Close'], length=14, append=True)

        self.df.ta.macd(fast=12, slow=26, signal=9, append=True)
        self.df.ta.bbands(length=20, std=2, append=True)

        # feature engineering to prepare data
        self.df['Lag1_Price'] = self.df['Close'].shift(1) # create feature of past 1-day price
        self.df['Lag2_Price'] = self.df['Close'].shift(2) # past 2-days price

        # one-hot encode the Month and Day columns
        month_one_hot = pd.get_dummies(self.df['Month'], prefix='month', dtype=float)
        day_one_hot = pd.get_dummies(self.df['Day'], prefix='day', dtype=float)

        # concatenate the one-hot encoded columns to the original dataset
        self.df = pd.concat([self.df, month_one_hot, day_one_hot], axis=1)
        self.df = self.df.set_index(self.df['Date'])

        self.df.dropna(inplace=True)
        self.df.drop(columns=['Date', 'Month', 'Day', 'Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    
        train_df = self.df.head(-20)
        test_df = self.df.tail(40)

        X_train = train_df.drop('Close', axis=1)
        y_train = train_df['Close']

        X_test = test_df.drop('Close', axis=1)
        y_test = test_df['Close']

        # scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # create sequences of data using TimeseriesGenerator
        train_generator = TimeseriesGenerator(X_train, y_train, length=self.sequence_length, batch_size=32)
        test_generator = TimeseriesGenerator(X_test, y_test, length=self.sequence_length, batch_size=32)

        # defining the model architecture
        model = Sequential()

        # adding LSTM layers
        model.add(LSTM(units=64, input_shape=(self.sequence_length, X_train.shape[1]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32))
        model.add(Dropout(0.1))

        # adding dense layer
        model.add(Dense(units=1))

        # compiling the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        # training the model
        model.fit(train_generator, epochs=100, validation_data=test_generator)

        print("Train MAPE: ", self.eval_train_score(model, train_generator, y_train))

        y_pred, y_actual, test_mape = self.eval_test_score(model, test_generator, y_test)
        print("Test MAPE: ", test_mape)

        future_dates = pd.date_range(start=self.date_end, periods=22, freq='B') #generate only trading days, excluding holidays

        # predicting future prices by calculating each features one-by-one and iteratively predicting the next day price
        history_train_df = train_df.copy()
        # ============================ forecasting code hided ================================
        output = {}

        return output

    # function to calculate MAPE
    def calculate_mape(self, y, y_pred):
        
        y_np = np.array(y)
        y_pred_np = np.array(y_pred)

        return round(np.mean(np.abs((y_np - y_pred_np) / y_np)) * 100, 4)

    def eval_train_score(self, model, train_input, y_train):
        y_train_pred = model.predict(train_input).flatten()
        y_train_actual = np.array(y_train[self.sequence_length:])
        mape = self.calculate_mape(y_train_actual, y_train_pred)

        return mape

    def eval_test_score(self, model, test_input, y_test):
        y_test_pred = model.predict(test_input).flatten()
        y_test_actual = np.array(y_test[self.sequence_length:])
        mape = self.calculate_mape(y_test_actual, y_test_pred)

        return y_test_pred, y_test_actual, mape
