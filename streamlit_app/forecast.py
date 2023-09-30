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
        history_train_df = history_train_df.loc[:, ['SMA_20', 'SMA_253', 'EMA_5', 'EMA_20', 'EMA_253', 'RSI_14',
                                                    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0',
                                                    'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'Lag1_Price',
                                                    'Lag2_Price', 'Close']]

        history_test_df = test_df.drop('Close', axis=1)
        history_test_df = history_test_df.tail(20)
        history_test_df['Close'] = y_pred # add the predicted price from test data
        history_test_df = history_test_df.loc[:, ['SMA_20', 'SMA_253', 'EMA_5', 'EMA_20', 'EMA_253', 'RSI_14',
                                                'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0',
                                                'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'Lag1_Price',
                                                'Lag2_Price', 'Close']]

        history_pred_df = pd.concat([history_train_df, history_test_df])

        for date in future_dates:
            target_date = str(date.date())
            print(target_date)

            prev_5_days = history_pred_df.loc[:target_date].tail(5) # EMA
            prev_20_days = history_pred_df.loc[:target_date].tail(20) # SMA,EMA,RSI,BBANDS
            prev_35_days = history_pred_df.loc[:target_date].tail(35) # MACD
            prev_253_days = history_pred_df.loc[:target_date].tail(253) # SMA, EMA
            
            # SMA and EMA from predicted price for the target_date
            SMA_20 = ta.sma(prev_20_days['Close'], length=20)[-1]
            SMA_253 = ta.sma(prev_253_days['Close'], length=253)[-1]
            EMA_5 = ta.ema(prev_5_days['Close'], length=5)[-1]
            EMA_20 = ta.ema(prev_20_days['Close'], length=20)[-1]
            EMA_253 = ta.ema(prev_253_days['Close'], length=253)[-1]

            MACD = prev_35_days.ta.macd(fast=12, slow=26, signal=9).iloc[-1]

            BB = prev_20_days.ta.bbands(length=20, std=2).iloc[-1]

            # assign Lag1_Price as the predicted price for the previous business day
            lag1_date = pd.date_range(end=target_date, periods=2, freq='B')[0]
            lag1_df = history_pred_df.loc[history_pred_df.index <= lag1_date]
            Lag1_Price = lag1_df.iloc[-1]['Close']

            # assign Lag2_Price as the predicted price for two business days ago
            lag2_date = pd.date_range(end=target_date, periods=3, freq='B')[0]
            lag2_df = history_pred_df.loc[history_pred_df.index <= lag2_date]
            Lag2_Price = lag2_df.iloc[-1]['Close']

            RSI_14 = ta.rsi(prev_20_days['Close'], length=14)[-1]

            # create a DataFrame for the features of the target date
            X_features_date = pd.DataFrame({
                'SMA_20': [SMA_20],
                'SMA_253': [SMA_253],
                'EMA_5': [EMA_5],
                'EMA_20': [EMA_20],
                'EMA_253': [EMA_253],
                'RSI_14': [RSI_14],
                'MACD_12_26_9' : [MACD['MACD_12_26_9']],
                'MACDh_12_26_9' : [MACD['MACDh_12_26_9']],
                'MACDs_12_26_9' : [MACD['MACDs_12_26_9']],
                'BBL_20_2.0' : [BB['BBL_20_2.0']],
                'BBM_20_2.0' : [BB['BBM_20_2.0']],
                'BBU_20_2.0' : [BB['BBU_20_2.0']],
                'BBB_20_2.0' : [BB['BBB_20_2.0']],
                'BBP_20_2.0' : [BB['BBP_20_2.0']],
                'Lag1_Price': [Lag1_Price],
                'Lag2_Price': [Lag2_Price],
            }, index=[pd.to_datetime(target_date)])

            X_prev_20_days = history_pred_df.loc[:target_date].tail(20).drop('Close', axis=1)

            X_features_forecast = pd.concat([X_prev_20_days, X_features_date])
            X_features_forecast.reset_index(inplace=True)
            X_features_forecast = X_features_forecast.rename(columns = {'index':'Date'})
            X_features_forecast['Month'] = X_features_forecast['Date'].dt.month
            X_features_forecast['Day'] = X_features_forecast['Date'].dt.dayofweek

            # Create an empty DataFrame with columns for all one-hot encoded features
            one_hot_cols = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4']

            if 5 in X_features_forecast['Month'].unique():
                remained_one_hot_cols = [ 'month_1', 'month_2', 'month_3', 'month_4', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']
            else:
                remained_one_hot_cols = [ 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']

            # one-hot encode the Month and Day columns
            month_one_hot = pd.get_dummies(X_features_forecast['Month'], prefix='month', dtype=float)
            day_one_hot = pd.get_dummies(X_features_forecast['Day'], prefix='day', dtype=float)

            # combine one-hot encoded dataframes
            one_hot_df = pd.concat([month_one_hot, day_one_hot], axis=1)
            for col in remained_one_hot_cols:
                one_hot_df[col] = 0

            one_hot_df = one_hot_df[one_hot_cols]

            # concatenate the one-hot encoded columns to the original dataset
            X_features_forecast = pd.concat([X_features_forecast, one_hot_df], axis=1)

            # preparing features for prediction
            X_features_forecast = X_features_forecast.set_index(X_features_forecast['Date'])
            X_features_forecast.drop(columns=['Date', 'Month', 'Day'], axis=1, inplace=True)

            date_index = []
            X_forecast = []

            for i in range(len(X_features_forecast)-self.sequence_length+1):
                X_forecast.append(X_features_forecast.iloc[i:i+self.sequence_length].values)
                date_index.append(X_features_forecast.index[i+self.sequence_length-1])
                
            X_forecast = np.array(X_forecast)
            date_index = np.array(date_index)

            X_forecast = X_forecast.reshape(-1, X_train.shape[-1])
            # scale the X_feature_forecasts data
            X_features_forecast_scaled = scaler.transform(X_forecast)


            # create a TimeseriesGenerator object for X_feature_forecasts
            feature_generator = TimeseriesGenerator(X_features_forecast_scaled, np.zeros(len(X_features_forecast_scaled)), length=self.sequence_length, batch_size=1)

            # predict next day price
            predicted_price = model.predict(feature_generator)

            # get last day of the prediction
            X_features_date['Close'] = predicted_price[-1][-1]
            history_pred_df = pd.concat([history_pred_df, X_features_date])
            print("===================== done adding ================")

        history_pred_df.index = history_pred_df.index.strftime('%Y-%m-%d')
        observed_period = history_pred_df.loc['2023-05-03':'2023-05-31']
        forecast_period = history_pred_df.loc['2023-06-01':]

        # calculate residuals from observed period
        residuals = sorted([x - y for x, y in zip(y_pred, y_actual)])

        output = {}
        output['observed_period'] = observed_period.to_dict()
        output['forecast_period'] = forecast_period.to_dict()
        output['y_actual'] = y_actual.tolist()
        output['residuals'] = residuals

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
