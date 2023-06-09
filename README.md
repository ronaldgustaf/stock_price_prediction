# stock_price_prediction

## ARIMA Model Parameters
- ACF and PACF plot
- Identifying stationarity of data
- Fit data on ARIMA model

## Stock Price Prediction using LSTM
### Exploratory Data Analyis (matplotlib)
- Identifying Seasonality using statsmodels' seasonal_decompose
- Exploring Simple Moving Average(SMA) and Exponential Moving Average(SMA) with different date range

## Feature Engineering
- Creating multiple new features: SMA_20, SMA_253, EMA_5, EMA_20, Lag1_Price, Lag2_Price, Month, DayOfWeek to feed into LSTM model
- One-hot Encoding for Month and DayOfWeek
- Splitting data: train (befoore 2023-01-01) test (20 days before 2023-01-01 + after 2023-01-01)
- Scaling: MinMax Scaler
- Sequence Creation: using sequence length=20 and generate sequences using Keras' TimeseriesGenerator

## Model Architecture
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    lstm (LSTM)                 (None, 20, 64)            22784     
                                                                    
    dropout (Dropout)           (None, 20, 64)            0         
                                                                    
    lstm_1 (LSTM)               (None, 20, 32)            12416     
                                                                    
    dropout_1 (Dropout)         (None, 20, 32)            0         
                                                                    
    lstm_2 (LSTM)               (None, 32)                8320      
                                                                    
    dropout_2 (Dropout)         (None, 32)                0         
                                                                    
    dense (Dense)               (None, 1)                 33        
                                                                    
    =================================================================
    Total params: 43,553
    Trainable params: 43,553
    Non-trainable params: 0
    _________________________________________________________________

## Equation

![eq1](/images/eq1.png "eq1")

Where LSTM consists of:
>![eq2](/images/eq2.png "eq2")

Each LSTM layer has 20 more time steps from t until t-20. So for each step:
LSTM layer equation (1 layer, 1 time step: t, repeated from t-20, t-19, …, t)
>![eq3](/images/eq3.png "eq3")
>![eq4](/images/eq4.png "eq4")

Where the outputs after computing from t-20 to t are,
>![eq5](/images/eq5.png "eq5")

Thus the output of LSTM is feeded into the last Dense layer becomes:
![eq6](/images/eq6.png "eq6")

## Training
- Performance Metric = MSE
- Optimizer = Adam(learning_rate=0.001)
- Epochs = 100

## Testing
- Score using Mean Absolute Percentage Error
- Test MAPE = 3.1884

## Hyperparameter Tuning
- Random Search
- Tuned model Test MAPE = 1.575

## Forecasting Price for Next 3 Weeks
- Extrapolation technique: predicting next day's price, and update features for the following future dates for further predictions
- Iteratively feeding next day's features to the LSTM model

## 95% Prediction Interval
- Reference: https://towardsdatascience.com/time-series-forecasting-prediction-intervals-360b1bf4b085
- Check distribution to be normal using using smirnov-kolmogorov, anderson-darling, d'agostino k-squared tests
- Visualization of price data using matplotlib
![HSBC Forecasted Stock Price](/images/hsbc_forecast.png "HSBC Forecasted Stock Price")