# stock_price_prediction
## Forecasting interface using streamlit
To run the project, first install the dependencies:
```pip install -r requirements.txt```

Then, start the FastAPI app:
```uvicorn api:app --reload```

Finally, start the Streamlit app:
```streamlit run main.py```

The web interface can be accessed at http://localhost:8501

# Project structure
The project has the following structure:

```
├── model_dev 
│   ├── arima_parameters.ipynb
│   ├── lstm_backtest.ipynb 
│   ├── lstm_forecast.ipynb 
├── streamlit_app 
│   ├── main.py
│   ├── forecast.py 
│   ├── api.py 
├── README.md 
 ```

`model_dev/`: Contains jupyter notebooks for modeling (ARIMA, Forecasting, and Backtesting).\n

`streamlit_app`: Contains the Streamlit web interface for forecasting.

> Disclaimer: for privacy purpose, model_dev files and forecast.py script are hided.

## ARIMA Model Parameters
- ACF and PACF plot
- Identifying stationarity of data
- Fit data on ARIMA model

## Stock Price Prediction using LSTM - 1 month forecast
### Exploratory Data Analyis (matplotlib)
- Identifying Seasonality using statsmodels' seasonal_decompose

### Feature Engineering
- Creating multiple new features: SMA_20, SMA_253, EMA_5, EMA_20, Lag1_Price, Lag2_Price, Month, DayOfWeek to feed into LSTM model
- Additional technical indicators: MACD, RSI, BBANDS
- One-hot Encoding for Month and DayOfWeek
- Splitting data: train-(until 2023-05-02) test-(20 days before 2023-05-02 + after 2023-05-02)
- Scaling: Standard Scaler
- Sequence Creation: using sequence length=20 and generate sequences using Keras' TimeseriesGenerator

### Model Architecture
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    lstm (LSTM)                 (None, 20, 64)            25088     
                                                                    
    dropout (Dropout)           (None, 20, 64)            0         
                                                                    
    lstm_1 (LSTM)               (None, 20, 32)            12416     
                                                                    
    dropout_1 (Dropout)         (None, 20, 32)            0         
                                                                    
    lstm_2 (LSTM)               (None, 32)                8320      
                                                                    
    dropout_2 (Dropout)         (None, 32)                0         
                                                                    
    dense (Dense)               (None, 1)                 33        
                                                                    
    =================================================================
    Total params: 45857 (179.13 KB)
    Trainable params: 45857 (179.13 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________

### Equation

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

### Training
- Performance Metric = MSE
- Optimizer = Adam(learning_rate=0.001)
- Epochs = 100

### Testing
- Score using Mean Absolute Percentage Error
- Test MAPE = 0.7199

### Hyperparameter Tuning
- Random Search
- Tuned model Test MAPE = 0.6994

### Forecasting Price for Next 3 Weeks
- Extrapolation technique: predicting next day's price, and update features for the following future dates for further predictions
- Iteratively feeding next day's features to the LSTM model

### 95% Prediction Interval
- Reference: https://towardsdatascience.com/time-series-forecasting-prediction-intervals-360b1bf4b085
- Check distribution to be normal using using smirnov-kolmogorov, anderson-darling, d'agostino k-squared tests
- Visualization of price data using matplotlib
![HSBC Forecasted Stock Price](/images/hsbc_forecast.png "HSBC Forecasted Stock Price")

## LSTM Backtesting Strategy - 1 year period

### Data Preparation similar to what we applied for the forecasting section
### Modification of extrapolation technique to only predict next 3 days instead next 1 month

### Backtesting
- Train initial model to serve as a base model which we feed the model with data until the last day before backtesting period starts: 2023-09-01

#### Strategy used in the next() function of backtesting.py:
- Retrieve the average of the next 3 days prediction from the current date using our forecasting function.
- Compare daily % change from current day's forecast to previous day's forecast, if its larger than price_delta, then we buy long position and vice versa for sell short position.
- Implementing stop-loss and take-profit with a pre-determined spread (price_delta) and position-sizing .
- Retraining the model every 20 trading days (1 month), feeding it the last 20 days of the actual data so it can keep up to the changing information/trends happening in the market.

#### Result
    Start                     2022-05-24 00:00:00
    End                       2024-09-12 00:00:00
    Duration                    842 days 00:00:00
    Exposure Time [%]                   30.633803
    Equity Final [$]                 65892.686079
    Equity Peak [$]                  69456.414091
    Return [%]                         558.926861
    Buy & Hold Return [%]               57.341165
    Return (Ann.) [%]                  130.827173
    Volatility (Ann.) [%]               59.016219
    Sharpe Ratio                           2.2168
    Sortino Ratio                       44.452487
    Calmar Ratio                        20.603894
    Max. Drawdown [%]                   -6.349633
    Avg. Drawdown [%]                   -1.964769
    Max. Drawdown Duration      119 days 00:00:00
    Avg. Drawdown Duration       18 days 00:00:00
    # Trades                                  115
    Win Rate [%]                        52.173913
    Best Trade [%]                      10.050958
    Worst Trade [%]                     -0.601202
    Avg. Trade [%]                        2.08315
    Max. Trade Duration           5 days 00:00:00
    Avg. Trade Duration           2 days 00:00:00
    Profit Factor                        8.444928
    Expectancy [%]                       2.138787
    SQN                                  5.468472
    _strategy                       LSTMStrategy2
    _equity_curve                             ...
    _trades                        Size  Entry...
    dtype: object

![Backtesting Chart](/images/backtesting_chart.jpeg "Backtesting Chart")

- We increased our initial equity from 10k to 66m with 559% return after 1 year period.
- Win rate is quite decent: ~52% which shows we are winning more.
- Good result of Profit Factor and Expectancy.
- High Sharpe ratio of 2.2
- Need to note that our strategy is very risky with high volatility. So, further improvements need to be made to address this issue despite a very high return % that we get.

Overall, our strategy using LSTM model that we made can give use a very high return % during the backtesting period, but additional measurements to take into account the risk should be made.

