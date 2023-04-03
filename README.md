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

Y_t=g\left(LSTM\left(x_t,h_{init},c_{init}\right)W_d+b_d\right)

Where LSTM consists of:
LSTM_1\left(x_t,h_{init},c_{init}\right)=o^1,h_t^1,c_t^1
LSTM_2\left(o^1,h_t^1,c_t^1\right)=o^2,h_t^2,c_t^2
LSTM_3\left(o^2,h_t^2,c_t^2\right)=o^3
\therefore LSTM\left(x,h_{init},c_{init}\right)=o^3

Each LSTM layer has 20 more time steps from t until t-20. So for each step:
LSTM layer equation (1 layer, 1 time step: t, repeated from t-20, t-19, …, t)
i_t=\sigma\left(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}c_{t-1}+b_i\right)
f_t=\sigma\left(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}c_{t-1}+b_f\right)
\widetilde{c_t}=tanh\left(W_{xc}x_t+W_{hc}h_{t-1}+b_c\right)
c_t=f_t.c_{t-1}+i_t.\widetilde{c_t}
o_t=\sigma\left(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}c_t+b_o\right)
h_t=o_t.tanh\left(c_t\right)
x_t:input\ vector\ at\ t
h_t:\ hidden\ state\ vector\ at\ t
c_t:memory\ cell\ state\ vector\ at\ t
i_t,f_t,and\ o_t:input,forget,and\ output\ gate\ vectors\ at\ t
W_{xi},W_{hi},W_{ci},W_{xf},W_{hf},W_{cf},W_{xc},W_{hc},W_{xo},W_{ho},and\ W_{co}: weight\ for\ input,hidden,and\ memory\ cell\ states and\ output
b_i,b_f,b_c,and\ b_o\∶\ bias\ vectors\ for\ the\ input,forget,candidate,and\ output\ gates,respectively
tanh\∶\ hyperbolic\ tangent\ activation\ function
\sigma\∶\ sigmoid\ activation\ function

Where the outputs after computing from t-20 to t are,
o^n=\left(o_t,o_{t-1},\ldots,o_{t-20}\right),{\ \ h}_t^n=h_t,{\ \ \ c}_t^n=c_t,\ \ n\ =\ layer\ order\ of\ LSTM

Thus the output of LSTM is feeded into the last Dense layer becomes:
Y_t=g\left(o^3W_d+b_d\right)
{g:\ activation\ function,\ \ W}_d,b_d:\ \ \ parameter\ of\ Dense\ Layer

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