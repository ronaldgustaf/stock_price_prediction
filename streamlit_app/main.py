import streamlit as st
import requests
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FastAPI endpoint
API_URL = "http://localhost:8000/forecast"

# Streamlit app
def main():
    st.title("Stock Price Forecasting using LSTM")

    # Model selection dropdown
    stock = st.text_input("Select a stock", "AAPL")
    date_start = st.text_input("Select a starting date", "YYYY-MM-DD")
    date_end = st.text_input("Select an ending date", "YYYY-MM-DD")

    # Make prediction on button click
    if st.button("Get Forecast"):
        # Prepare feature data as JSON payload
        data = {
            "stock": stock,
            "date_start": date_start,
            "date_end": date_end,
        }

        # Call FastAPI endpoint and get prediction result
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, json=data)

        output = response.json()

        # st.write(f"Prediction: {response.json()}")
        observed_period = pd.DataFrame.from_dict(output['observed_period'])
        observed_period.index = pd.to_datetime(observed_period.index)

        forecast_period = pd.DataFrame.from_dict(output['forecast_period'])
        forecast_period.index = pd.to_datetime(forecast_period.index)

        y_actual = np.array(output['y_actual'])
        residuals = output['residuals']

        forecast_price = np.array(forecast_period['Close'])

        # create a scatter plot with Price on the y-axis and Date on the x-axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(observed_period.index, y_actual, label='Observed period')
        ax.scatter(observed_period.index, observed_period['Close'], label='Estimated period')
        ax.scatter(forecast_period.index, forecast_period['Close'], label='Forecast period')

        # add lines for the observed period and forecast period
        ax.plot(observed_period.index, y_actual, label='Observed price')
        ax.plot(observed_period.index, observed_period['Close'], label='Estimated price')
        ax.plot(forecast_period.index, forecast_period['Close'], label='Forecasted price')

        # creating 95% CI
        RMSFE = np.sqrt(sum([x**2 for x in residuals]) / len(residuals))
        se = 1.96*RMSFE
        upper_bound = forecast_price + se
        lower_bound = forecast_price - se
        ax.fill_between(forecast_period.index, upper_bound, lower_bound, alpha=0.2, label='95% CI', color='b')

        ax.set_title(f'{stock} Forecasted Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        st.pyplot(fig)
    
if __name__ == "__main__":
    main()