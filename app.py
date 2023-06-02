import os
import io
import base64
import seaborn as sns
import plotly.io as pio
import json
import plotly.graph_objects as go
from prophet import Prophet
from flask import Flask, render_template, request
import pandas as pd
from flask import g
from statsmodels.tsa.arima.model import ARIMA
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib.dates as mdates
#import AutoDateLocator
from datetime import timezone


app = Flask(__name__)

df = pd.read_csv('df3.csv')
#df1 = pd.read_csv('prophet_data_one.csv')

'''def forecast_temperatures(df, forecast_periods, model_path='arima_model.pkl'):
    g.df = df
    with open(model_path, 'rb') as file:
        model_fit = joblib.load(file)

    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_values = forecast.predicted_mean
    lower_bound = forecast.conf_int().iloc[:, 0]
    upper_bound = forecast.conf_int().iloc[:, 1]

    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(hours=1), periods=forecast_periods, freq='H')

    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Temperature': forecast_values})
    forecast_df.set_index('Date', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Temperature (C)'], label='Actual')
    plt.plot(forecast_df.index, forecast_df['Temperature'], label='Forecast')
    plt.fill_between(forecast_df.index, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Forecast')
    plt.legend()
    plt.savefig('static/forecast.png')

    return forecast_df'''

'''def make_forecast(dataset, forecast_periods, sequence_length=24, model_path='lstm_model.pkl'):
    g.df = dataset
    model = joblib.load(model_path)
    temperatures = dataset['Temperature (C)'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temperatures_scaled = scaler.fit_transform(temperatures)
    last_sequence = temperatures_scaled[-sequence_length:]
    forecast = []

    for _ in range(forecast_periods):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        forecast.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast = forecast.flatten()
    last_date = dataset.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(hours=1), periods=forecast_periods, freq='H')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Temperature': forecast})
    forecast_df.set_index('Date', inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(dataset.index, dataset['Temperature (C)'], label='Actual')
    plt.plot(forecast_df.index, forecast_df['Temperature'], label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Forecast')
    plt.legend()
    plt.savefig('static/forecast.png')

    return forecast_df'''



def plot_forecast(df_prophet, future_periods, saved_model_path='prophet_model.joblib'):
    model = joblib.load(saved_model_path)
    future_dates = model.make_future_dataframe(periods=future_periods, include_history=False)
    forecast = model.predict(future_dates)
    last_365_days = df_prophet['ds'].tail(365)
    last_365_days_values = df_prophet['y'].tail(365)
    '''plt.figure(figsize=(10, 6))
       plt.plot(forecast['ds'], forecast['yhat'], 'r-', label='Forecast')
       plt.plot(last_365_days, last_365_days_values, 'g-', label='Last 365 Days')
       plt.xlabel('Date')
       plt.ylabel('Value')
       plt.title('Forecast vs Last 365 Days')
       plt.legend()
        
       return plt'''

    fig = go.Figure()

   
    fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='red')
        ))

    
    fig.add_trace(go.Scatter(
    x=last_365_days,
    y=last_365_days_values,
    mode='lines',
    name='Last 365 Days',
    line=dict(color='green')
        ))

   
    fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    title='Forecast vs Last 365 Days'
        )

   
    fig.show()































def forecast_temperatures(df, forecast_periods, model_path='arima_model.pkl' , last_date = None):
    with open(model_path, 'rb') as file:
        model_fit = joblib.load(file)

    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_values = forecast.predicted_mean
    lower_bound = forecast.conf_int().iloc[:, 0]
    upper_bound = forecast.conf_int().iloc[:, 1]
    
    if last_date is None:
        last_date = dataset.index[-1]
    #last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_periods, freq='H')

    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Temperature': forecast_values})
    forecast_df.set_index('Date', inplace=True)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Temperature (C)'], label='Actual')
    plt.plot(forecast_df.index, forecast_df['Temperature'], label='Forecast')
    plt.fill_between(forecast_df.index, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Forecast')
    plt.legend()

    

    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return forecast_df, image_base64




def make_forecast(dataset, forecast_periods, sequence_length=24, model_path='lstm_model.pkl', last_date=None):
    g.df = dataset
    model = joblib.load(model_path)
    temperatures = dataset['Temperature (C)'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temperatures_scaled = scaler.fit_transform(temperatures)
    last_sequence = temperatures_scaled[-sequence_length:]
    forecast = []

    for _ in range(forecast_periods):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        forecast.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast = forecast.flatten()

    if last_date is None:
        last_date = dataset.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_periods, freq='H')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Temperature': forecast})
    forecast_df.set_index('Date', inplace=True)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(dataset.index, dataset['Temperature (C)'], label='Actual')
    plt.plot(forecast_df.index, forecast_df['Temperature'], label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Forecast')
    plt.legend()
    plt.show()

    #image_file = 'forecast_plot.png'
    #plt.savefig(image_file)
    #plt.close()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    
    image_data = base64.b64encode(buffer.read()).decode('utf-8')

    
    plt.close(fig)

    return forecast_df , image_data



@app.route('/')
def home():
   
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    forecast_periods = int(request.form['forecast_periods'])
    model_choice = request.form['model_choice']

    if model_choice == 'ARIMA':
         last_date = pd.Timestamp(df.index[-1])
         forecast, image_data = forecast_temperatures(df, forecast_periods, model_path='arima_model.pkl' , last_date = last_date)

    elif model_choice == 'LSTM':
        last_date = pd.Timestamp(df.index[-1])
        #image_file = 'forecast_plot.png'
        #forecast ,_ = make_forecast(df, forecast_periods, sequence_length=24, model_path='lstm_model.pkl', last_date=last_date)
        forecast, image_data = make_forecast(df, forecast_periods, sequence_length=24, model_path='lstm_model.pkl',
                                      last_date=last_date)
        
    elif model_choice == 'Prophet':
        forecast_periods = int(request.form['forecast_periods'])
        df_prophet = pd.read_csv('prophet_data_one.csv')  # Replace with your dataset
        #plot = plot_forecast(df_prophet, forecast_periods)
        plot_forecast(df_prophet, forecast_periods)
        #image_data = plot_to_img(plot)
        
        #return render_template('result.html', image_data=image_data)
        return render_template('prophet.html')


    #return render_template('result.html' , forecast = forecast)
    return render_template('result.html', forecast=forecast, image_data=image_data)

@app.route('/data.json')
def serve_data():
    try:
        with open('data.json', 'r') as file:
            data = file.read()
        return data, 200, {'Content-Type': 'application/json'}
    except FileNotFoundError:
        return 'Data file not found', 404
