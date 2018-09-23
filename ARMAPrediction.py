import wrds
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from arch import arch_model

db = wrds.Connection()
# andrew95
# Cynfudan95

db.list_libraries()

data_merged = db.raw_sql("select permno, prc, shrout, date, hsiccd from crspq.dsf where prc>0 and shrout*prc < 50000 and hsiccd < 3600 and hsiccd >3500 and date = '2012-10-22'")

data_stock = db.raw_sql("select permno, vol, date from crspq.dsf where permno= 11394.0 and date > '2012-10-22'")

dataframe=data_stock[['date', 'vol']]

series=dataframe.set_index('date')

difference=series-series.shift()
difference.dropna(inplace=True)

# Multistep Forecast
split_point = round(len(difference) *0.99)
dataset, validation = difference[0:split_point], difference[split_point:]

# plot_acf(dataset, lags=10)
# plot_pacf(dataset, lags=10)

mod=ARMA(dataset, order=(7,2))
result=mod.fit()

forecast = result.forecast(steps=len(validation))[0]

error=np.transpose(validation)-forecast
error=error.transpose()

multistepForecast=pd.DataFrame(validation)
multistepForecast['forecast']=forecast
multistepForecast['error']=error
print(multistepForecast)

err2=np.square(multistepForecast['error'])
RMSE=np.sqrt(err2.sum()/len(validation))
print(RMSE)

# Single Step Forecast
split_point = round(len(difference) *0.99)
dataset, validation = difference[0:split_point], difference[split_point:]

forecastList=[]

for i in range(len(validation)):
    train=difference[0:split_point+i]
    mod = ARMA(train, order=(7, 2))
    result = mod.fit()
    forecast = result.forecast(steps=1)[0]
    forecastList.append(forecast[0])

error=np.transpose(validation)-forecastList
error=error.transpose()

singleStepForecast=pd.DataFrame(validation)
singleStepForecast['forecast']=forecastList
singleStepForecast['error']=error
print(singleStepForecast)

err2=np.square(singleStepForecast['error'])
RMSE2=np.sqrt(err2.sum()/len(validation))
print(RMSE2)

# Normalize the data
normalized=(series-np.mean(series))/np.std(series)

difference=normalized-normalized.shift()
difference.dropna(inplace=True)

split_point = round(len(difference) *0.99)
dataset, validation = difference[0:split_point], difference[split_point:]

mod=ARMA(dataset, order=(7,2))
result=mod.fit()

forecast = result.forecast(steps=len(validation))[0]

error=np.transpose(validation)-forecast
error=error.transpose()

normalizedMultistepForecast=pd.DataFrame(validation)
normalizedMultistepForecast['forecast']=forecast
normalizedMultistepForecast['error']=error
print(normalizedMultistepForecast)

err2=np.square(normalizedMultistepForecast['error'])
normalizedRMSE=np.sqrt(err2.sum()/len(validation))
print(normalizedRMSE)


split_point = round(len(difference) *0.99)
dataset, validation = difference[0:split_point], difference[split_point:]

forecastList=[]

for i in range(len(validation)):
    train=difference[0:split_point+i]
    mod = ARMA(train, order=(7, 2))
    result = mod.fit()
    forecast = result.forecast(steps=1)[0]
    forecastList.append(forecast[0])

error=np.transpose(validation)-forecastList
error=error.transpose()

normalizedSingleStepForecast=pd.DataFrame(validation)
normalizedSingleStepForecast['forecast']=forecastList
normalizedSingleStepForecast['error']=error
print(normalizedSingleStepForecast)

err2=np.square(normalizedSingleStepForecast['error'])
normalizedRMSE2=np.sqrt(err2.sum()/len(validation))
print(normalizedRMSE2)

