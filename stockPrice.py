import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA


series = pd.read_csv('Stock Price.csv', usecols=[1, 2], index_col='date', parse_dates=True)
# print(series)

series_log=np.log(series)
# print(series_log)

difference=series_log-series_log.shift()
# print(difference)

difference.dropna(inplace=True)
# print(difference)

merge=pd.merge(pd.merge(series, series_log, on='date'), difference, on='date')
print(merge)

