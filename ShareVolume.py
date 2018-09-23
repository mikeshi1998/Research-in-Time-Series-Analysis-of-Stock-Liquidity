import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA

series = pd.read_csv('Microsoft Share Volume.csv', index_col='date', parse_dates=True)

difference=series-series.shift()
difference.dropna(inplace=True)

series_log=np.log(series)

log_difference=series_log-series_log.shift()
log_difference.dropna(inplace=True)

merge=pd.merge(pd.merge(series, difference, on='date'), pd.merge(series_log, log_difference, on='date'), on='date')
print(merge)

series.plot()
plt.show()

plot_acf(series, lags=10)
plt.show()

plot_pacf(series, lags=10)
plt.show()

difference.plot()
plt.show()

plot_acf(difference, lags=10)
plt.show()

plot_pacf(difference, lags=10)
plt.show()

log_difference.plot()
plt.show()

plot_acf(log_difference, lags=10)
plt.show()

plot_pacf(log_difference, lags=10)
plt.show()

# mod=ARMA(log_difference, order=(1, 1))
# result=mod.fit()
# print(result.summary())
#
# mod2=ARMA(log_difference, order=(3, 1))
# result2=mod2.fit()
# print(result2.summary())
#
# series2=pd.read_csv("Microsoft Share Volume.csv", usecols=[1])
# print(series2)
#
# mod3=ARMA(series2, order=(3, 2))
# result3=mod3.fit()
# print(result3.summary())
# result3.plot_predict(start=2000, end=2800)
# plt.show()
#
# series3=pd.read_csv("Microsoft Share Volume Monthly.csv", index_col='date', parse_dates=True)
# month=series3.to_period(freq="M")
# print(month)
#
# series_log=np.log(month)
# print(series_log)
#
# difference=series_log-series_log.shift()
# print(difference)
#
# difference.dropna(inplace=True)
# print(difference)
#
# month.plot()
# plt.show()
#
# plot_acf(month, lags=30)
# plt.show()
#
# plot_pacf(month, lags=30)
# plt.show()

# mod4=ARMA(month, order=(1, 1))
# result4=mod4.fit()
# print(result4.summary())
# result4.plot_predict(start='2000-01', end='2018-12')
# plt.show()



# look at differences and rerun analysis
# use small-cap stock
# add EGARCH assumption
