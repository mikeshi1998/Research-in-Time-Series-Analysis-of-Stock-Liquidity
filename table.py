from numpy import *
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARMAResults
from pandas import Series


x=range(40)
x=reshape(x, (10, 4))

for p in range(10):
    for q in range(4):
        series = Series.from_csv('Microsoft Share Volume.csv', header=0)
        difference = series - series.shift()
        difference.dropna(inplace=True)
        mod=ARMA(difference, order=(p, q))
        res=mod.fit()
        x[p][q]=res.bic

print(x)

