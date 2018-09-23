import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from arch import arch_model

series = pd.read_csv('Microsoft Share Volume.csv', index_col='date', parse_dates=True)
print(series)
series.plot()
plt.show()

mod1 = arch_model(series,vol='EGarch',p=2,o=0,q=2)
res = mod1.fit()
print(res.summary())
res.plot()
plt.show()