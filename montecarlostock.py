# -*- coding: utf-8 -*-
"""MonteCarloStock.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nh3G3zDU6h1VdYh_sD9UfyYnpHdt4KNB
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
y_symbols = ['SCHAND.NS', 'TATAPOWER.NS', 'ITC.NS']
from datetime import datetime
startdate = datetime(2022,12,1)
enddate = datetime(2022,12,15)
data = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)

data.head()

log_returns = np.log(1+data.pct_change())
log_returns.head()

sns.distplot(log_returns.iloc[1:])
plt.xlabel("Daily Return")
plt.ylabel("Frequency")

data.plot(figsize=(15,6))

log_returns.plot(figsize=(15,6))

u = log_returns.mean()
var = log_returns.var()

drift = u - (0.5*var)
drift

stddev = log_returns.std()

x = np.random.rand(10,2)
x

norm.ppf(x)

Z = norm.ppf(np.random.rand(50,10000))
Z

t_intervals = 1000
iterations = 10