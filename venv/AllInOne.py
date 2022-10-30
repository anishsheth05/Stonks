# importing libraries required - yfinance to get data, matplotlib to plot stuff, numpy for the fast arrays
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numba as nb
import pandas as pd
from pandas.plotting import register_matplotlib_converters
# here the graph is initialized and set up to look coolio

register_matplotlib_converters()
sns.set(context='paper', style='darkgrid', font_scale=0.6)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')

# below is the name of the stock, if you want to analyze a certain stock, put its ticker here
ticker = 'SPY'
stock = yf.download(ticker, period='2y', interval='1d')
slen = len(stock)
length = np.arange(0, slen, 1)
parsing = np.array([length, stock.Close])
# parsing looks like this:
# [ [0, 1, 2, 3, 4]
#   [2.33, 1.33, 6.35, 3.86, 5.34] ]
# print(parsing[1])
data = np.empty((slen, 2))
for i in range(0, slen):
    data[i] = np.array([parsing[0][i], parsing[1][i]])
# now data looks like this:
# [ [0,3.42], [1, 4,32], [2, 4.56] ]
# print(data)
# finally, we plot the stock but use the initial stuff,
# with data and parsing being ready to use in analysis later
ax1.plot(stock.Close, label=ticker)


# calculating simple/exponential moving averages

Data = stock.Close
Index = Data.index
# Change the window length for different smoothnesses
SMA20 = Data.rolling(window=20).mean()
SMA50 = Data.rolling(window=50).mean()
ax1.plot(SMA20, label='20d SMA')
ax1.plot(SMA50, label='50d SMA')
# based off this whether you should buy
trading_positions_raw = SMA20 - SMA50
trading_positions_raw.tail()
trading_positions = trading_positions_raw.apply(np.sign)*2/3
trading_positions = trading_positions.shift(1)
trading_positions.fillna(0)


# calculating the RSI, building on the already known knowledge of the SMA
@nb.jit(fastmath=True, nopython=True)
def calc_rsi(array, deltas, avg_gain, avg_loss, n):
    # Use Wilder smoothing method
    up = lambda x:  x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = n+1
    for d in deltas[n+1:]:
        avg_gain = ((avg_gain * (n-1)) + up(d)) / n
        avg_loss = ((avg_loss * (n-1)) + down(d)) / n
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            array[i] = 100 - (100 / (1 + rs))
        else:
            array[i] = 100
        i += 1

    return array


def get_rsi(array, n=14):

    deltas = np.append([0], np.diff(array))

    avg_gain = np.sum(deltas[1:n+1].clip(min=0)) / n
    avg_loss = -np.sum(deltas[1:n+1].clip(max=0)) / n

    array = np.empty(deltas.shape[0])
    array.fill(np.nan)

    array = calc_rsi(array, deltas, avg_gain, avg_loss, n)
    return array


RSI = pd.DataFrame(get_rsi(Data), index=Index)
RSI_min = RSI.rolling(20, min_periods=1, center=False).min()
RSI_max = RSI.rolling(20, min_periods=1, center=False).max()

trading_positions[RSI_min[0] <= 10] += 1/4
trading_positions[RSI_min[0] <= 30] += 1/4
trading_positions[RSI_min[0] > 30] -= 1/4
trading_positions[RSI_max[0] >= 90] -= 1/4
trading_positions[RSI_max[0] >= 70] -= 1/4
trading_positions[RSI_max[0] < 70] += 1/4


ax3.plot(RSI, label='RSI')
ax3.plot(RSI_max, label='Rolling Max')
ax3.plot(RSI_min, label='Rolling Min')
ax3.plot(RSI_max.rolling(10).mean(), label='Max Trend')
ax3.plot(RSI_min.rolling(10).mean(), label='Min Trend')
ax3.legend(loc='lower left')
# this entire length plots the stock by itself very nicely
ax1.set_title(ticker + ' Stock')
ax1.legend(loc='upper left')
ax1.set_ylabel('Value ($)')
trading_positions[trading_positions > 0] = 1
trading_positions[trading_positions < 0] = -1
ax2.plot(trading_positions)
ax2.set_ylabel('Trading Position')
plt.show()
