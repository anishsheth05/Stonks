# First importing all libraries and setting up prerequisites
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

register_matplotlib_converters()
data = pd.read_excel('graph_data.xlsx')
plt.style.use('classic')
plt.style.use('stonks.mplstyle')

# setting the the first figure with three sub-graphs
fig1 = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig1.add_gridspec(10, 1)
main = fig1.add_subplot(gs[:6, 0])
trade_pos = fig1.add_subplot(gs[-2:, 0], sharex=main)
percent_bandwidth = fig1.add_subplot(gs[6, 0], sharex=main)
RSI = fig1.add_subplot(gs[7, 0], sharex=main)

# stuff for entire figure
fig1.suptitle('Graph of a Stock and Analysis', size=15)

# stuff for the first graph (with the stock and other stuffs
main.plot(data['Date'], data['Close'], label='Stock', color='blue')
#main.plot(data['Date'], data['Predictions'].shift(1), color='black', label='Future Predictions', alpha=0.5)
main.plot(data['Date'], data['20-Day Moving Average'], label='20d SMA', color='orange', alpha=0.75)
main.plot(data['Date'], data['50-Day Moving Average'], label='50d SMA', color='red', alpha=0.75)
main.plot(data['Date'], data['Top Bollinger Band'], label='Bollinger Bands', color='green', alpha=0.5)
main.plot(data['Date'], data['Bottom Bollinger Band'], color='green', alpha=0.5)
main.set_xlim(left=data['Date'][225])
main.set_ylabel('Price (USD$)', fontdict={'fontsize': 9.25})
main.minorticks_on()
main.tick_params(labelbottom=False)
main.legend(loc='upper left', fontsize='small')

# stuff for the second graph with %B
percent_bandwidth.plot(data['Date'], data['%B'], label='%B from Bollinger Bands', color='green')
percent_bandwidth.legend(fontsize='x-small')
percent_bandwidth.set_yticks([-0.5, 0, 0.5, 1, 1.5])
percent_bandwidth.tick_params(labelbottom=False)

# stuff for fourth graph with RSI
RSI.plot(data['Date'], data['RSI'], label='RSI', color='purple')
RSI.set_yticks([0, 30, 70, 100])
RSI.tick_params(labelbottom=False)
RSI.legend(fontsize='x-small')

# stuff for last graph with trading position (buy/sell)
trade_pos.plot(data['Date'], data['Trading Position'], label='Trading Position', alpha=0.5)
trade_pos.scatter(data['Date'], data['Trading Position Shift Points'], marker='x', color='red', alpha=1)
trade_pos.set_yticks([-1, 0, 1])
trade_pos.set_yticklabels(['Sell', 'Hold', 'Buy'])
trade_pos.legend(fontsize='x-small')

plt.show()
