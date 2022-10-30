import pandas as pd
import numpy as np
# prerequisites complete, beginning stock configuration/analysis

# getting the data from the excel sheet
stock = pd.read_excel('stock_data.xlsx', index_col=0)
stock.Close.plot(label='Close')
stock.Open.plot(label='Open')
close = pd.DataFrame(stock.Close)

# Getting SMAs
SMA20 = close.rolling(20).mean().dropna()
SMA50 = close.rolling(50).mean().dropna()

# Using SMAs to initialize the trading position and its shift point DataFrames
trade_pos = pd.DataFrame(data=(SMA20 - SMA50)).apply(np.sign).fillna(0).tail(452)

# Calculating RSI
deltas = close.diff()
up = deltas[deltas[:] > 0].fillna(0)
down = abs(deltas[deltas[:] < 0].fillna(0))
gain = up.rolling(14).mean().fillna(0)
loss = down.rolling(14).mean().fillna(0)
index = gain.index
for i in range(22, len(gain)):
    gain['Close'][index[i]] = (gain['Close'][index[i - 1]] * 13 + up['Close'][index[i]]) / 14
    loss['Close'][index[i]] = (loss['Close'][index[i - 1]] * 13 + down['Close'][index[i]]) / 14
RS = gain / abs(loss)
RSI = (100 - (100 / (1 + RS)))



# Calculating Bollinger Bands (includes standard deviation in process
STD = close.rolling(20).std() * 2
top_Bollinger = SMA20 + STD
bot_Bollinger = SMA20 - STD
percent_bandwidth = (close.shift(1) - bot_Bollinger) / (top_Bollinger - bot_Bollinger)

# Not Using a for loop for analysis of every indicator at once
index = trade_pos.index
for i in range(0, len(trade_pos)):
    trade_pos['Close'][index[i]] = trade_pos['Close'][index[i]]
    if abs(RSI['Close'][index[i]] - 50) - 20 > 0:
        if np.sign(RSI['Close'][index[i]] - 50) == np.sign(trade_pos['Close'][index[i]]):
            trade_pos['Close'][index[i]] /= 2
# trade_pos['Close'][index[i]]/2 +
trade_pos_points = trade_pos.diff()
trade_pos_points[abs(trade_pos_points) < 1] = 0
trade_pos_points = trade_pos_points.replace(0, np.nan).shift(-1).apply(np.sign) * -1

# Now we are done with analysis, and storing begins:

# Renaming all DataFrame columns from close to what they actually are (so we can graph later)
SMA20 = SMA20.rename(columns={'Close': '20-Day Moving Average'})
SMA50 = SMA50.rename(columns={'Close': '50-Day Moving Average'})
top_Bollinger = top_Bollinger.rename(columns={'Close': 'Top Bollinger Band'})
bot_Bollinger = bot_Bollinger.rename(columns={'Close': 'Bottom Bollinger Band'})
percent_bandwidth = percent_bandwidth.rename(columns={'Close': '%B'})
RSI = RSI.rename(columns={'Close': 'RSI'})

trade_pos = trade_pos.rename(columns={'Close': 'Trading Position'})
trade_pos_points = trade_pos_points.rename(columns={'Close': 'Trading Position Shift Points'})

# putting it all in one DataFrame
stonk = close.join(SMA20.join(SMA50).join(top_Bollinger).join(bot_Bollinger).join(percent_bandwidth)
                   .join(RSI).join(trade_pos).join(trade_pos_points))

# writing that DataFrame to 'graph_data.xlsx', where the analyzed data sits to be graphed/displayed
with pd.ExcelWriter('graph_data.xlsx') as writer:
    stonk.to_excel(writer)
