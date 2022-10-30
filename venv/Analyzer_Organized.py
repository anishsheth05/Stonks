import pandas as pd
import numpy as np

# import statements complete, following are the function definitions


def moving_average(dataframe=pd.DataFrame(), n=20):  # easily get moving averages w/o fuss
    return dataframe.rolling(n).mean().dropna()


def relative_strength_index(dataframe=pd.DataFrame(), n=14):  # the entire code for generating RSI, for neatness
    deltas = dataframe.diff()
    up = deltas[deltas[:] > 0].fillna(0)
    down = abs(deltas[deltas[:] < 0].fillna(0))
    gain = up.rolling(n).mean().fillna(0)
    loss = down.rolling(n).mean().fillna(0)
    RSI_index = gain.index
    for k in range(n, len(gain)):
        gain['Close'][RSI_index[k]] = (gain['Close'][RSI_index[k - 1]] * (n - 1) + up['Close'][RSI_index[k]]) / n
        loss['Close'][RSI_index[k]] = (loss['Close'][RSI_index[k - 1]] * (n - 1) + down['Close'][RSI_index[k]]) / n
    relative_strength = gain / abs(loss)
    return 100 - (100 / (1 + relative_strength)), relative_strength


def percent_b(stock_prices, bottom_bollinger, top_bollinger):
    return (stock_prices - bottom_bollinger) / (top_bollinger - bottom_bollinger)


# prerequisites complete, beginning stock configuration/analysis

# getting the data from the excel sheet
stock = pd.read_excel('stock_data.xlsx', index_col=0)
close = pd.DataFrame(stock['Close'])

# Getting SMAs
SMA20 = moving_average(close, n=20)
SMA50 = moving_average(close, n=50)

# Using SMAs to initialize the trading position and its shift point DataFrames
trade_pos = pd.DataFrame(data=(SMA20 - SMA50)).apply(np.sign).fillna(0).tail(452)

# Calculating RSI
RSI, RS = relative_strength_index(close)

# Calculating Bollinger Bands (includes standard deviation in process
STD = close.rolling(20).std() * 2
top_Bollinger = SMA20 + STD
bot_Bollinger = SMA20 - STD
percent_bandwidth = percent_b(close, bot_Bollinger, top_Bollinger)

# Using a for loop for analysis of every indicator at once
index = trade_pos.index
for i in range(0, len(trade_pos)):
    trade_pos['Close'][index[i]] = trade_pos['Close'][index[i]] / 2
    trade_pos['Close'][index[i]] += np.sign(close['Close'][index[i]] - SMA20['Close'][index[i]]) / 2
    if abs(RSI['Close'][index[i]] - 50) - 20 > 0:
        if np.sign(RSI['Close'][index[i]] - 50) == np.sign(trade_pos['Close'][index[i]]):
            trade_pos['Close'][index[i]] /= 2
trade_pos_points = trade_pos.copy()
trade_pos_points[abs(trade_pos_points.diff()) < 1] = 0
trade_pos_points = trade_pos_points.replace(0, np.nan)


# Now we are done with most analysis, and renaming begins:
# Renaming all DataFrame columns from close to what they actually are (so we can graph later)
SMA20 = SMA20.rename(columns={'Close': '20-Day Moving Average'})
SMA50 = SMA50.rename(columns={'Close': '50-Day Moving Average'})
STD = STD.rename(columns={'Close': 'Standard Deviation'})
top_Bollinger = top_Bollinger.rename(columns={'Close': 'Top Bollinger Band'})
bot_Bollinger = bot_Bollinger.rename(columns={'Close': 'Bottom Bollinger Band'})
percent_bandwidth = percent_bandwidth.rename(columns={'Close': '%B'})
RSI = RSI.rename(columns={'Close': 'RSI'})

trade_pos = trade_pos.rename(columns={'Close': 'Trading Position'})
trade_pos_points = trade_pos_points.rename(columns={'Close': 'Trading Position Shift Points'})

# putting it all in one DataFrame
stonk = close.join(SMA20).join(SMA50).join(top_Bollinger).join(bot_Bollinger).join(STD).join(percent_bandwidth)\
    .join(RSI).join(trade_pos).join(trade_pos_points)

# writing that DataFrame to 'graph_data.xlsx', where the analyzed data sits to be graphed/displayed
with pd.ExcelWriter('graph_data.xlsx') as writer:
    stonk.to_excel(writer)
