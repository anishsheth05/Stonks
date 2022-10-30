import pandas as pd
import yfinance as yf

ticker = str.upper(input('Enter ticker here: '))
stock = yf.download(tickers=ticker, period='2y', interval='1d', progress=True)
with pd.ExcelWriter('stock_data.xlsx') as writer:
    stock.to_excel(writer)
