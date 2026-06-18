import pandas as pd
import yfinance as yf

def load_data(ticker, start_date, end_date) -> pd.DataFrame:
  data_matrix=yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
  data_matrix=data_matrix.sort_index() #sort values
  pd.DataFrame(data_matrix).to_csv("prices.csv") #save to csv
  return pd.DataFrame(data_matrix) 



