from data_loading import load_data
from optimizer import cost_inspection_backtest, evaluate_robust_weights

if __name__ =="__main__":
  # Defensive (safe-haven) ETFs
  SD = ["GLD", "TLT", "XLP", "XLU", "XLV"]

  # Aggressive (pro-cyclical) ETFs
  SA = ["XLK", "XLF", "XLY", "XLI", "XLE"]

  # Combined universe
  tickers = SD + SA
  start="2013-01-01"
  end="2026-06-01"
  
  #LOAD DATA FROM YAHOO FINANCE
  #load_data(tickers, start, end)
  
  #BACKTEST, SET start AND end to perform the backtest for financial distress periods.
  start_back=""
  end_back=""
  
  #COST INSPECTION BACKTEST
  #print("COST INSPECTION BACKTEST")
  #cost_inspection_backtest(tickers, window=252, start=start_back, end=end_back)

  #FINANCIAL DISTRESS BACKTEST ON WEIGHTS
  #COVID PERIOD
  evaluate_robust_weights(start="2020-02-01", end="2020-06-30")
  
  #INFLATION PERIOD
  #evaluate_robust_weights(start="2021-10-01", end="2022-10-31")
  
  #GEOPOLITICAL TENSION
  #evaluate_robust_weights(start="2026-02-07", end="2026-06-01")

  