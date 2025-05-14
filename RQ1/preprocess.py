import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
import logging 

from RQ1.constants import DATA_DIR
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_forex_data(tickers: list[str], start_date: str, end_date: str, interval: str='1h') -> pd.DataFrame:
    """
    Download forex data from Yahoo Finance.
    Returns a concatenated DataFrame with a 'tic' column per pair.
    """
    df = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    if df is None:
        logging.warning("Yahoofinance couldn't download a dataframe.")
    return df

def preprocess_forex_data(df) -> pd.DataFrame:
    """
    Add technical indicators and clean data.
    """
    df = add_all_ta_features(
        df,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )
    
    # Fill missing values if any remain
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

if __name__ == "__main__":
    # Config
    tickers = ["EURUSD=X",]
    start_date = None
    end_date = None 
    interval = "1h"

    print("Downloading data...")
    df = download_forex_data(tickers, start_date, end_date, interval)
    print(f"Downloaded shape: {df.shape}")
    print(df.head())


    df.columns = df.columns.droplevel(1)
    print(df.head())

    print("Preprocessing...")
    df = preprocess_forex_data(df)
    print(f"Processed shape: {df.shape}")
    print(df.head())
    print(df.columns)
    
    save_dataframe(df, DATA_DIR / "data.csv")
