# src/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sp500_tickers():
    """
    Fetches the current S&P 500 tickers from Wikipedia.
    Note: This introduces survivorship bias if backtesting far into the past.
    """
    logging.info("Fetching S&P 500 tickers from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    tickers = table['Symbol'].tolist()
    # Clean up tickers (e.g., BRK.B -> BRK-B for yfinance)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def download_price_data(tickers, start_date, end_date):
    """
    Downloads adjusted close prices and volumes for a list of tickers.
    """
    logging.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        auto_adjust=True, # Uses Adjusted Close for splits/dividends
        progress=False
    )
    
    # yfinance returns a MultiIndex column DataFrame if multiple tickers are requested
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].copy()
        volumes = data['Volume'].copy()
    else:
        # Single ticker case
        prices = pd.DataFrame(data['Close'], columns=[tickers[0]])
        volumes = pd.DataFrame(data['Volume'], columns=[tickers[0]])
        
    prices.index = pd.to_datetime(prices.index)
    volumes.index = pd.to_datetime(volumes.index)
    
    return prices, volumes

def clean_price_data(prices, max_missing_pct=0.10):
    """
    Cleans price data by removing tickers with too much missing data
    and forward-filling the rest (to prevent look-ahead bias).
    """
    logging.info(f"Shape before cleaning: {prices.shape}")
    
    # Calculate percentage of missing values per ticker
    missing_pct = prices.isnull().sum() / len(prices)
    
    # Keep only tickers with missing data below the threshold
    valid_tickers = missing_pct[missing_pct <= max_missing_pct].index
    clean_prices = prices[valid_tickers].copy()
    
    # Forward fill missing prices (e.g., trading halts), then backward fill starting NaNs
    clean_prices = clean_prices.ffill().bfill()
    
    logging.info(f"Shape after cleaning: {clean_prices.shape} ({len(prices.columns) - len(valid_tickers)} dropped)")
    
    return clean_prices

# src/data_loader.py (Add this function)

def download_fundamentals(tickers):
    """
    Downloads annual Net Income and Shareholder Equity for a list of tickers.
    Note: yfinance returns the last 4 years of annual data.
    """
    logging.info(f"Fetching fundamentals for {len(tickers)} tickers...")
    fundamental_data = {}
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # Fetch annual financials and balance sheet
            income_stmt = t.financials.T # Net Income is here
            balance_sheet = t.balance_sheet.T # Stockholders Equity is here
            
            if 'Net Income' in income_stmt.columns and 'Stockholders Equity' in balance_sheet.columns:
                df = pd.DataFrame({
                    'net_income': income_stmt['Net Income'],
                    'equity': balance_sheet['Stockholders Equity']
                })
                fundamental_data[ticker] = df
        except Exception as e:
            logging.warning(f"Could not fetch data for {ticker}: {e}")
            
    return fundamental_data