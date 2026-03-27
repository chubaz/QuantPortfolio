# src/factors.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_momentum(prices, lookback_days=252, skip_days=21):
    """
    Computes the momentum factor.
    Standard practice is trailing 12-month return (252 trading days),
    skipping the most recent month (21 days) to avoid the short-term reversal effect.
    """
    logging.info(f"Computing momentum (lookback={lookback_days}, skip={skip_days})...")
    
    # Price 12 months ago
    past_prices = prices.shift(lookback_days)
    # Price 1 month ago
    recent_prices = prices.shift(skip_days)
    
    # Return between t-252 and t-21
    momentum = (recent_prices / past_prices) - 1
    return momentum

def compute_volatility(returns, window=252):
    """
    Computes annualized historical volatility over a rolling window.
    """
    logging.info(f"Computing annualized rolling volatility (window={window})...")
    # Calculate rolling standard deviation of daily returns and annualize it
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def compute_cross_sectional_zscore(factor_df):
    """
    Computes the cross-sectional z-score for a given factor DataFrame.
    This standardizes the score for each stock relative to the rest of the universe ON THAT SPECIFIC DAY.
    """
    logging.info("Computing cross-sectional z-scores...")
    # Calculate daily cross-sectional mean (across columns)
    mean = factor_df.mean(axis=1)
    
    # Calculate daily cross-sectional standard deviation
    std = factor_df.std(axis=1).replace(0, np.nan) # Prevent division by zero
    
    # Z-score = (Value - Mean) / Standard Deviation
    zscores = factor_df.sub(mean, axis=0).div(std, axis=0)
    
    return zscores

# src/factors.py (Add these functions)

def compute_roe(fundamental_dict, price_index):
    """
    Computes Return on Equity (ROE) and aligns it to the daily price index.
    ROE = Net Income / Total Equity
    """
    logging.info("Computing Quality factor (ROE)...")
    roe_df = pd.DataFrame(index=price_index)
    
    for ticker, data in fundamental_dict.items():
        # Calculate ROE
        roe_series = data['net_income'] / data['equity']
        
        # Shift data forward by 90 days to avoid look-ahead bias 
        # (simulating the delay in earnings reports)
        roe_series.index = roe_series.index + pd.Timedelta(days=90)
        
        # Reindex to match daily price dates and forward-fill
        roe_df[ticker] = roe_series.reindex(price_index, method='ffill')
        
    return roe_df


# Add this to src/factors.py

def compute_synthetic_esg(tickers, price_index):
    """
    Generates synthetic historical ESG scores (0 to 100) for a list of tickers.
    In a production environment, this would be replaced by a database query 
    to an ESG vendor like MSCI or Refinitiv.
    """
    logging.info("Generating synthetic ESG scores for the universe...")
    
    # Set a seed so the scores are reproducible every time you run the backtest
    np.random.seed(42)
    
    # Generate a random ESG score between 30 and 90 for each ticker
    # (Tech/Healthcare usually score higher, Energy/Industrials lower in reality,
    # but random distribution is fine for testing the architectural tilt)
    base_scores = np.random.uniform(30, 90, size=len(tickers))
    
    # Create a DataFrame aligned to our price dates
    esg_df = pd.DataFrame(index=price_index, columns=tickers)
    
    # Apply the static score across all dates
    for i, ticker in enumerate(tickers):
        esg_df[ticker] = base_scores[i]
        
    return esg_df