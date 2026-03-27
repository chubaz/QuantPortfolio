# src/backtest.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_target_weights(composite_scores, top_n=10):
    """
    Converts daily composite scores into target portfolio weights.
    We rebalance at the end of each month, buying the top_n stocks in equal weights.
    """
    logging.info(f"Generating monthly target weights for top {top_n} stocks...")
    
    # Resample daily scores to Business Month End ('BME')
    monthly_scores = composite_scores.resample('BME').last()
    
    weights_list = []
    
    for date, scores in monthly_scores.iterrows():
        # Drop NaNs (e.g., stocks without enough history) and sort descending
        valid_scores = scores.dropna().sort_values(ascending=False)
        
        if valid_scores.empty:
            continue
            
        # Select the top N highest scoring stocks
        top_stocks = valid_scores.head(top_n)
        
        # Equal-weight them (1 / N)
        w = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        w.name = date
        weights_list.append(w)
        
    # Combine all monthly rows into a single DataFrame of weights over time
    target_weights = pd.concat(weights_list, axis=1).T
    
    # Fill NaN with 0 (if a stock drops out of the top N, its weight becomes 0)
    target_weights = target_weights.fillna(0.0)
    
    return target_weights


# Add this to the bottom of src/backtest.py

def calculate_portfolio_returns(target_weights, daily_returns):
    """
    Simulates the daily returns of the portfolio.
    Crucially, it shifts weights by 1 period to prevent look-ahead bias.
    """
    logging.info("Calculating daily portfolio returns...")
    
    # Reindex the monthly weights to the daily returns calendar and forward-fill.
    # This means if we rebalance on Jan 31, we hold those weights for all days in Feb.
    daily_weights = target_weights.reindex(daily_returns.index).ffill()
    
    # SHIFT BY 1: Apply the weights from day t-1 to the returns of day t.
    # This simulates executing the trade at the close of the rebalance day.
    daily_weights = daily_weights.shift(1)
    
    # Calculate daily portfolio return: Sum of (Weight * Asset Return) for each day
    portfolio_returns = (daily_weights * daily_returns).sum(axis=1)
    
    # Drop the first few days where we don't have weights yet
    portfolio_returns = portfolio_returns.loc[target_weights.index[0]:]
    
    return portfolio_returns

# src/backtest.py (Append or update these functions)

def calculate_turnover(target_weights):
    """
    Calculates the monthly turnover of the portfolio.
    Turnover = sum(abs(new_weights - old_weights)) / 2
    """
    # Calculate the difference in weights from month to month
    weight_diff = target_weights.diff().fillna(target_weights.iloc[0])
    
    # Absolute turnover (sum of all buys and sells)
    # We divide by 2 because buying $1 requires selling $1 (standard definition)
    turnover = weight_diff.abs().sum(axis=1) / 2
    return turnover

def calculate_portfolio_returns_net(target_weights, daily_returns, cost_bps=20):
    """
    Calculates portfolio returns after subtracting transaction costs 
    at each rebalance event.
    """
    # 1. Get gross daily returns (same logic as before)
    daily_weights = target_weights.reindex(daily_returns.index).ffill().shift(1)
    gross_returns = (daily_weights * daily_returns).sum(axis=1)
    
    # 2. Calculate turnover at each rebalance date
    turnover = calculate_turnover(target_weights)
    
    # 3. Convert bps to decimals (e.g., 20 bps = 0.0020)
    cost_pct = cost_bps / 10000
    
    # 4. Create a transaction cost series aligned with daily returns
    # Costs only occur on the day of rebalance
    tx_costs = pd.Series(0.0, index=gross_returns.index)
    for date, t_over in turnover.items():
        if date in tx_costs.index:
            tx_costs.loc[date] = t_over * cost_pct
            
    # 5. Net Return = Gross Return - Transaction Costs
    net_returns = gross_returns - tx_costs
    
    # Filter for the backtest period
    net_returns = net_returns.loc[target_weights.index[0]:]
    
    return net_returns, turnover

# src/backtest.py (Append or replace the return calculation function)

def calculate_portfolio_performance(target_weights, daily_returns, cost_bps=20):
    """
    Calculates both Gross and Net portfolio returns by accounting for turnover costs.
    
    Args:
        target_weights: DataFrame of monthly target weights
        daily_returns: DataFrame of daily asset returns
        cost_bps: Total transaction cost (fees + slippage) in basis points (e.g., 20 bps = 0.002)
    """
    logging.info(f"Calculating performance with {cost_bps} bps transaction cost...")
    
    # 1. Map monthly weights to daily timeframe
    daily_weights = target_weights.reindex(daily_returns.index).ffill()
    
    # 2. Calculate Gross Returns (Shift weights by 1 to avoid look-ahead bias)
    portfolio_gross_returns = (daily_weights.shift(1) * daily_returns).sum(axis=1)
    
    # 3. Calculate Turnover and Transaction Costs
    # Turnover occurs when the target weight today differs from the target weight yesterday
    # We calculate the absolute change in weights across all assets
    weight_changes = target_weights.diff().abs().sum(axis=1)
    
    # Convert bps to decimal (e.g., 20 bps -> 0.002)
    cost_pct = cost_bps / 10000
    transaction_costs = weight_changes * cost_pct
    
    # 4. Apply costs to the returns on rebalance dates
    # We create a daily cost series
    daily_costs = pd.Series(0.0, index=daily_returns.index)
    # Rebalance costs usually hit on the first trading day of the new period
    daily_costs.update(transaction_costs)
    
    portfolio_net_returns = portfolio_gross_returns - daily_costs
    
    # 5. Cleanup
    start_date = target_weights.index[0]
    return portfolio_gross_returns.loc[start_date:], portfolio_net_returns.loc[start_date:]