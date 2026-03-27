# src/hyperopt.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_weight_grid(step=0.10):
    """
    Generates all possible combinations of 3 factor weights that sum to 1.0.
    """
    grid = []
    values = np.arange(0.0, 1.0 + step, step)
    
    for w_mom in values:
        for w_vol in values:
            for w_qual in values:
                # Keep only combinations that sum to exactly 1.0 
                if np.isclose(w_mom + w_vol + w_qual, 1.0):
                    grid.append({'mom': w_mom, 'vol': w_vol, 'qual': w_qual})
    
    logging.info(f"Generated {len(grid)} hyperparameter combinations.")
    return grid

def walk_forward_optimization(z_mom, z_vol, z_qual, prices, lookback_months=12):
    """
    Performs Walk-Forward Validation to dynamically find the best factor weights.
    Evaluates the 'Top 10' equal-weight return of each grid combo in the training window.
    """
    logging.info(f"Starting Walk-Forward Optimization (Lookback: {lookback_months}M)...")
    
    # 1. Prepare Data at a Monthly Frequency
    z_mom_m = z_mom.resample('BME').last()
    z_vol_m = z_vol.resample('BME').last()
    z_qual_m = z_qual.resample('BME').last()
    
    # Calculate 1-month forward returns for evaluation
    monthly_prices = prices.resample('BME').last()
    fwd_returns = monthly_prices.pct_change().shift(-1)
    
    grid = generate_weight_grid(step=0.10)
    dates = z_mom_m.index
    
    dynamic_scores_list = []
    weight_history = []
    
    # 2. Walk Forward Through Time
    # We start at 'lookback_months' because we need initial training data
    for i in range(lookback_months, len(dates)):
        current_date = dates[i]
        
        # The Training Window (e.g., the last 12 months)
        train_dates = dates[i - lookback_months : i]
        
        best_sharpe = -np.inf
        best_weights = None
        
        # 3. Grid Search within the Training Window
        for w in grid:
            # Build the signal for the training period
            train_scores = (w['mom'] * z_mom_m.loc[train_dates] + 
                            w['vol'] * z_vol_m.loc[train_dates] + 
                            w['qual'] * z_qual_m.loc[train_dates])
            
            # Evaluate this signal's predictive power
            strat_returns = []
            for td in train_dates:
                td_scores = train_scores.loc[td].dropna().sort_values(ascending=False)
                if len(td_scores) < 10: 
                    continue
                # Take the Top 10 stocks and see how they performed the NEXT month
                top10_tickers = td_scores.head(10).index
                td_fwd_ret = fwd_returns.loc[td, top10_tickers].mean()
                strat_returns.append(td_fwd_ret)
                
            # Calculate Training Sharpe Ratio (Annualized)
            if len(strat_returns) > 0:
                mean_ret = np.mean(strat_returns)
                std_ret = np.std(strat_returns)
                sharpe = (mean_ret / std_ret) * np.sqrt(12) if std_ret > 0 else 0
            else:
                sharpe = 0
                
            # Update winner
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w
                
        # 4. Apply the Winning Weights to the Current Out-of-Sample Month
        current_score = (best_weights['mom'] * z_mom_m.loc[[current_date]] + 
                         best_weights['vol'] * z_vol_m.loc[[current_date]] + 
                         best_weights['qual'] * z_qual_m.loc[[current_date]])
                         
        dynamic_scores_list.append(current_score)
        weight_history.append({'Date': current_date, **best_weights})
        
    # Combine results
    dynamic_composite_scores = pd.concat(dynamic_scores_list)
    weight_history_df = pd.DataFrame(weight_history).set_index('Date')
    
    return dynamic_composite_scores, weight_history_df