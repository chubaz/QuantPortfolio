# src/optimizer.py
import pandas as pd
import numpy as np
import logging
import scipy.optimize as sco

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def negative_utility(weights, expected_returns, cov_matrix, risk_aversion):
    """
    The Objective Function. 
    SciPy only knows how to MINIMIZE. To maximize utility, we minimize the NEGATIVE utility.
    """
    port_return = np.dot(weights, expected_returns)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    utility = port_return - (0.5 * risk_aversion * port_var)
    return -utility # Return negative so SciPy minimizes it

def generate_optimized_weights(composite_scores, daily_returns, top_n=20, max_weight=0.15, lookback_days=252):
    """
    Generates target weights using raw SciPy SLSQP optimization.
    """
    logging.info(f"Generating optimized weights via SciPy (Max Weight: {max_weight*100}%)...")
    
    monthly_scores = composite_scores.resample('BME').last()
    weights_list = []
    
    for date, scores in monthly_scores.iterrows():
        valid_scores = scores.dropna().sort_values(ascending=False)
        if valid_scores.empty:
            continue
            
        candidates = valid_scores.head(top_n)
        initial_tickers = candidates.index.tolist()
        
        historical_returns = daily_returns.loc[:date]
        if len(historical_returns) < lookback_days:
            continue 
            
        recent_returns = historical_returns[initial_tickers].tail(lookback_days)
        clean_returns = recent_returns.ffill().fillna(0.0)
        
        # Drop zero-variance assets
        variances = clean_returns.var()
        valid_tickers = variances[variances > 1e-8].index.tolist()
        clean_returns = clean_returns[valid_tickers]
        tickers = clean_returns.columns.tolist()
        num_assets = len(tickers)
        
        if num_assets < 3:
            w = pd.Series(1.0/min(10, len(initial_tickers)), index=initial_tickers[:10])
            w.name = date
            weights_list.append(w)
            continue

        # 1. Prepare Math Inputs (Covariance & Expected Returns)
        # We add a tiny number (1e-6) to the diagonal to ensure matrix stability (Regularization)
        cov_matrix = np.cov(clean_returns.T) + np.eye(num_assets) * 1e-6
        
        current_scores = candidates.loc[tickers]
        min_s, max_s = current_scores.min(), current_scores.max()
        if max_s > min_s:
            expected_returns = 0.01 + 0.19 * ((current_scores - min_s) / (max_s - min_s)).values
        else:
            expected_returns = np.full(num_assets, 0.10)

        # 2. Setup SciPy Constraints & Bounds
        # Constraint: Weights must sum to 1.0 (100% invested)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        
        # Bounds: Each weight must be between 0.0 and dynamically adjusted max_weight
        current_max = max_weight
        if num_assets * current_max <= 1.01:
            current_max = min(1.0, (1.0 / num_assets) + 0.05)
            
        bounds = tuple((0.0, current_max) for _ in range(num_assets))
        
        # 3. Initial Guess (Start the hiker in the middle of the property)
        initial_guess = np.full(num_assets, 1.0 / num_assets)

        # 4. Run the SLSQP Solver
        try:
            result = sco.minimize(
                negative_utility,           # Function to minimize
                initial_guess,              # Starting point
                args=(expected_returns, cov_matrix, 2.0), # Extra arguments for the function
                method='SLSQP',             # Sequential Least SQuares Programming
                bounds=bounds,              # 0 to 15% fences
                constraints=constraints,    # Sum to 1 rule
                options={'ftol': 1e-9, 'maxiter': 100} # Tolerance and limits
            )
            
            if result.success:
                # Clean up tiny floating point dust (e.g., 1e-15 becomes 0.0)
                cleaned_weights = np.where(result.x < 1e-4, 0.0, result.x)
                # Re-normalize just in case
                cleaned_weights = cleaned_weights / np.sum(cleaned_weights)
                
                w = pd.Series(cleaned_weights, index=tickers)
                w.name = date
                weights_list.append(w)
            else:
                logging.warning(f"SciPy solver failed on {date.date()}: {result.message}. Falling back.")
                w = pd.Series(1.0/min(10, len(initial_tickers)), index=initial_tickers[:10])
                w.name = date
                weights_list.append(w)
                
        except Exception as e:
            logging.warning(f"Optimization error on {date.date()}: {e}")
            w = pd.Series(1.0/min(10, len(initial_tickers)), index=initial_tickers[:10])
            w.name = date
            weights_list.append(w)

    target_weights = pd.concat(weights_list, axis=1).T
    target_weights = target_weights.fillna(0.0)
    
    return target_weights