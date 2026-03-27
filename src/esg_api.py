# src/esg_api.py
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_esg_data(tickers):
    """
    Connects to Yahoo Finance to fetch current Sustainalytics ESG Risk Scores.
    Lower score = Lower Risk (Better ESG Profile).
    """
    logging.info(f"Fetching live ESG data for {len(tickers)} tickers...")
    esg_data = []
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            # yfinance returns a DataFrame for sustainability
            sust = t.sustainability
            
            if sust is not None and not sust.empty:
                # Extract the Total ESG Risk Score
                # Yahoo's format changes occasionally, so we use a robust loc/get
                total_esg = sust.loc['totalEsg', 0] if 'totalEsg' in sust.index else None
                env_score = sust.loc['environmentScore', 0] if 'environmentScore' in sust.index else None
                soc_score = sust.loc['socialScore', 0] if 'socialScore' in sust.index else None
                gov_score = sust.loc['governanceScore', 0] if 'governanceScore' in sust.index else None
                
                esg_data.append({
                    'Ticker': ticker,
                    'Total_ESG_Risk': total_esg,
                    'Environment': env_score,
                    'Social': soc_score,
                    'Governance': gov_score
                })
            else:
                logging.warning(f"No ESG data available for {ticker}")
                
        except Exception as e:
            logging.error(f"Failed to fetch ESG for {ticker}: {e}")
            
    # Convert to a clean DataFrame
    df = pd.DataFrame(esg_data).set_index('Ticker')
    return df