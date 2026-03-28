## 📈 Quant Portfolio Architecture: Multi-Factor MVO System

An end-to-end quantitative research workflow built in Python to engineer, optimize, and deploy multi-factor equity strategies. 
This project demonstrates production-grade data sanitization, Walk-Forward hyperparameter tuning, and Mean-Variance Optimization (MVO).

### 🚀 Key Features
- Alpha Engine: Constructs cross-sectional Z-score signals combining Momentum (12M-1M), Quality (ROE), Low Volatility, and a synthetic ESG tilt.
- Walk-Forward Validation: Utilizes rolling Time-Series Cross-Validation to dynamically adapt factor weights to market regimes, ensuring 100% out-of-sample backtest integrity.
- SciPy-Based Optimizer: Replaces naive equal-weighting with a Sequential Least SQuares Programming (SLSQP) solver. Maximizes risk-adjusted utility subject to a 15% maximum position limit.
- Interactive Web App: A Streamlit dashboard allowing users to manipulate transaction costs (bps), weight constraints, and factor blends on the fly.

### 🛠️ Tech StackCore Math & Data: pandas, numpy, scipy (SLSQP solver)Financial Data: yfinance (Prices & Fundamentals)
Reporting & Visuals: quantstats, plotly, matplotlibFrontend Deployment: streamlit

### 🧠 Quantitative Risk Controls
- Matrix Regularization: Injects tiny constants ($1e-6$) to covariance diagonals to prevent Singular Matrix (LinAlgError) crashes.
- Outlier Clipping: Mathematically bounds ROE inputs to handle "infinity" divisions caused by zero-equity companies.
- Turnover Friction: Simulates realistic execution drag by penalizing the backtest 20 bps per gross turnover.


## ⚠️ Research Disclosure & Known Biases

To maintain transparency, the following limitations of this current research environment are disclosed:

1. **Survivorship Bias:** The current universe uses modern S&P 500 constituents. In a production environment, a "Point-in-Time" universe would be used to include companies that were subsequently delisted.
2. **Look-Ahead Bias Mitigation:** Fundamental data (ROE) is manually lagged by 90 days to simulate standard earnings reporting cycles.
3. **Execution Assumptions:** This model assumes "at-the-close" execution on the last business day of each month. In reality, market impact and slippage for large orders may vary from the flat 20bps assumption.
4. **Liquidity Filter:** The current 30-name universe is a proxy for high-liquidity stocks. A production version would require a dynamic median-daily-volume (MDV) filter.


## 💻 How to Run the Dashboard Locally

git clone https://github.com/chubaz/QuantPortfolio.git
cd quant-portfolio-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit App
streamlit run app/dashboard.py
