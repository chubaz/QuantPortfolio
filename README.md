📈 Quant Portfolio Architecture: Multi-Factor MVO System

An end-to-end quantitative research workflow built in Python to engineer, optimize, and deploy multi-factor equity strategies. 
This project demonstrates production-grade data sanitization, Walk-Forward hyperparameter tuning, and Mean-Variance Optimization (MVO).

🚀 Key Features
- Alpha Engine: Constructs cross-sectional Z-score signals combining Momentum (12M-1M), Quality (ROE), Low Volatility, and a synthetic ESG tilt.
- Walk-Forward Validation: Utilizes rolling Time-Series Cross-Validation to dynamically adapt factor weights to market regimes, ensuring 100% out-of-sample backtest integrity.
- SciPy-Based Optimizer: Replaces naive equal-weighting with a Sequential Least SQuares Programming (SLSQP) solver. Maximizes risk-adjusted utility subject to a 15% maximum position limit.
- Interactive Web App: A Streamlit dashboard allowing users to manipulate transaction costs (bps), weight constraints, and factor blends on the fly.

🛠️ Tech StackCore Math & Data: pandas, numpy, scipy (SLSQP solver)Financial Data: yfinance (Prices & Fundamentals)
Reporting & Visuals: quantstats, plotly, matplotlibFrontend Deployment: streamlit

🧠 Quantitative Risk Controls
- Matrix Regularization: Injects tiny constants ($1e-6$) to covariance diagonals to prevent Singular Matrix (LinAlgError) crashes.
- Outlier Clipping: Mathematically bounds ROE inputs to handle "infinity" divisions caused by zero-equity companies.
- Turnover Friction: Simulates realistic execution drag by penalizing the backtest 20 bps per gross turnover.


💻 How to Run the Dashboard Locally

git clone https://github.com/yourusername/quant-portfolio-project.git
cd quant-portfolio-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit App
streamlit run app/dashboard.py
