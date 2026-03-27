import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import quantstats as qs
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import download_price_data, clean_price_data, download_fundamentals
from factors import (compute_momentum, compute_volatility, compute_roe, 
                     compute_synthetic_esg, compute_cross_sectional_zscore)
from signals import build_composite_signal
from optimizer import generate_optimized_weights
from backtest import calculate_portfolio_performance

# --- THEME DEFINITION (LIGHT MODE) ---
COLORS = {
    "bg_main": "#FFFFFF",
    "bg_card": "#F8F9FA",
    "bg_sidebar": "#F1F3F5",
    "accent": "#007BFF",      # Professional Blue Strategy
    "benchmark": "#DC3545",   # Standard Red Benchmark
    "text_main": "#212529",
    "text_muted": "#6C757D",
    "border": "#DEE2E6",
    "success": "#28A745",
    "warning": "#FFC107"
}

# --- SET PAGE CONFIG ---
st.set_page_config(
    page_title="AlphaStream | Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING (LIGHT MODE) ---
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: {COLORS['bg_main']};
        color: {COLORS['text_main']};
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['bg_sidebar']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Card/Metric Styling */
    div[data-testid="stMetric"] {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: {COLORS['text_muted']};
        transition: all 0.3s;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['accent']} !important;
        color: white !important;
        font-weight: bold;
    }}
    
    /* Button & Slider Styling */
    .stButton>button {{
        background-color: {COLORS['accent']};
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: bold;
        transition: opacity 0.3s;
    }}
    .stButton>button:hover {{
        opacity: 0.85;
        color: white;
    }}
    
    /* DataFrame/Table styling */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}

    /* Typography */
    h1, h2, h3 {{
        color: {COLORS['text_main']} !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .stCaption {{
        color: {COLORS['text_muted']} !important;
    }}
    
    /* Metrics Deltas */
    [data-testid="stMetricDelta"] {{
        font-weight: 500;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- PLOTLY TEMPLATE (LIGHT MODE) ---
def apply_custom_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text_main'],
        xaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border'], linecolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border'], linecolor=COLORS['border']),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- DATA PIPELINE ---
@st.cache_data(show_spinner="📥 Synchronizing with Market Data...")
def get_processed_data():
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
                'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 
                'COST', 'DIS', 'KO', 'PEP', 'CSCO', 'WMT', 'TMO', 'MCD', 'ABT', 'CRM']
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*4)).strftime('%Y-%m-%d')
    
    prices_raw, _ = download_price_data(universe, start_date, end_date)
    prices = clean_price_data(prices_raw)
    fundamentals = download_fundamentals(prices.columns)
    
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.clip(lower=-0.90, upper=1.0).dropna(how='all')
    
    z_mom = compute_cross_sectional_zscore(compute_momentum(prices).replace([np.inf, -np.inf], np.nan).dropna(how='all'))
    z_vol = -compute_cross_sectional_zscore(compute_volatility(returns).replace([np.inf, -np.inf], np.nan).dropna(how='all'))
    
    raw_roe = compute_roe(fundamentals, prices.index).replace([np.inf, -np.inf], np.nan)
    raw_roe = raw_roe.clip(lower=-50, upper=50) 
    z_roe = compute_cross_sectional_zscore(raw_roe.dropna(how='all'))
    z_esg = compute_cross_sectional_zscore(compute_synthetic_esg(prices.columns, prices.index))
    
    return prices, returns, z_mom, z_vol, z_roe, z_esg

# --- UI UTILS ---
def plot_equity_curve(strat, bench):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strat.index, y=(1+strat).cumprod(), name="Strategy", line=dict(color=COLORS['accent'], width=2.5)))
    fig.add_trace(go.Scatter(x=bench.index, y=(1+bench).cumprod(), name="S&P 500", line=dict(color=COLORS['benchmark'], width=1.5, dash='dot')))
    fig.update_layout(title="Performance: Growth of $1", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return apply_custom_theme(fig)

def plot_drawdowns(returns):
    dd = qs.stats.to_drawdown_series(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', name="Drawdown", line=dict(color=COLORS['benchmark'])))
    fig.update_layout(title="Risk: Drawdown Exposure", xaxis_title="Date", yaxis_title="Drawdown")
    return apply_custom_theme(fig)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"<h1 style='color:{COLORS['accent']}; font-size: 24px; margin-bottom: 0;'>AlphaStream</h1>", unsafe_allow_html=True)
    st.caption("Quantitative Portfolio Terminal")
    st.markdown("---")
    
    st.header("🎯 Factor Tilts")
    w_mom = st.slider("Momentum", 0, 100, 40)
    w_vol = st.slider("Low Volatility", 0, 100, 30)
    w_qual = st.slider("Quality (ROE)", 0, 100, 30)
    w_esg = st.slider("ESG Tilt", 0, 100, 0)
    
    st.markdown("---")
    st.header("⚙️ Optimization")
    max_w = st.slider("Max Stock Weight (%)", 5, 30, 15) / 100.0
    costs = st.slider("Trading Costs (bps)", 0, 100, 20)
    top_n = st.number_input("Universe Selection", 5, 30, 20)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 EXECUTE REBALANCE"):
        st.cache_data.clear()
        st.rerun()

# --- EXECUTION ---
try:
    prices, returns, z_mom, z_vol, z_roe, z_esg = get_processed_data()
    
    total = w_mom + w_vol + w_qual + w_esg
    weights = {'mom': w_mom/total, 'vol': w_vol/total, 'qual': w_qual/total, 'esg': w_esg/total} if total > 0 else {'mom': 0.25, 'vol': 0.25, 'qual': 0.25, 'esg': 0.25}

    with st.spinner("⚡ Processing Quantum Signals..."):
        scores = build_composite_signal(z_mom, z_vol, z_roe, z_esg=z_esg, weights=weights)
        opt_weights = generate_optimized_weights(scores, returns, top_n=top_n, max_weight=max_w)
        gross, net = calculate_portfolio_performance(opt_weights, returns, cost_bps=costs)

    spy = yf.download('SPY', start=net.index[0], end=net.index[-1], progress=False)['Close']
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
    spy_ret = spy.pct_change().dropna()
    common_idx = net.index.intersection(spy_ret.index)
    net_aligned, spy_aligned = net.loc[common_idx], spy_ret.loc[common_idx]

    # --- MAIN DASHBOARD ---
    st.title("AlphaStream Quant Terminal")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 PERFORMANCE", "🔬 FACTOR ANALYSIS", "⚖️ ALLOCATION", "📋 TARGETS"])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy CAGR", f"{qs.stats.cagr(net_aligned):.2%}")
        m2.metric("Sharpe Ratio", f"{qs.stats.sharpe(net_aligned):.2f}")
        m3.metric("Max Drawdown", f"{qs.stats.max_drawdown(net_aligned):.2%}")
        m4.metric("Win Rate", f"{qs.stats.win_rate(net_aligned):.1%}")

        col_main, col_side = st.columns([2, 1])
        with col_main:
            st.plotly_chart(plot_equity_curve(net_aligned, spy_aligned), use_container_width=True)
            st.plotly_chart(plot_drawdowns(net_aligned), use_container_width=True)
        with col_side:
            st.subheader("Risk Decomposition")
            greeks = qs.stats.greeks(net_aligned, spy_aligned)
            important_stats = {
                "Ann. Volatility": f"{qs.stats.volatility(net_aligned):.2%}",
                "Sortino": f"{qs.stats.sortino(net_aligned):.2f}",
                "Calmar": f"{qs.stats.calmar(net_aligned):.2f}",
                "Profit Factor": f"{qs.stats.profit_factor(net_aligned):.2f}",
                "Alpha": f"{greeks['alpha']:.2f}",
                "Beta": f"{greeks['beta']:.2f}",
            }
            st.table(pd.Series(important_stats, name="Metric Value"))
            
            monthly = qs.stats.monthly_returns(net_aligned)
            fig_heat = px.imshow(monthly, labels=dict(x="Month", y="Year", color="Return"),
                                 x=monthly.columns, y=monthly.index,
                                 color_continuous_scale=[[0, COLORS['benchmark']], [0.5, COLORS['bg_card']], [1, COLORS['success']]],
                                 text_auto=".1%")
            st.plotly_chart(apply_custom_theme(fig_heat), use_container_width=True)

    with tab2:
        col_f1, col_f2 = st.columns(2)
        latest_date = z_mom.index[-1]
        with col_f1:
            scatter_df = pd.DataFrame({'Momentum': z_mom.loc[latest_date], 'Quality': z_roe.loc[latest_date], 'Volatility': -z_vol.loc[latest_date]}).dropna()
            fig_scat = px.scatter(scatter_df, x='Momentum', y='Quality', size=np.abs(scatter_df['Volatility'])+1,
                                  hover_name=scatter_df.index, title=f"Factor Landscape: {latest_date.date()}",
                                  color='Momentum', color_continuous_scale='Bluered')
            st.plotly_chart(apply_custom_theme(fig_scat), use_container_width=True)
        with col_f2:
            factor_data = pd.DataFrame({'Momentum': z_mom.stack(), 'Low Vol': z_vol.stack(), 'Quality': z_roe.stack(), 'ESG': z_esg.stack()}).corr()
            fig_corr = px.imshow(factor_data, text_auto=".2f", title="Cross-Factor Correlation", color_continuous_scale='RdBu_r')
            st.plotly_chart(apply_custom_theme(fig_corr), use_container_width=True)

    with tab3:
        sorted_cols = opt_weights.mean().sort_values(ascending=False).index
        fig_area = px.area(opt_weights[sorted_cols], title="Historical Allocation", color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(apply_custom_theme(fig_area), use_container_width=True)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            latest_w = opt_weights.iloc[-1][opt_weights.iloc[-1] > 0.01]
            fig_pie = px.pie(values=latest_w.values, names=latest_w.index, hole=0.4, title="Current Portfolio Mix")
            st.plotly_chart(apply_custom_theme(fig_pie), use_container_width=True)
        with col_p2:
            weight_diff = opt_weights.diff().abs().sum(axis=1) / 2
            fig_turn = px.bar(weight_diff, title="Rebalance Intensity (Turnover)", color_discrete_sequence=[COLORS['accent']])
            st.plotly_chart(apply_custom_theme(fig_turn), use_container_width=True)

    with tab4:
        current_holdings = opt_weights.iloc[-1][opt_weights.iloc[-1] > 0].sort_values(ascending=False)
        holdings_df = pd.DataFrame({"Ticker": current_holdings.index, "Weight": current_holdings.values,
                                    "Mom Z": z_mom.iloc[-1].reindex(current_holdings.index),
                                    "Qual Z": z_roe.iloc[-1].reindex(current_holdings.index),
                                    "Vol Z": z_vol.iloc[-1].reindex(current_holdings.index)})
        st.dataframe(holdings_df.style.format({"Weight": "{:.2%}", "Mom Z": "{:.2f}", "Qual Z": "{:.2f}", "Vol Z": "{:.2f}"}), 
                     use_container_width=True, hide_index=True)
        st.download_button("💾 EXPORT TARGETS", holdings_df.to_csv(index=False), f"targets_{datetime.now().date()}.csv", "text/csv")

except Exception as e:
    st.error(f"⚠️ SYSTEM ERROR: {e}")
    with st.expander("Debug Trace"): st.code(e)
