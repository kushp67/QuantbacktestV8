# -*- coding: utf-8 -*- 
"""
Created on Sat Feb  1 12:14:03 2025

Enhanced by Kush Patel
Updated by ChatGPT for additional position sizing, optimization, composite strategy functionality, bug fixes,
and new strategies integration (MACD, Momentum, Mean Reversion, VWAP, Stochastic Oscillator).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import norm, ttest_ind
from statsmodels.tsa.stattools import adfuller, acf
import itertools

# ------------------------------
# Set page configuration
# ------------------------------
st.set_page_config(
    page_title="QuantBacktest Pro",
    page_icon="📈",
    layout="wide",
)

# ------------------------------
# Advanced Custom CSS Styling
# ------------------------------
def local_css():
    st.markdown(
        """
        <style>
        /* Global Styles */
        html, body {
            background: #E7F7EE; /* Mint Green background */
            font-family: 'Open Sans', sans-serif;
            color: #333333;
            margin: 0;
            padding: 0;
        }
        .reportview-container .main {
            background: #E7F7EE;
        }
        /* Custom Header */
        header {
            background-color: #5EB583; /* Accent Green */
            padding: 1rem;
            text-align: center;
            color: #FFFFFF;
            font-size: 2.5rem;
            font-weight: 700;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-bottom: 2px solid #4AA371;
            margin-bottom: 1rem;
        }
        /* Sidebar Styling */
        .css-1d391kg { 
            background-color: #FFFFFF; /* Clean White */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #E1F0E5;
        }
        /* Metric Cards */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E1F0E5;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease;
            margin: 10px;
        }
        .metric-card:hover {
            transform: scale(1.03);
        }
        .metric-card h3 {
            margin-bottom: 10px;
            color: #333333;
        }
        .metric-card p {
            font-size: 1.8rem;
            font-weight: 600;
            color: #5EB583;
        }
        /* Buttons */
        button, .stButton>button {
            background-color: #5EB583;
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        button:hover, .stButton>button:hover {
            background-color: #4AA371;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        /* Tab Headers */
        .stTabs [data-baseweb="tab-list"] button {
            font-weight: 600;
            color: #333333;
            background: #FFFFFF;
            border: 1px solid #E1F0E5;
            border-radius: 6px;
            padding: 8px 16px;
            margin-right: 4px;
            transition: background 0.3s ease;
        }
        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: #5EB583;
            color: #FFFFFF;
            border-color: #5EB583;
        }
        /* Chart Containers */
        .chart-container {
            background: #FFFFFF;
            border: 1px solid #E1F0E5;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        /* Links */
        a {
            color: #5EB583;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #E7F7EE;
        }
        ::-webkit-scrollbar-thumb {
            background: #C5E8D4;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #A9DAB6;
        }
        </style>
        """, unsafe_allow_html=True
    )

local_css()

# ------------------------------
# Data Acquisition Module
# ------------------------------
class DataFetcher:
    """
    Fetch historical stock price data using yfinance.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str, interval: str = "1d"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def fetch(self) -> pd.DataFrame:
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval)
        if data.empty:
            raise ValueError("No data fetched. Please check ticker, date range, and interval.")
        data.dropna(inplace=True)
        return data

# ------------------------------
# Strategy Builder Module (Stock Strategies)
# ------------------------------
class Strategy:
    """
    Base Strategy class. Subclasses should implement generate_signals.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")

# Original Strategies
class SMACrossoverStrategy(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

class RSITradingStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, data: pd.DataFrame) -> pd.Series:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = self.compute_rsi(data)
        sig = [1 if r < self.oversold else 0 for r in signals['rsi']]
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        return signals

class BollingerBandsStrategy(Strategy):
    def __init__(self, window: int = 20, std_multiplier: float = 2.0):
        self.window = window
        self.std_multiplier = std_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        rolling_mean = data['Close'].rolling(window=self.window, min_periods=1).mean()
        rolling_std = data['Close'].rolling(window=self.window, min_periods=1).std()
        upper_band = rolling_mean + self.std_multiplier * rolling_std
        lower_band = rolling_mean - self.std_multiplier * rolling_std
        sig = []
        current = 0
        for price, lb, ub in zip(data['Close'], lower_band, upper_band):
            if price < lb:
                current = 1
            elif price > ub:
                current = 0
            sig.append(current)
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        signals['rolling_mean'] = rolling_mean
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        return signals

class SecondDerivativeMAStrategy(Strategy):
    def __init__(self, ma_window: int = 50, threshold: float = 0.1):
        self.ma_window = ma_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['ma'] = data['Close'].rolling(window=self.ma_window, min_periods=1).mean()
        signals['second_deriv'] = signals['ma'].diff().diff()
        sig = []
        prev = 0
        for val in signals['second_deriv']:
            if pd.isna(val):
                sig.append(prev)
            elif val > self.threshold:
                prev = 1
                sig.append(1)
            elif val < -self.threshold:
                prev = 0
                sig.append(0)
            else:
                sig.append(prev)
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        return signals

class IchimokuCloudStrategy(Strategy):
    def __init__(self, conversion_period: int = 9, base_period: int = 26, leading_period: int = 52, displacement: int = 26):
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.leading_period = leading_period
        self.displacement = displacement

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        high_conv = data['High'].rolling(window=self.conversion_period, min_periods=1).max()
        low_conv = data['Low'].rolling(window=self.conversion_period, min_periods=1).min()
        conversion_line = (high_conv + low_conv) / 2
        high_base = data['High'].rolling(window=self.base_period, min_periods=1).max()
        low_base = data['Low'].rolling(window=self.base_period, min_periods=1).min()
        base_line = (high_base + low_base) / 2
        leading_span_a = ((conversion_line + base_line) / 2).shift(self.displacement)
        high_lead_b = data['High'].rolling(window=self.leading_period, min_periods=1).max()
        low_lead_b = data['Low'].rolling(window=self.leading_period, min_periods=1).min()
        leading_span_b = ((high_lead_b + low_lead_b) / 2).shift(self.displacement)
        signals['conversion_line'] = conversion_line
        signals['base_line'] = base_line
        signals['leading_span_a'] = leading_span_a
        signals['leading_span_b'] = leading_span_b
        close = data['Close']
        signals['signal'] = np.where(
            (close > leading_span_a) &
            (close > leading_span_b) &
            (conversion_line > base_line),
            1.0,
            0.0
        )
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

# New Strategies

class MACDStrategy(Strategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        ema_fast = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        signals['macd'] = macd_line
        signals['signal_line'] = signal_line
        signals['signal'] = np.where(macd_line > signal_line, 1.0, 0.0)
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

class MeanReversionStrategy(Strategy):
    def __init__(self, window: int = 20, entry_z: float = -1.5, exit_z: float = 0.0):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        
        # Ensure 'Close' is a Series. If it's a DataFrame (e.g., multiple tickers), select the first column.
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        signals['mean'] = close.rolling(window=self.window, min_periods=1).mean()
        signals['std'] = close.rolling(window=self.window, min_periods=1).std()
        signals['z_score'] = (close - signals['mean']) / (signals['std'] + 1e-9)
        
        # Generate trading signals based on the z_score thresholds.
        signals['signal'] = np.where(
            signals['z_score'] < self.entry_z, 1.0,
            np.where(signals['z_score'] > self.exit_z, 0.0, np.nan)
        )
        signals['signal'].fillna(method='ffill', inplace=True)
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

class VWAPStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['cum_price_volume'] = (data['Close'] * data['Volume']).cumsum()
        signals['cum_volume'] = data['Volume'].cumsum()
        signals['vwap'] = signals['cum_price_volume'] / (signals['cum_volume'] + 1e-9)
        signals['signal'] = np.where(data['Close'] > signals['vwap'], 1.0, 0.0)
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

class StochasticStrategy(Strategy):
    def __init__(self, k_period: int = 14, d_period: int = 3, oversold: float = 20, overbought: float = 80):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        low_min = data['Low'].rolling(window=self.k_period, min_periods=1).min()
        high_max = data['High'].rolling(window=self.k_period, min_periods=1).max()
        signals['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min + 1e-9))
        signals['%D'] = signals['%K'].rolling(window=self.d_period, min_periods=1).mean()
        signals['signal'] = np.where((signals['%K'] < self.oversold) & (signals['%K'] > signals['%D']), 1.0,
                             np.where((signals['%K'] > self.overbought) & (signals['%K'] < signals['%D']), 0.0, np.nan))
        signals['signal'].fillna(method='ffill', inplace=True)
        signals['positions'] = pd.Series(signals['signal']).diff().fillna(0.0)
        return signals

# Composite Strategy remains unchanged
class CompositeStrategy(Strategy):
    """
    Composite strategy that combines multiple individual strategies.
    The combined signal is based on a simple majority vote.
    """
    def __init__(self, strategies):
        self.strategies = strategies

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals_df = pd.DataFrame(index=data.index)
        for i, strat in enumerate(self.strategies):
            sig = strat.generate_signals(data)['signal']
            signals_df[f'signal_{i}'] = sig
        signals_df['combined_signal'] = (signals_df.sum(axis=1) >= (len(self.strategies)/2)).astype(float)
        signals_df['positions'] = signals_df['combined_signal'].diff().fillna(0.0)
        return signals_df[['combined_signal', 'positions']]

# Helper function for composite strategy generation
def composite_strategy_generate_signals(data, strategies):
    signals_df = pd.DataFrame(index=data.index)
    for i, strat in enumerate(strategies):
        sig = strat.generate_signals(data)['signal']
        signals_df[f'signal_{i}'] = sig
    signals_df['combined_signal'] = (signals_df.sum(axis=1) >= (len(strategies)/2)).astype(float)
    signals_df['positions'] = signals_df['combined_signal'].diff().fillna(0.0)
    return signals_df[['combined_signal', 'positions']]

# ------------------------------
# Stock Backtesting Engine
# ------------------------------
class Backtester:
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 100000.0, shares: int = 100,
                 position_sizing: str = "fixed", risk_fraction: float = 0.01):
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.shares = shares
        self.position_sizing = position_sizing.lower()  # "fixed", "dynamic", or "fixed fractional"
        self.risk_fraction = risk_fraction

    def run_backtest(self) -> pd.DataFrame:
        cash = self.initial_capital
        position = 0
        portfolio_values = []
        for i in range(len(self.data)):
            price = float(self.data['Close'].iloc[i])
            signal = self.signals['signal'].iloc[i] if 'signal' in self.signals.columns else self.signals['combined_signal'].iloc[i]
            if position == 0 and signal == 1:
                if self.position_sizing == "fixed":
                    shares_to_buy = self.shares
                elif self.position_sizing == "dynamic":
                    shares_to_buy = int(cash // price)
                elif self.position_sizing == "fixed fractional":
                    shares_to_buy = int((cash * self.risk_fraction) // price)
                else:
                    shares_to_buy = self.shares
                cash -= shares_to_buy * price
                position = shares_to_buy
            elif position > 0 and signal == 0:
                cash += position * price
                position = 0
            total = cash + position * price
            portfolio_values.append(total)
        portfolio = pd.DataFrame(index=self.data.index, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

    def run_backtest_custom(self, profit_target: float, stop_loss: float) -> pd.DataFrame:
        cash = self.initial_capital
        position = 0
        entry_price = None
        total_values = []
        for i in range(len(self.data)):
            price = float(self.data['Close'].iloc[i])
            signal = self.signals['signal'].iloc[i] if 'signal' in self.signals.columns else self.signals['combined_signal'].iloc[i]
            if position == 0:
                if signal == 1:
                    if self.position_sizing == "fixed":
                        shares_to_buy = self.shares
                    elif self.position_sizing == "dynamic":
                        shares_to_buy = int(cash // price)
                    elif self.position_sizing == "fixed fractional":
                        shares_to_buy = int((cash * self.risk_fraction) // price)
                    else:
                        shares_to_buy = self.shares
                    position = shares_to_buy
                    entry_price = price
                    cash -= price * shares_to_buy
            else:
                if price >= entry_price * (1 + profit_target) or price <= entry_price * (1 - stop_loss) or signal == 0:
                    cash += position * price
                    position = 0
                    entry_price = None
            total = cash + position * price
            total_values.append(total)
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['total'] = total_values
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

# ------------------------------
# Advanced Analytics Functions
# ------------------------------
def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-9)

def compute_sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate/252
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = negative_returns.std()
    return np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-9)

def compute_calmar_ratio(annual_return, max_drawdown):
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

def compute_drawdown_metrics(portfolio):
    total = portfolio['total']
    peak = total.cummax()
    drawdown = (total - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean()
    recovery_times = []
    trough = total.iloc[0]
    trough_date = total.index[0]
    for date, value in total.items():
        if value < trough:
            trough = value
            trough_date = date
        if value >= peak.loc[date] and trough < value:
            recovery_times.append((date - trough_date).days)
            trough = value
    avg_recovery = np.mean(recovery_times) if recovery_times else np.nan
    return max_drawdown, avg_drawdown, avg_recovery

def monte_carlo_simulation(returns, initial_value, num_simulations, horizon):
    simulated_values = []
    daily_returns = np.ravel(returns.dropna().values)
    for _ in range(num_simulations):
        sim_return = np.random.choice(daily_returns, size=int(horizon), replace=True)
        sim_growth = np.prod(1 + sim_return)
        simulated_values.append(initial_value * sim_growth)
    return np.array(simulated_values)

def compute_VaR_CVaR(simulated_values, confidence_level=0.95):
    VaR = np.percentile(simulated_values, (1 - confidence_level) * 100)
    CVaR = simulated_values[simulated_values <= VaR].mean()
    return VaR, CVaR

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def get_market_data(ticker="SPY", start_date="2020-01-01", end_date="2021-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def perform_statistical_tests(strategy_returns, market_returns):
    a = np.ravel(strategy_returns.dropna().values)
    b = np.ravel(market_returns.dropna().values)
    t_stat, p_value = ttest_ind(a, b, equal_var=False)
    adf_result = adfuller(strategy_returns.dropna())
    autocorr = acf(strategy_returns.dropna(), nlags=20)
    return t_stat, p_value, adf_result, autocorr

def compute_beta(strategy_returns, market_returns):
    common_index = strategy_returns.index.intersection(market_returns.index)
    if len(common_index) < 2:
        return np.nan
    a = np.ravel(strategy_returns.loc[common_index].dropna().values)
    b = np.ravel(market_returns.loc[common_index].dropna().values)
    if len(b) < 2:
        return np.nan
    covariance = np.cov(a, b)[0, 1]
    variance = np.var(b)
    return covariance / variance if variance != 0 else np.nan

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_results(portfolio: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio", color="#2980b9")
    ax.set_title("Portfolio Performance", fontsize=16, fontweight='600')
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_buy_hold_comparison(portfolio: pd.DataFrame, data: pd.DataFrame, initial_capital: float):
    bh_shares = initial_capital / data['Close'].iloc[0]
    buy_hold = bh_shares * data['Close']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio", color="#27ae60")
    ax.plot(data.index, buy_hold, label="Buy & Hold", linestyle='--', color="#c0392b")
    ax.set_title("Strategy vs. Buy & Hold Comparison", fontsize=16, fontweight='600')
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_monte_carlo(simulated_values):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(simulated_values, bins=50, alpha=0.7, color='#f39c12')
    ax.set_title("Monte Carlo Simulation: Final Portfolio Values", fontsize=16, fontweight='600')
    ax.set_xlabel("Final Portfolio Value")
    ax.set_ylabel("Frequency")
    return fig

def plot_beta_comparison(strategy_returns, market_returns):
    if hasattr(strategy_returns, "squeeze"):
        strategy_returns = strategy_returns.squeeze()
    if hasattr(market_returns, "squeeze"):
        market_returns = market_returns.squeeze()
    df = pd.DataFrame({
        'Strategy Returns': strategy_returns,
        'Market Returns': market_returns
    }).dropna()
    fig = px.scatter(
        df, 
        x='Market Returns', 
        y='Strategy Returns',
        trendline='ols',
        title='Strategy vs. Market Returns (Interactive Beta Analysis)',
        labels={'Market Returns': 'Market Returns', 'Strategy Returns': 'Strategy Returns'}
    )
    return fig

def plot_qq(returns):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(returns, line='s', ax=ax, alpha=0.5)
    ax.set_title('QQ-Plot of Strategy Returns', fontsize=16, fontweight='600')
    fig.tight_layout()
    return fig

def generate_report(portfolio, market_data, annual_return, max_dd, avg_dd, rec_time, sharpe, sortino, calmar, beta):
    report_dict = {
        "Total Return (%)": [(portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0] * 100],
        "Annualized Return (%)": [annual_return * 100],
        "Sharpe Ratio": [sharpe],
        "Sortino Ratio": [sortino],
        "Calmar Ratio": [calmar],
        "Max Drawdown (%)": [max_dd * 100],
        "Average Drawdown (%)": [avg_dd * 100],
        "Average Recovery Time (days)": [rec_time],
        "Portfolio Beta": [beta]
    }
    report_df = pd.DataFrame(report_dict)
    return report_df

# ------------------------------
# Optimization Functions
# ------------------------------
def optimize_strategy(strategy_name, data, initial_capital, shares, sizing_method, risk_fraction, metric="Total Return", composite_choices=None):
    best_metric = -np.inf
    best_params = None
    best_portfolio = None

    if strategy_name in ["SMA Crossover", "Custom Profit/Stop"]:
        for short_window in [20, 30, 40, 50]:
            for long_window in [100, 150, 200, 250]:
                if short_window >= long_window:
                    continue
                strat = SMACrossoverStrategy(short_window=short_window, long_window=long_window)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                if strategy_name == "Custom Profit/Stop":
                    for pt in [0.05, 0.10, 0.15]:
                        for sl in [0.03, 0.05, 0.10]:
                            portfolio = backtester.run_backtest_custom(profit_target=pt, stop_loss=sl)
                            total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                            days = (portfolio.index[-1] - portfolio.index[0]).days
                            annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                            sharpe = compute_sharpe_ratio(portfolio['returns'])
                            score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                            if score > best_metric:
                                best_metric = score
                                best_params = {"short_window": short_window, "long_window": long_window,
                                               "profit_target": pt, "stop_loss": sl}
                                best_portfolio = portfolio
                else:
                    portfolio = backtester.run_backtest()
                    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                    days = (portfolio.index[-1] - portfolio.index[0]).days
                    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                    sharpe = compute_sharpe_ratio(portfolio['returns'])
                    score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                    if score > best_metric:
                        best_metric = score
                        best_params = {"short_window": short_window, "long_window": long_window}
                        best_portfolio = portfolio

    elif strategy_name == "RSI Trading":
        for period in [10, 14, 20]:
            for oversold in [20, 30, 40]:
                for overbought in [60, 70, 80]:
                    if oversold >= overbought:
                        continue
                    strat = RSITradingStrategy(period=period, oversold=oversold, overbought=overbought)
                    signals = strat.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                    portfolio = backtester.run_backtest()
                    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                    days = (portfolio.index[-1] - portfolio.index[0]).days
                    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                    sharpe = compute_sharpe_ratio(portfolio['returns'])
                    score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                    if score > best_metric:
                        best_metric = score
                        best_params = {"period": period, "oversold": oversold, "overbought": overbought}
                        best_portfolio = portfolio

    elif strategy_name == "Bollinger Bands":
        for window in [10, 20, 30]:
            for std_multiplier in [1.5, 2.0, 2.5]:
                strat = BollingerBandsStrategy(window=window, std_multiplier=std_multiplier)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                portfolio = backtester.run_backtest()
                total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                days = (portfolio.index[-1] - portfolio.index[0]).days
                annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                sharpe = compute_sharpe_ratio(portfolio['returns'])
                score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                if score > best_metric:
                    best_metric = score
                    best_params = {"window": window, "std_multiplier": std_multiplier}
                    best_portfolio = portfolio

    elif strategy_name == "Second Derivative MA":
        for ma_window in [10, 20, 30, 50]:
            for threshold in [0.1, 0.5, 1.0]:
                strat = SecondDerivativeMAStrategy(ma_window=ma_window, threshold=threshold)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                portfolio = backtester.run_backtest()
                total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                days = (portfolio.index[-1] - portfolio.index[0]).days
                annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                sharpe = compute_sharpe_ratio(portfolio['returns'])
                score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                if score > best_metric:
                    best_metric = score
                    best_params = {"ma_window": ma_window, "threshold": threshold}
                    best_portfolio = portfolio

    elif strategy_name == "Ichimoku Cloud":
        for conv in [7, 9, 11]:
            for base in [22, 26, 30]:
                for lead in [48, 52, 56]:
                    for disp in [22, 26, 30]:
                        strat = IchimokuCloudStrategy(conversion_period=conv, base_period=base, leading_period=lead, displacement=disp)
                        signals = strat.generate_signals(data)
                        backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                        portfolio = backtester.run_backtest()
                        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                        days = (portfolio.index[-1] - portfolio.index[0]).days
                        annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                        sharpe = compute_sharpe_ratio(portfolio['returns'])
                        score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                        if score > best_metric:
                            best_metric = score
                            best_params = {"conversion_period": conv, "base_period": base, "leading_period": lead, "displacement": disp}
                            best_portfolio = portfolio

    # New Strategy: MACD
    elif strategy_name == "MACD":
        for fast in [10, 12, 14]:
            for slow in [24, 26, 28]:
                if fast >= slow:
                    continue
                for signal_period in [7, 9, 11]:
                    strat = MACDStrategy(fast_period=fast, slow_period=slow, signal_period=signal_period)
                    signals = strat.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                    portfolio = backtester.run_backtest()
                    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                    days = (portfolio.index[-1] - portfolio.index[0]).days
                    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                    sharpe = compute_sharpe_ratio(portfolio['returns'])
                    score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                    if score > best_metric:
                        best_metric = score
                        best_params = {"fast_period": fast, "slow_period": slow, "signal_period": signal_period}
                        best_portfolio = portfolio

    # New Strategy: Momentum
    elif strategy_name == "Momentum":
        for window in [10, 14, 20]:
            for threshold in [0.0, 1.0, 2.0]:
                strat = MomentumStrategy(window=window, threshold=threshold)
                signals = strat.generate_signals(data)
                backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                portfolio = backtester.run_backtest()
                total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                days = (portfolio.index[-1] - portfolio.index[0]).days
                annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                sharpe = compute_sharpe_ratio(portfolio['returns'])
                score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                if score > best_metric:
                    best_metric = score
                    best_params = {"window": window, "threshold": threshold}
                    best_portfolio = portfolio

    # New Strategy: Mean Reversion
    elif strategy_name == "Mean Reversion":
        for window in [20, 30, 40]:
            for entry_z in [-2.0, -1.5, -1.0]:
                for exit_z in [0.0, 0.5]:
                    strat = MeanReversionStrategy(window=window, entry_z=entry_z, exit_z=exit_z)
                    signals = strat.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                    portfolio = backtester.run_backtest()
                    total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                    days = (portfolio.index[-1] - portfolio.index[0]).days
                    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                    sharpe = compute_sharpe_ratio(portfolio['returns'])
                    score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                    if score > best_metric:
                        best_metric = score
                        best_params = {"window": window, "entry_z": entry_z, "exit_z": exit_z}
                        best_portfolio = portfolio

    # New Strategy: VWAP (no parameters to tune)
    elif strategy_name == "VWAP":
        strat = VWAPStrategy()
        signals = strat.generate_signals(data)
        backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
        portfolio = backtester.run_backtest()
        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
        days = (portfolio.index[-1] - portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
        sharpe = compute_sharpe_ratio(portfolio['returns'])
        best_metric = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
        best_params = {}  # No tunable parameters
        best_portfolio = portfolio

    # New Strategy: Stochastic Oscillator
    elif strategy_name == "Stochastic Oscillator":
        for k_period in [10, 14, 20]:
            for d_period in [3, 5]:
                for oversold in [20, 25]:
                    for overbought in [75, 80]:
                        strat = StochasticStrategy(k_period=k_period, d_period=d_period, oversold=oversold, overbought=overbought)
                        signals = strat.generate_signals(data)
                        backtester = Backtester(data, signals, initial_capital, shares, sizing_method, risk_fraction)
                        portfolio = backtester.run_backtest()
                        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
                        days = (portfolio.index[-1] - portfolio.index[0]).days
                        annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
                        sharpe = compute_sharpe_ratio(portfolio['returns'])
                        score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
                        if score > best_metric:
                            best_metric = score
                            best_params = {"k_period": k_period, "d_period": d_period, "oversold": oversold, "overbought": overbought}
                            best_portfolio = portfolio

    elif strategy_name == "Composite Strategy":
        # composite_choices is a list of strategies to combine
        param_grids = {}
        if "SMA Crossover" in composite_choices:
            param_grids["SMA Crossover"] = [{"short_window": s, "long_window": l} for s in [20,30,40,50] for l in [100,150,200,250] if s < l]
        if "RSI Trading" in composite_choices:
            param_grids["RSI Trading"] = [{"period": p, "oversold": o, "overbought": ob} for p in [10,14,20] for o in [20,30,40] for ob in [60,70,80] if o < ob]
        if "Bollinger Bands" in composite_choices:
            param_grids["Bollinger Bands"] = [{"window": w, "std_multiplier": m} for w in [10,20,30] for m in [1.5,2.0,2.5]]
        if "Second Derivative MA" in composite_choices:
            param_grids["Second Derivative MA"] = [{"ma_window": mw, "threshold": t} for mw in [10,20,30,50] for t in [0.1,0.5,1.0]]
        if "Ichimoku Cloud" in composite_choices:
            param_grids["Ichimoku Cloud"] = [{"conversion_period": cp, "base_period": bp, "leading_period": lp, "displacement": d} for cp in [7,9,11] for bp in [22,26,30] for lp in [48,52,56] for d in [22,26,30]]
        if "MACD" in composite_choices:
            param_grids["MACD"] = [{"fast_period": f, "slow_period": s, "signal_period": sp} for f in [10,12,14] for s in [24,26,28] for sp in [7,9,11] if f < s]
        if "Momentum" in composite_choices:
            param_grids["Momentum"] = [{"window": w, "threshold": t} for w in [10,14,20] for t in [0.0, 1.0, 2.0]]
        if "Mean Reversion" in composite_choices:
            param_grids["Mean Reversion"] = [{"window": w, "entry_z": ez, "exit_z": exz} for w in [20,30,40] for ez in [-2.0,-1.5,-1.0] for exz in [0.0,0.5]]
        if "VWAP" in composite_choices:
            param_grids["VWAP"] = [{}]
        if "Stochastic Oscillator" in composite_choices:
            param_grids["Stochastic Oscillator"] = [{"k_period": k, "d_period": d, "oversold": o, "overbought": ob} for k in [10,14,20] for d in [3,5] for o in [20,25] for ob in [75,80]]
        grids = [param_grids[strat] for strat in composite_choices]
        for combo in itertools.product(*grids):
            strategy_instances = []
            params_used = {}
            for strat_name, params in zip(composite_choices, combo):
                params_used[strat_name] = params
                if strat_name == "SMA Crossover":
                    strategy_instances.append(SMACrossoverStrategy(short_window=params["short_window"], long_window=params["long_window"]))
                elif strat_name == "RSI Trading":
                    strategy_instances.append(RSITradingStrategy(period=params["period"], oversold=params["oversold"], overbought=params["overbought"]))
                elif strat_name == "Bollinger Bands":
                    strategy_instances.append(BollingerBandsStrategy(window=params["window"], std_multiplier=params["std_multiplier"]))
                elif strat_name == "Second Derivative MA":
                    strategy_instances.append(SecondDerivativeMAStrategy(ma_window=params["ma_window"], threshold=params["threshold"]))
                elif strat_name == "Ichimoku Cloud":
                    strategy_instances.append(IchimokuCloudStrategy(conversion_period=params["conversion_period"],
                                                                     base_period=params["base_period"],
                                                                     leading_period=params["leading_period"],
                                                                     displacement=params["displacement"]))
                elif strat_name == "MACD":
                    strategy_instances.append(MACDStrategy(fast_period=params["fast_period"],
                                                           slow_period=params["slow_period"],
                                                           signal_period=params["signal_period"]))
                elif strat_name == "Momentum":
                    strategy_instances.append(MomentumStrategy(window=params["window"], threshold=params["threshold"]))
                elif strat_name == "Mean Reversion":
                    strategy_instances.append(MeanReversionStrategy(window=params["window"],
                                                                      entry_z=params["entry_z"],
                                                                      exit_z=params["exit_z"]))
                elif strat_name == "VWAP":
                    strategy_instances.append(VWAPStrategy())
                elif strat_name == "Stochastic Oscillator":
                    strategy_instances.append(StochasticStrategy(k_period=params["k_period"],
                                                                   d_period=params["d_period"],
                                                                   oversold=params["oversold"],
                                                                   overbought=params["overbought"]))
            composite_signals = composite_strategy_generate_signals(data, strategy_instances)
            backtester = Backtester(data, composite_signals, initial_capital, shares, sizing_method, risk_fraction)
            portfolio = backtester.run_backtest()
            total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
            days = (portfolio.index[-1] - portfolio.index[0]).days
            annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
            sharpe = compute_sharpe_ratio(portfolio['returns'])
            score = total_return if metric=="Total Return" else annual_return if metric=="Annualized Return" else sharpe
            if score > best_metric:
                best_metric = score
                best_params = params_used.copy()
                best_portfolio = portfolio

    return best_params, best_metric, best_portfolio


# ------------------------------
# Streamlit Web App
# ------------------------------
def main():
    # Custom Header
    st.markdown("<header>QuantBacktest Pro</header>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 1.2rem; margin-bottom: 1rem;'>A professional quantitative backtesting platform for stocks trading strategies.</div>", unsafe_allow_html=True)

    # Sidebar – Enhanced Backtest Settings
    st.sidebar.header("Backtest Settings")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2021, 1, 1))
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
    
    # New Sidebar: Select Data Interval
    interval = st.sidebar.selectbox("Data Interval", options=["1m", "5m", "15m", "1h", "1d"], index=4,
                                    help="Select the frequency for data updates (e.g., minute, hourly, daily).")

    # Position Sizing Method selection
    position_sizing_method = st.sidebar.radio("Position Sizing Method", options=["Fixed", "Dynamic", "Fixed Fractional"])
    if position_sizing_method == "Fixed":
        shares = st.sidebar.number_input("Number of Shares", value=100, step=1)
    else:
        shares = 0  # Not used for dynamic or fixed fractional
    
    risk_fraction = 0.01
    if position_sizing_method == "Fixed Fractional":
        risk_fraction = st.sidebar.slider("Risk Fraction", min_value=0.01, max_value=0.5, value=0.01, step=0.01)

    # Optimization toggle
    optimize = st.sidebar.checkbox("Optimize Strategy Parameters")
    if optimize:
        metric_choice = st.sidebar.selectbox("Optimization Metric", ["Total Return", "Sharpe Ratio", "Annualized Return"])

    # Strategy selection: include new strategies in the options
    strategy_options = [
        "SMA Crossover", "RSI Trading", "Bollinger Bands", "Custom Profit/Stop", 
        "Second Derivative MA", "Ichimoku Cloud", "MACD", "Momentum", 
        "Mean Reversion", "VWAP", "Stochastic Oscillator", "Composite Strategy"
    ]
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)

    # For Composite Strategy, allow selection of multiple strategies
    composite_choices = None
    if selected_strategy == "Composite Strategy":
        composite_choices = st.sidebar.multiselect("Select strategies to combine", 
                                                   options=[
                                                       "SMA Crossover", "RSI Trading", "Bollinger Bands", 
                                                       "Second Derivative MA", "Ichimoku Cloud", "MACD", 
                                                       "Momentum", "Mean Reversion", "VWAP", "Stochastic Oscillator"
                                                   ],
                                                   default=["SMA Crossover", "RSI Trading"])
        if len(composite_choices) < 2:
            st.error("Please select at least two strategies for the composite strategy.")
            return

    portfolio = None
    try:
        # Pass the selected interval to the DataFetcher
        fetcher = DataFetcher(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), interval=interval)
        data = fetcher.fetch()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Parameter inputs for individual strategies (if not composite)
    if selected_strategy != "Composite Strategy" and not optimize:
        if selected_strategy in ["SMA Crossover", "Custom Profit/Stop"]:
            st.sidebar.subheader("SMA Parameters")
            sma_short = st.sidebar.slider("Short Window", min_value=5, max_value=100, value=50)
            sma_long = st.sidebar.slider("Long Window", min_value=20, max_value=300, value=200)
        if selected_strategy == "RSI Trading":
            st.sidebar.subheader("RSI Parameters")
            rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14)
            oversold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30)
            overbought = st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70)
        if selected_strategy == "Bollinger Bands":
            st.sidebar.subheader("Bollinger Bands Parameters")
            bb_window = st.sidebar.slider("Window", min_value=10, max_value=100, value=20)
            bb_std_multiplier = st.sidebar.slider("Std Dev Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        if selected_strategy == "Second Derivative MA":
            st.sidebar.subheader("Second Derivative MA Parameters")
            sd_ma_window = st.sidebar.slider("MA Window", min_value=5, max_value=100, value=50)
            sd_threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=10.0, value=0.1, step=0.1)
        if selected_strategy == "Ichimoku Cloud":
            st.sidebar.subheader("Ichimoku Cloud Parameters")
            conv_period = st.sidebar.slider("Conversion Period", min_value=5, max_value=15, value=9)
            base_period = st.sidebar.slider("Base Period", min_value=20, max_value=40, value=26)
            lead_period = st.sidebar.slider("Leading Period", min_value=40, max_value=60, value=52)
            displacement = st.sidebar.slider("Displacement", min_value=10, max_value=30, value=26)
        if selected_strategy == "MACD":
            st.sidebar.subheader("MACD Parameters")
            macd_fast = st.sidebar.slider("Fast Period", min_value=5, max_value=20, value=12)
            macd_slow = st.sidebar.slider("Slow Period", min_value=20, max_value=40, value=26)
            macd_signal = st.sidebar.slider("Signal Period", min_value=5, max_value=15, value=9)
        if selected_strategy == "Momentum":
            st.sidebar.subheader("Momentum Parameters")
            momentum_window = st.sidebar.slider("Window", min_value=5, max_value=30, value=14)
            momentum_threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        if selected_strategy == "Mean Reversion":
            st.sidebar.subheader("Mean Reversion Parameters")
            mr_window = st.sidebar.slider("Window", min_value=10, max_value=50, value=20)
            mr_entry = st.sidebar.slider("Entry Z-Score", min_value=-3.0, max_value=0.0, value=-1.5, step=0.1)
            mr_exit = st.sidebar.slider("Exit Z-Score", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        if selected_strategy == "Stochastic Oscillator":
            st.sidebar.subheader("Stochastic Oscillator Parameters")
            sto_k = st.sidebar.slider("%K Period", min_value=5, max_value=20, value=14)
            sto_d = st.sidebar.slider("%D Period", min_value=2, max_value=10, value=3)
            sto_oversold = st.sidebar.slider("Oversold Level", min_value=10, max_value=40, value=20)
            sto_overbought = st.sidebar.slider("Overbought Level", min_value=60, max_value=90, value=80)
        # For "Custom Profit/Stop"
        if selected_strategy == "Custom Profit/Stop":
            st.sidebar.subheader("Profit-taking / Stop-loss Settings")
            profit_target = st.sidebar.slider("Profit Target (%)", min_value=1, max_value=50, value=10) / 100.0
            stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=1, max_value=50, value=5) / 100.0
    # For composite strategy manual input, let the user supply parameters for each chosen strategy
    composite_params = {}
    if selected_strategy == "Composite Strategy" and not optimize:
        for strat in composite_choices:
            st.sidebar.subheader(f"{strat} Parameters")
            if strat == "SMA Crossover":
                composite_params["SMA Crossover"] = {
                    "short_window": st.sidebar.slider("SMA Short Window", min_value=5, max_value=100, value=50, key="comp_sma_short"),
                    "long_window": st.sidebar.slider("SMA Long Window", min_value=20, max_value=300, value=200, key="comp_sma_long")
                }
            elif strat == "RSI Trading":
                composite_params["RSI Trading"] = {
                    "period": st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14, key="comp_rsi_period"),
                    "oversold": st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30, key="comp_rsi_oversold"),
                    "overbought": st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70, key="comp_rsi_overbought")
                }
            elif strat == "Bollinger Bands":
                composite_params["Bollinger Bands"] = {
                    "window": st.sidebar.slider("BB Window", min_value=10, max_value=100, value=20, key="comp_bb_window"),
                    "std_multiplier": st.sidebar.slider("BB Std Dev Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1, key="comp_bb_multiplier")
                }
            elif strat == "Second Derivative MA":
                composite_params["Second Derivative MA"] = {
                    "ma_window": st.sidebar.slider("SD MA Window", min_value=5, max_value=100, value=50, key="comp_sd_ma"),
                    "threshold": st.sidebar.slider("SD Threshold", min_value=0.0, max_value=10.0, value=0.1, step=0.1, key="comp_sd_threshold")
                }
            elif strat == "Ichimoku Cloud":
                composite_params["Ichimoku Cloud"] = {
                    "conversion_period": st.sidebar.slider("Conversion Period", min_value=5, max_value=15, value=9, key="comp_conv"),
                    "base_period": st.sidebar.slider("Base Period", min_value=20, max_value=40, value=26, key="comp_base"),
                    "leading_period": st.sidebar.slider("Leading Period", min_value=40, max_value=60, value=52, key="comp_lead"),
                    "displacement": st.sidebar.slider("Displacement", min_value=10, max_value=30, value=26, key="comp_disp")
                }
            elif strat == "MACD":
                composite_params["MACD"] = {
                    "fast_period": st.sidebar.slider("MACD Fast Period", min_value=5, max_value=20, value=12, key="comp_macd_fast"),
                    "slow_period": st.sidebar.slider("MACD Slow Period", min_value=20, max_value=40, value=26, key="comp_macd_slow"),
                    "signal_period": st.sidebar.slider("MACD Signal Period", min_value=5, max_value=15, value=9, key="comp_macd_signal")
                }
            elif strat == "Momentum":
                composite_params["Momentum"] = {
                    "window": st.sidebar.slider("Momentum Window", min_value=5, max_value=30, value=14, key="comp_mom_window"),
                    "threshold": st.sidebar.slider("Momentum Threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="comp_mom_threshold")
                }
            elif strat == "Mean Reversion":
                composite_params["Mean Reversion"] = {
                    "window": st.sidebar.slider("Mean Reversion Window", min_value=10, max_value=50, value=20, key="comp_mr_window"),
                    "entry_z": st.sidebar.slider("Entry Z-Score", min_value=-3.0, max_value=0.0, value=-1.5, step=0.1, key="comp_mr_entry"),
                    "exit_z": st.sidebar.slider("Exit Z-Score", min_value=0.0, max_value=3.0, value=0.0, step=0.1, key="comp_mr_exit")
                }
            elif strat == "VWAP":
                composite_params["VWAP"] = {}  # No additional parameters
            elif strat == "Stochastic Oscillator":
                composite_params["Stochastic Oscillator"] = {
                    "k_period": st.sidebar.slider("Stochastic %K Period", min_value=5, max_value=20, value=14, key="comp_sto_k"),
                    "d_period": st.sidebar.slider("Stochastic %D Period", min_value=2, max_value=10, value=3, key="comp_sto_d"),
                    "oversold": st.sidebar.slider("Oversold Level", min_value=10, max_value=40, value=20, key="comp_sto_oversold"),
                    "overbought": st.sidebar.slider("Overbought Level", min_value=60, max_value=90, value=80, key="comp_sto_overbought")
                }
    
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            if optimize:
                if selected_strategy == "Composite Strategy":
                    best_params, best_metric, portfolio = optimize_strategy(
                        selected_strategy, data, initial_capital, shares,
                        position_sizing_method.lower(), risk_fraction,
                        metric=metric_choice, composite_choices=composite_choices
                    )
                    st.sidebar.markdown(f"**Optimal Parameters Found:** {best_params}")
                    st.sidebar.markdown(f"**Best {metric_choice}:** {best_metric:.4f}")
                else:
                    best_params, best_metric, portfolio = optimize_strategy(
                        selected_strategy, data, initial_capital, shares,
                        position_sizing_method.lower(), risk_fraction,
                        metric=metric_choice
                    )
                    st.sidebar.markdown(f"**Optimal Parameters Found:** {best_params}")
                    st.sidebar.markdown(f"**Best {metric_choice}:** {best_metric:.4f}")
            else:
                if selected_strategy == "SMA Crossover":
                    strategy = SMACrossoverStrategy(short_window=sma_short, long_window=sma_long)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    if selected_strategy == "Custom Profit/Stop":
                        portfolio = backtester.run_backtest_custom(profit_target, stop_loss)
                    else:
                        portfolio = backtester.run_backtest()
                elif selected_strategy == "RSI Trading":
                    strategy = RSITradingStrategy(period=rsi_period, oversold=oversold, overbought=overbought)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Bollinger Bands":
                    strategy = BollingerBandsStrategy(window=bb_window, std_multiplier=bb_std_multiplier)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Second Derivative MA":
                    strategy = SecondDerivativeMAStrategy(ma_window=sd_ma_window, threshold=sd_threshold)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Ichimoku Cloud":
                    strategy = IchimokuCloudStrategy(conversion_period=conv_period, base_period=base_period, leading_period=lead_period, displacement=displacement)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "MACD":
                    strategy = MACDStrategy(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Momentum":
                    strategy = MomentumStrategy(window=momentum_window, threshold=momentum_threshold)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Mean Reversion":
                    strategy = MeanReversionStrategy(window=mr_window, entry_z=mr_entry, exit_z=mr_exit)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "VWAP":
                    strategy = VWAPStrategy()
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Stochastic Oscillator":
                    strategy = StochasticStrategy(k_period=sto_k, d_period=sto_d, oversold=sto_oversold, overbought=sto_overbought)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()
                elif selected_strategy == "Custom Profit/Stop":
                    strategy = SMACrossoverStrategy(short_window=sma_short, long_window=sma_long)
                    signals = strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest_custom(profit_target, stop_loss)
                elif selected_strategy == "Composite Strategy":
                    strategy_instances = []
                    for strat in composite_choices:
                        params = composite_params[strat]
                        if strat == "SMA Crossover":
                            strategy_instances.append(SMACrossoverStrategy(short_window=params["short_window"], long_window=params["long_window"]))
                        elif strat == "RSI Trading":
                            strategy_instances.append(RSITradingStrategy(period=params["period"], oversold=params["oversold"], overbought=params["overbought"]))
                        elif strat == "Bollinger Bands":
                            strategy_instances.append(BollingerBandsStrategy(window=params["window"], std_multiplier=params["std_multiplier"]))
                        elif strat == "Second Derivative MA":
                            strategy_instances.append(SecondDerivativeMAStrategy(ma_window=params["ma_window"], threshold=params["threshold"]))
                        elif strat == "Ichimoku Cloud":
                            strategy_instances.append(IchimokuCloudStrategy(conversion_period=params["conversion_period"], base_period=params["base_period"], leading_period=params["leading_period"], displacement=params["displacement"]))
                        elif strat == "MACD":
                            strategy_instances.append(MACDStrategy(fast_period=params["fast_period"], slow_period=params["slow_period"], signal_period=params["signal_period"]))
                        elif strat == "Momentum":
                            strategy_instances.append(MomentumStrategy(window=params["window"], threshold=params["threshold"]))
                        elif strat == "Mean Reversion":
                            strategy_instances.append(MeanReversionStrategy(window=params["window"], entry_z=params["entry_z"], exit_z=params["exit_z"]))
                        elif strat == "VWAP":
                            strategy_instances.append(VWAPStrategy())
                        elif strat == "Stochastic Oscillator":
                            strategy_instances.append(StochasticStrategy(k_period=params["k_period"], d_period=params["d_period"], oversold=params["oversold"], overbought=params["overbought"]))
                    composite_strategy = CompositeStrategy(strategy_instances)
                    signals = composite_strategy.generate_signals(data)
                    backtester = Backtester(data, signals, initial_capital, shares, position_sizing_method.lower(), risk_fraction)
                    portfolio = backtester.run_backtest()

    if portfolio is not None:
        st.subheader("Performance Summary")
        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
        days = (portfolio.index[-1] - portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
        sharpe = compute_sharpe_ratio(portfolio['returns'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Return</h3><p>{total_return * 100:.2f}%</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Annualized Return</h3><p>{annual_return * 100:.2f}%</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Sharpe Ratio</h3><p>{sharpe:.2f}</p></div>', unsafe_allow_html=True)
        max_dd, avg_dd, rec_time = compute_drawdown_metrics(portfolio)
        sortino = compute_sortino_ratio(portfolio['returns'])
        calmar = compute_calmar_ratio(annual_return, max_dd)
        st.markdown("---")
        st.subheader("Strategy vs. Buy & Hold Comparison")
        fig2 = plot_buy_hold_comparison(portfolio, data, initial_capital)
        st.pyplot(fig2)
        st.markdown("### Advanced Analytics Dashboard")
        st.markdown("Explore detailed analytics in the tabs below:")
        market_data = get_market_data("SPY", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        market_returns = market_data['Close'].pct_change().fillna(0.0)
        beta = compute_beta(portfolio['returns'], market_returns)
        tabs = st.tabs([
            "Performance Overview", 
            "Beta Analysis", 
            "QQ Plot", 
            "Monte Carlo Simulation", 
            "Statistical Edge", 
            "Hedge Optimization", 
            "Options Greeks", 
            "Export Report"
        ])
        with tabs[0]:
            st.markdown("#### Performance Overview")
            st.write(f"**Annualized Return (%):** {annual_return * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
            st.write(f"**Sortino Ratio:** {sortino:.2f}")
            st.write(f"**Calmar Ratio:** {calmar:.2f}")
            st.write(f"**Max Drawdown (%):** {max_dd * 100:.2f}%")
            st.write(f"**Average Drawdown (%):** {avg_dd * 100:.2f}%")
            st.write(f"**Average Recovery Time (days):** {rec_time:.1f}")
            st.write(f"**Portfolio Beta (vs. SPY):** {beta:.2f}")
        with tabs[1]:
            st.markdown("#### Beta Analysis")
            beta_fig = plot_beta_comparison(portfolio['returns'], market_returns)
            st.plotly_chart(beta_fig, use_container_width=True)
        with tabs[2]:
            st.markdown("#### QQ Plot")
            qq_fig = plot_qq(portfolio['returns'].dropna())
            st.pyplot(qq_fig)
        with tabs[3]:
            st.markdown("#### Monte Carlo Simulation & Risk Metrics")
            num_simulations = 1000
            horizon = 252
            simulated_vals = monte_carlo_simulation(portfolio['returns'], portfolio['total'].iloc[-1], num_simulations, horizon)
            fig_mc = plot_monte_carlo(simulated_vals)
            st.pyplot(fig_mc)
            VaR, CVaR = compute_VaR_CVaR(simulated_vals, confidence_level=0.95)
            st.write(f"Value at Risk (95%): {VaR:.2f}")
            st.write(f"Conditional VaR (95%): {CVaR:.2f}")
            blowup_prob = np.mean(simulated_vals < initial_capital) * 100
            st.write(f"Blowup Probability (% final < initial): {blowup_prob:.2f}%")
        with tabs[4]:
            st.markdown("#### Statistical Edge & Market Comparison")
            t_stat, p_value, adf_result, autocorr = perform_statistical_tests(portfolio['returns'], market_returns)
            st.write(f"t-test Statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
            st.write("ADF Test Result:")
            st.write(adf_result)
            st.write("Autocorrelation (first 10 lags):")
            st.write(autocorr[:10])
        with tabs[5]:
            st.markdown("#### Hedge Optimization")
            st.write(f"Portfolio Beta (vs. SPY): {beta:.2f}")
            if beta > 1:
                st.write("Suggestion: The portfolio is more volatile than SPY. Consider hedging with SPY put options or other risk mitigation strategies.")
            elif beta < 1:
                st.write("Suggestion: The portfolio is less volatile than SPY.")
            else:
                st.write("Suggestion: The portfolio beta is close to 1, matching the market.")
        with tabs[6]:
            st.markdown("#### Options Greeks Tracking")
            st.markdown("Enter parameters below to compute options Greeks:")
            S = st.number_input("Underlying Price (S)", value=float(data['Close'].iloc[-1]))
            K = st.number_input("Strike Price (K)", value=float(data['Close'].iloc[-1]))
            T = st.number_input("Time to Expiration (years)", value=0.25, step=0.01)
            r = st.number_input("Risk-Free Rate (annual)", value=0.02, step=0.001)
            sigma = st.number_input("Volatility (annual)", value=0.2, step=0.01)
            option_type = st.selectbox("Option Type", ["call", "put"])
            greeks = compute_greeks(S, K, T, r, sigma, option_type)
            st.write("Computed Options Greeks:")
            st.write(greeks)
        with tabs[7]:
            st.markdown("#### Export Summary Report")
            report_df = generate_report(portfolio, market_data, annual_return, max_dd, avg_dd, rec_time, sharpe, compute_sortino_ratio(portfolio['returns']), calmar, beta)
            st.dataframe(report_df)
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name='quantbacktest_report.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
