import json

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(code):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.split("\n")]})

add_md("""# 📈 Capstone 2026 — Data-Driven Stock Analysis using Time Series Models
**Organised by:** Consulting & Analytics Club, IIT Guwahati × StockGro  
**Platform:** StockGro (virtual NSE/BSE trading)  
**Capital:** ₹10,00,000 virtual portfolio  

---
## 🗂️ Table of Contents
1. Task 1: Stock Universe Selection
2. Task 2: Data Preprocessing (Missing Values, ADF, Scaling, Split)
3. Task 3: Time Series Forecasting (ARIMA, LSTM, Prophet)
4. Task 4: Volatility & Trend Analysis (Log Returns, GARCH, STL)
5. Task 5: Portfolio Construction & Capital Allocation
6. Task 6: Model Comparison
7. Task 7: StockGro Virtual Trading Execution
8. Task 8: Performance Tracking — Predicted vs Actual""")

add_code("""# Global Imports & Configuration
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for nbconvert
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from arch import arch_model

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
FIGSIZE = (16, 6)
tf.random.set_seed(42)
np.random.seed(42)

# PROJECT CONSTANTS
TOTAL_CAPITAL   = 1_000_000
START_DATE      = '2021-01-01'
END_DATE        = '2025-12-31'
TRAIN_END       = '2025-06-30'
TEST_START      = '2025-07-01'
FORECAST_DAYS   = 2 # Predicting 2 days beyond dataset
SEQ_LEN         = 60
print("All imports successful!")""")

add_md("""---
## 📌 Task 1 — Stock Universe Selection
We select **5 large-cap NSE stocks** spanning **5 distinct sectors**:
- `HDFCBANK.NS` (Banking)
- `TCS.NS` (IT)
- `SUNPHARMA.NS` (Pharma)
- `ITC.NS` (FMCG)
- `TATAMOTORS.NS` (Auto)""")

add_code("""# TASK 1: STOCK UNIVERSE & DATA DOWNLOAD
STOCKS = {
    'HDFCBANK.NS': 'Banking',
    'TCS.NS'     : 'IT',
    'SUNPHARMA.NS': 'Pharma',
    'ITC.NS'     : 'FMCG',
    'TATAMOTORS.NS': 'Auto'
}
TICKERS = list(STOCKS.keys())

print(f'Downloading daily data for {len(TICKERS)} stocks...')
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, interval='1d')

# --- Robust column handling for any yfinance version ---
if isinstance(raw.columns, pd.MultiIndex):
    close = raw['Close'].copy()
else:
    close = raw[['Close']].copy()
    close.columns = TICKERS

# Flatten column names if they are tuples
if isinstance(close.columns[0], tuple):
    close.columns = [c[-1] for c in close.columns]

# Ensure column names are exactly our TICKERS list
# (yfinance may reorder them alphabetically)
close = close[[t for t in TICKERS if t in close.columns]]

# Remove timezone from index if present
if hasattr(close.index, 'tz') and close.index.tz is not None:
    close.index = close.index.tz_localize(None)

print(f'Dataset shape : {close.shape}')
print(f'Columns       : {list(close.columns)}')
print(f'Index range   : {close.index.min()} to {close.index.max()}')
display(close.tail())

# Plot Normalised Closing Prices
fig, ax = plt.subplots(figsize=FIGSIZE)
for ticker in TICKERS:
    if ticker in close.columns:
        norm = close[ticker] / close[ticker].iloc[0] * 100
        ax.plot(norm, label=f'{ticker}')
ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', label='Test Start')
ax.set_title('Task 1 — Normalised Closing Price')
ax.legend()
plt.tight_layout()
plt.show()""")

add_md("""---
## 📌 Task 2 — Data Preprocessing
1. Handle Missing Values (ffill, bfill)
2. Train / Test Split
3. ADF Stationarity Test""")

add_code("""# TASK 2: DATA PREPROCESSING
close = close.ffill().bfill()
print('Missing values after fill:', close.isnull().sum().sum())

train = close.loc[:TRAIN_END].copy()
test  = close.loc[TEST_START:].copy()
print(f'Train: {len(train)} days | Test: {len(test)} days')

# Sanity check — make sure train is not empty
assert len(train) > 0, f"Train set is empty! close index: {close.index.min()} - {close.index.max()}"
assert len(test)  > 0, f"Test set is empty!"

def adf_test(series):
    s = series.dropna()
    if len(s) < 20:
        return False, 1.0   # not enough data, assume non-stationary
    result = adfuller(s)
    return result[1] <= 0.05, result[1] # is_stationary, p-value

adf_res = []
diff_flags = {}
for ticker in TICKERS:
    stat, p = adf_test(train[ticker])
    diff_flags[ticker] = 0 if stat else 1
    adf_res.append({'Ticker': ticker, 'Stationary': stat, 'p-value': round(p, 6), 'd': diff_flags[ticker]})

display(pd.DataFrame(adf_res))""")

add_md("""---
## 📌 Task 4 — Volatility & Trend Analysis (Executed before Task 3 for better flow)""")

add_code("""# TASK 4: VOLATILITY & TREND
log_returns = np.log(close / close.shift(1))
# Drop only the first row (NaN from shift), keep per-column NaNs
log_returns = log_returns.iloc[1:]

vol_20d = log_returns.rolling(20).std() * np.sqrt(252)
# Drop rows where ALL values are NaN (initial rolling window)
vol_20d = vol_20d.dropna(how='all')

print(f"log_returns shape: {log_returns.shape}, vol_20d shape: {vol_20d.shape}")

latest_vol = vol_20d.iloc[-1]

print("\\nLatest Annualised Volatility:")
display(latest_vol)

fig, ax = plt.subplots(figsize=FIGSIZE)
for ticker in TICKERS:
    series = vol_20d[ticker].dropna()
    ax.plot(series, label=ticker)
ax.set_title('20-Day Rolling Annualised Volatility')
ax.legend()
plt.tight_layout()
plt.show()

# STL Decomposition (Example: TATAMOTORS)
tata_series = close['TATAMOTORS.NS'].dropna()
print(f"TATAMOTORS series length for decomposition: {len(tata_series)}")
decomp = seasonal_decompose(tata_series, model='multiplicative', period=252)
fig = decomp.plot()
fig.suptitle('STL Decomposition - TATAMOTORS.NS', y=1.02)
plt.tight_layout()
plt.show()""")

add_md("""---
## 📌 Task 3 — Time Series Forecasting
Forecasting with ARIMA, Prophet, and LSTM.""")

add_code("""# TASK 3: FORECASTING
results_dict = {t: {} for t in TICKERS}
forecasts_2d = {t: {} for t in TICKERS}

for ticker in TICKERS:
    print(f"\\n{'='*50}")
    print(f"--- Modeling {ticker} ---")
    train_s = train[ticker].dropna()
    test_s = test[ticker].dropna()
    print(f"    Train pts: {len(train_s)}, Test pts: {len(test_s)}")
    
    # --- ARIMA ---
    try:
        model_arima = pm.auto_arima(train_s, d=diff_flags[ticker], seasonal=False, suppress_warnings=True)
        arima_preds = model_arima.predict(n_periods=len(test_s))
        arima_fut = model_arima.predict(n_periods=len(test_s) + 2)[-2:].values if hasattr(model_arima.predict(n_periods=len(test_s) + 2), 'values') else model_arima.predict(n_periods=len(test_s) + 2)[-2:]
        print(f"    ARIMA order: {model_arima.order}")
    except Exception as e:
        print(f"    ARIMA failed: {e}")
        arima_preds = np.full(len(test_s), train_s.iloc[-1])
        arima_fut = np.array([train_s.iloc[-1], train_s.iloc[-1]])
    
    # --- PROPHET ---
    try:
        df_prophet = train_s.reset_index()
        # Rename columns robustly — first col is date, second is value
        df_prophet.columns = ['ds', 'y']
        # Remove timezone if present
        if hasattr(df_prophet['ds'].dtype, 'tz') or str(df_prophet['ds'].dtype).startswith('datetime64[ns,'):
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=len(test_s) + 2, freq='B')  # business days
        fcst = m.predict(future)
        prophet_all = fcst['yhat'].values
        prophet_preds = prophet_all[len(train_s):len(train_s)+len(test_s)]
        prophet_fut = prophet_all[-2:]
        
        # If prophet_preds is shorter than test, pad with last value
        if len(prophet_preds) < len(test_s):
            prophet_preds = np.pad(prophet_preds, (0, len(test_s) - len(prophet_preds)), 
                                   mode='edge')
        print(f"    Prophet fitted OK")
    except Exception as e:
        print(f"    Prophet failed: {e}")
        prophet_preds = np.full(len(test_s), train_s.iloc[-1])
        prophet_fut = np.array([train_s.iloc[-1], train_s.iloc[-1]])
    
    # --- LSTM ---
    try:
        sc = MinMaxScaler()
        train_sc = sc.fit_transform(train_s.values.reshape(-1, 1))
        
        X_tr, y_tr = [], []
        for i in range(SEQ_LEN, len(train_sc)):
            X_tr.append(train_sc[i-SEQ_LEN:i, 0])
            y_tr.append(train_sc[i, 0])
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))
        
        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mse')
        lstm.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)
        
        # Build test input sequence
        full_data = close[ticker].values
        inputs = full_data[len(full_data) - len(test_s) - SEQ_LEN:]
        inputs = sc.transform(inputs.reshape(-1, 1))
        X_te = []
        for i in range(SEQ_LEN, len(inputs)):
            X_te.append(inputs[i-SEQ_LEN:i, 0])
        X_te = np.array(X_te)
        X_te = np.reshape(X_te, (X_te.shape[0], X_te.shape[1], 1))
        
        lstm_preds = sc.inverse_transform(lstm.predict(X_te, verbose=0)).flatten()
        
        # Forecast future 2 days
        last_seq = inputs[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        f1 = lstm.predict(last_seq, verbose=0)
        last_seq_2 = np.append(last_seq[:, 1:, :], f1.reshape(1, 1, 1), axis=1)
        f2 = lstm.predict(last_seq_2, verbose=0)
        lstm_fut = sc.inverse_transform(np.vstack([f1[0], f2[0]])).flatten()
        print(f"    LSTM fitted OK")
    except Exception as e:
        print(f"    LSTM failed: {e}")
        lstm_preds = np.full(len(test_s), train_s.iloc[-1])
        lstm_fut = np.array([train_s.iloc[-1], train_s.iloc[-1]])
    
    # Ensure all prediction arrays match test length
    min_len = min(len(arima_preds), len(prophet_preds), len(lstm_preds), len(test_s))
    
    results_dict[ticker] = {
        'ARIMA': np.array(arima_preds[:min_len]),
        'Prophet': np.array(prophet_preds[:min_len]),
        'LSTM': np.array(lstm_preds[:min_len]),
        'Actual': test_s.values[:min_len]
    }
    
    forecasts_2d[ticker] = {
        'ARIMA': np.array(arima_fut).flatten()[:2],
        'Prophet': np.array(prophet_fut).flatten()[:2],
        'LSTM': np.array(lstm_fut).flatten()[:2]
    }

print("\\n✅ All models trained successfully!")""")

add_md("""---
## 📌 Task 6 — Model Comparison""")

add_code("""# TASK 6: MODEL COMPARISON
metrics = []
for ticker in TICKERS:
    act = results_dict[ticker]['Actual']
    for model_name in ['ARIMA', 'Prophet', 'LSTM']:
        preds = results_dict[ticker][model_name]
        n = min(len(act), len(preds))
        a, p_ = act[:n], preds[:n]
        
        mape = mean_absolute_percentage_error(a, p_)
        rmse = np.sqrt(mean_squared_error(a, p_))
        dir_acc = np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p_))) if n > 1 else 0
        
        metrics.append({'Ticker': ticker, 'Model': model_name, 
                        'MAPE': round(mape, 4), 'RMSE': round(rmse, 2), 
                        'DirAcc': round(dir_acc, 4)})

df_metrics = pd.DataFrame(metrics)
print("\\n📊 Per-Stock Metrics:")
display(df_metrics)

print("\\n📊 Average Metrics by Model:")
display(df_metrics.groupby('Model')[['MAPE', 'RMSE', 'DirAcc']].mean().sort_values('MAPE'))

# Visualisation — RMSE comparison
fig, ax = plt.subplots(figsize=(12, 5))
df_metrics.pivot(index='Ticker', columns='Model', values='RMSE').plot(kind='bar', ax=ax)
ax.set_title('RMSE Comparison across Models')
ax.set_ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""")

add_md("""---
## 📌 Task 5 — Portfolio Construction
Using **Strategy B (Volatility-Aware Sizing)**: w_i = (1/vol_i) / sum(1/vol_j)""")

add_code("""# TASK 5: PORTFOLIO CONSTRUCTION
inv_vol = 1 / latest_vol
weights = inv_vol / inv_vol.sum()

alloc = pd.DataFrame({
    'Sector': [STOCKS[t] for t in weights.index],
    'Volatility': latest_vol,
    'Weight': weights,
    'Amount (₹)': (weights * TOTAL_CAPITAL).round(0)
})
print("\\n💰 Portfolio Allocation (₹10,00,000 Capital):")
display(alloc.sort_values('Amount (₹)', ascending=False))

# Pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(alloc['Amount (₹)'], labels=alloc.index, autopct='%1.1f%%', startangle=140)
ax.set_title('Portfolio Allocation — Volatility-Aware')
plt.tight_layout()
plt.show()""")

add_md("""---
## 📌 Task 7 & 8 — Virtual Trading Execution & Performance Tracking
1. Log into StockGro.
2. Execute the allocations calculated in Task 5.
3. Compare the predictions from `forecasts_2d` with the actual market closing prices on Day 1 and Day 2.""")

add_code("""# TASK 7 & 8: PREDICTIONS FOR VIRTUAL TRADING
print("\\n🔮 Next 2 Days Predictions (For StockGro Trading):")
pred_df = pd.DataFrame({t: forecasts_2d[t]['LSTM'] for t in TICKERS}, index=['Day 1', 'Day 2'])
display(pred_df.round(2))

# Summary of recommended trades
print("\\n📋 Recommended Trade Summary:")
for ticker in TICKERS:
    last_price = close[ticker].iloc[-1]
    day1_pred = forecasts_2d[ticker]['LSTM'][0]
    pct_change = ((day1_pred - last_price) / last_price) * 100
    action = "BUY 🟢" if pct_change > 0 else "HOLD/SELL 🔴"
    allocated = alloc.loc[ticker, 'Amount (₹)']
    shares = int(allocated / last_price)
    print(f"  {ticker:18s} | Last: ₹{last_price:>10,.2f} | Pred: ₹{day1_pred:>10,.2f} | "
          f"Chg: {pct_change:+.2f}% | {action} | Shares: {shares}")

print("\\n✅ Capstone 2026 Analysis Complete!")""")

with open(r"c:\Users\Lenovo\OneDrive\Desktop\capstone-project\Capstone_2026_Time_Series_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
