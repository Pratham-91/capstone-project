import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
TOTAL_CAPITAL = 1_000_000
START_DATE = '2021-01-01'
END_DATE = '2025-12-31'
TRAIN_END = '2025-06-30'
TEST_START = '2025-07-01'
SEQ_LEN = 60
TICKERS = ['HDFCBANK.NS', 'TCS.NS', 'SUNPHARMA.NS', 'ITC.NS']
SECTORS = {
    'HDFCBANK.NS': 'Banking',
    'TCS.NS': 'IT',
    'SUNPHARMA.NS': 'Pharma',
    'ITC.NS': 'FMCG'
}

print("Fetching historical data...")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, interval='1d')

if isinstance(raw.columns, pd.MultiIndex):
    close = raw['Close'].copy()
else:
    close = raw[['Close']].copy()
    close.columns = TICKERS

if len(close.columns) > 0 and isinstance(close.columns[0], tuple):
    close.columns = [c[-1] if isinstance(c, tuple) else c for c in close.columns]

if hasattr(close.index, 'tz') and close.index.tz is not None:
    close.index = close.index.tz_localize(None)

close = close[TICKERS]
close = close.ffill().bfill()

# Calculate log returns, volatility, correlation
print("Calculating volatility and correlation...")
log_returns = np.log(close / close.shift(1)).iloc[1:]
vol_20d = log_returns.rolling(20).std() * np.sqrt(252)
vol_20d = vol_20d.dropna(how='all')

latest_vol = vol_20d.iloc[-1]
inv_vol = 1 / latest_vol
weights = inv_vol / inv_vol.sum()
alloc_amount = weights * TOTAL_CAPITAL

correlation_matrix = log_returns.corr()

# Train test split
train = close.loc[:TRAIN_END].copy()
test = close.loc[TEST_START:].copy()

forecasts = {}

print("Training LSTM models...")
for ticker in TICKERS:
    print(f"Training for {ticker}...")
    train_s = train[ticker].dropna()
    test_s = test[ticker].dropna()
    
    sc = MinMaxScaler()
    train_sc = sc.fit_transform(train_s.values.reshape(-1, 1))
    
    X_tr, y_tr = [], []
    for i in range(SEQ_LEN, len(train_sc)):
        X_tr.append(train_sc[i-SEQ_LEN:i, 0])
        y_tr.append(train_sc[i, 0])
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1))
    
    lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)
    
    full_data = close[ticker].values
    inputs = full_data[len(full_data) - len(test_s) - SEQ_LEN:]
    inputs = sc.transform(inputs.reshape(-1, 1))
    X_te = []
    for i in range(SEQ_LEN, len(inputs)):
        X_te.append(inputs[i-SEQ_LEN:i, 0])
    X_te = np.array(X_te).reshape((-1, SEQ_LEN, 1))
    
    lstm_preds = sc.inverse_transform(lstm.predict(X_te, verbose=0)).flatten()
    
    forecasts[ticker] = {
        'actual': [float(x) for x in test_s.values],
        'predicted': [float(x) for x in lstm_preds],
    }

print("Preparing dashboard data...")
# Convert index to string dates
dates_all = [d.strftime('%Y-%m-%d') for d in close.index]
test_dates = [d.strftime('%Y-%m-%d') for d in test.index]
vol_dates = [d.strftime('%Y-%m-%d') for d in vol_20d.index]

data = {
    'tickers': TICKERS,
    'sectors': [SECTORS[t] for t in TICKERS],
    'dates': dates_all,
    'test_dates': test_dates,
    'vol_dates': vol_dates,
    'history': {t: list(close[t].values) for t in TICKERS},
    'volatility_history': {t: list(vol_20d[t].values) for t in TICKERS},
    'latest_volatility': {t: float(latest_vol[t]) for t in TICKERS},
    'allocation': {t: float(alloc_amount[t]) for t in TICKERS},
    'weights': {t: float(weights[t]) for t in TICKERS},
    'correlation': correlation_matrix.values.tolist(),
    'forecasts': forecasts
}

with open(os.path.join('dashboard', 'data.json'), 'w') as f:
    json.dump(data, f)

print("Dashboard data successfully saved to dashboard/data.json!")
