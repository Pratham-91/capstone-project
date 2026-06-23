"""
Fix the Capstone notebook cells directly in the .ipynb JSON.
Addresses: yfinance column mapping issues, empty series guards, and robust decomposition.
"""
import json

NB_PATH = r"c:\Users\Lenovo\OneDrive\Desktop\capstone-project\Capstone_2026_Time_Series_Analysis.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def get_cell_source(cell):
    return "".join(cell["source"])

def set_cell_source(cell, code):
    cell["source"] = [line + "\n" for line in code.split("\n")]
    cell["outputs"] = []
    cell["execution_count"] = None

# ── Fix Cell: TASK 1 — Data Download ──
TASK1_CODE = r"""# TASK 1: STOCK UNIVERSE & DATA DOWNLOAD
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

# Debug: show raw column structure
print(f"Raw columns type: {type(raw.columns)}")
if isinstance(raw.columns, pd.MultiIndex):
    print(f"Raw MultiIndex levels: {raw.columns.levels}")
    print(f"Raw MultiIndex names : {raw.columns.names}")

# --- Robust column extraction for any yfinance version ---
if isinstance(raw.columns, pd.MultiIndex):
    # MultiIndex: ('Close', 'HDFCBANK.NS'), etc.
    close = raw['Close'].copy()
else:
    # Single ticker fallback
    close = raw[['Close']].copy()
    close.columns = TICKERS

# Flatten tuple column names if needed
if len(close.columns) > 0 and isinstance(close.columns[0], tuple):
    close.columns = [c[-1] if isinstance(c, tuple) else c for c in close.columns]

# Remove timezone from index
if hasattr(close.index, 'tz') and close.index.tz is not None:
    close.index = close.index.tz_localize(None)

# Reorder to match our TICKERS list
available = [t for t in TICKERS if t in close.columns]
missing = [t for t in TICKERS if t not in close.columns]
if missing:
    print(f"WARNING: Missing tickers in download: {missing}")
close = close[available]

# Check for all-NaN columns and drop them
nan_cols = close.columns[close.isna().all()]
if len(nan_cols) > 0:
    print(f"WARNING: All-NaN columns found and dropped: {list(nan_cols)}")
    close = close.drop(columns=nan_cols)
    # Update TICKERS to only include available ones
    TICKERS = [t for t in TICKERS if t in close.columns]

# Forward/back fill BEFORE any analysis
close = close.ffill().bfill()

print(f'\nDataset shape : {close.shape}')
print(f'Columns       : {list(close.columns)}')
print(f'Index range   : {close.index.min()} to {close.index.max()}')
print(f'NaN per col   : \n{close.isna().sum()}')
display(close.tail())

# Plot Normalised Closing Prices
fig, ax = plt.subplots(figsize=FIGSIZE)
for ticker in TICKERS:
    norm = close[ticker] / close[ticker].iloc[0] * 100
    ax.plot(norm, label=f'{ticker}')
ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', label='Test Start')
ax.set_title('Task 1 — Normalised Closing Price')
ax.legend()
plt.tight_layout()
plt.show()
"""

# ── Fix Cell: TASK 2 — Preprocessing ──
TASK2_CODE = r"""# TASK 2: DATA PREPROCESSING
print('Missing values after fill:', close.isnull().sum().sum())

train = close.loc[:TRAIN_END].copy()
test  = close.loc[TEST_START:].copy()
print(f'Train: {len(train)} days | Test: {len(test)} days')

# Sanity check
assert len(train) > 0, f"Train set is empty! Index range: {close.index.min()} to {close.index.max()}"
assert len(test)  > 0, f"Test set is empty!"

def adf_test(series):
    s = series.dropna()
    if len(s) < 20:
        return False, 1.0
    result = adfuller(s)
    return result[1] <= 0.05, result[1]

adf_res = []
diff_flags = {}
for ticker in TICKERS:
    stat, p = adf_test(train[ticker])
    diff_flags[ticker] = 0 if stat else 1
    adf_res.append({'Ticker': ticker, 'Stationary': stat, 'p-value': round(p, 6), 'd': diff_flags[ticker]})

display(pd.DataFrame(adf_res))
"""

# ── Fix Cell: TASK 4 — Volatility ──
TASK4_CODE = r"""# TASK 4: VOLATILITY & TREND
log_returns = np.log(close / close.shift(1)).iloc[1:]  # drop first NaN row

vol_20d = log_returns.rolling(20).std() * np.sqrt(252)
vol_20d = vol_20d.dropna(how='all')

print(f"log_returns shape: {log_returns.shape}, vol_20d shape: {vol_20d.shape}")

latest_vol = vol_20d.iloc[-1]
print("\nLatest Annualised Volatility:")
display(latest_vol)

fig, ax = plt.subplots(figsize=FIGSIZE)
for ticker in TICKERS:
    series = vol_20d[ticker].dropna()
    if len(series) > 0:
        ax.plot(series, label=ticker)
ax.set_title('20-Day Rolling Annualised Volatility')
ax.legend()
plt.tight_layout()
plt.show()

# STL Decomposition — pick the ticker with the most data points
best_ticker = max(TICKERS, key=lambda t: close[t].dropna().shape[0])
decomp_series = close[best_ticker].dropna()
print(f"\nSTL Decomposition for {best_ticker} ({len(decomp_series)} observations)")

if len(decomp_series) >= 504:
    decomp = seasonal_decompose(decomp_series, model='multiplicative', period=252)
    fig = decomp.plot()
    fig.suptitle(f'STL Decomposition - {best_ticker}', y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print(f"  Skipped: need >= 504 obs, have {len(decomp_series)}")
"""

# ── Fix Cell: TASK 3 — Forecasting ──
TASK3_CODE = r"""# TASK 3: FORECASTING
results_dict = {t: {} for t in TICKERS}
forecasts_2d = {t: {} for t in TICKERS}

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"--- Modeling {ticker} ---")
    train_s = train[ticker].dropna()
    test_s = test[ticker].dropna()
    print(f"    Train pts: {len(train_s)}, Test pts: {len(test_s)}")
    
    if len(train_s) < SEQ_LEN + 10 or len(test_s) < 2:
        print(f"    SKIPPED — insufficient data")
        results_dict[ticker] = {'ARIMA': np.array([]), 'Prophet': np.array([]),
                                'LSTM': np.array([]), 'Actual': test_s.values}
        forecasts_2d[ticker] = {'ARIMA': np.array([0,0]), 'Prophet': np.array([0,0]),
                                'LSTM': np.array([0,0])}
        continue
    
    # --- ARIMA ---
    try:
        model_arima = pm.auto_arima(train_s, d=diff_flags[ticker], seasonal=False, suppress_warnings=True)
        arima_preds = model_arima.predict(n_periods=len(test_s))
        arima_all = model_arima.predict(n_periods=len(test_s) + 2)
        arima_fut = np.array(arima_all)[-2:]
        print(f"    ARIMA order: {model_arima.order}")
    except Exception as e:
        print(f"    ARIMA failed: {e}")
        arima_preds = np.full(len(test_s), train_s.iloc[-1])
        arima_fut = np.array([train_s.iloc[-1], train_s.iloc[-1]])
    
    # --- PROPHET ---
    try:
        df_prophet = train_s.reset_index()
        df_prophet.columns = ['ds', 'y']
        if hasattr(df_prophet['ds'].dtype, 'tz'):
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=len(test_s) + 2, freq='B')
        fcst = m.predict(future)
        prophet_all = fcst['yhat'].values
        prophet_preds = prophet_all[len(train_s):len(train_s)+len(test_s)]
        prophet_fut = prophet_all[-2:]
        
        if len(prophet_preds) < len(test_s):
            prophet_preds = np.pad(prophet_preds, (0, len(test_s) - len(prophet_preds)), mode='edge')
        print(f"    Prophet OK")
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
        
        last_seq = inputs[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        f1 = lstm.predict(last_seq, verbose=0)
        last_seq_2 = np.append(last_seq[:, 1:, :], f1.reshape(1, 1, 1), axis=1)
        f2 = lstm.predict(last_seq_2, verbose=0)
        lstm_fut = sc.inverse_transform(np.vstack([f1[0], f2[0]])).flatten()
        print(f"    LSTM OK")
    except Exception as e:
        print(f"    LSTM failed: {e}")
        lstm_preds = np.full(len(test_s), train_s.iloc[-1])
        lstm_fut = np.array([train_s.iloc[-1], train_s.iloc[-1]])
    
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

print("\n✅ All models trained successfully!")
"""

# ── Fix Cell: TASK 6 — Model Comparison ──
TASK6_CODE = r"""# TASK 6: MODEL COMPARISON
metrics = []
for ticker in TICKERS:
    act = results_dict[ticker]['Actual']
    for model_name in ['ARIMA', 'Prophet', 'LSTM']:
        preds = results_dict[ticker][model_name]
        n = min(len(act), len(preds))
        if n < 2:
            continue
        a, p_ = act[:n], preds[:n]
        
        mape = mean_absolute_percentage_error(a, p_)
        rmse = np.sqrt(mean_squared_error(a, p_))
        dir_acc = np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p_)))
        
        metrics.append({'Ticker': ticker, 'Model': model_name,
                        'MAPE': round(mape, 4), 'RMSE': round(rmse, 2),
                        'DirAcc': round(dir_acc, 4)})

df_metrics = pd.DataFrame(metrics)
print("\n📊 Per-Stock Metrics:")
display(df_metrics)

print("\n📊 Average Metrics by Model:")
display(df_metrics.groupby('Model')[['MAPE', 'RMSE', 'DirAcc']].mean().sort_values('MAPE'))

fig, ax = plt.subplots(figsize=(12, 5))
df_metrics.pivot(index='Ticker', columns='Model', values='RMSE').plot(kind='bar', ax=ax)
ax.set_title('RMSE Comparison across Models')
ax.set_ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

# ── Fix Cell: TASK 5 — Portfolio ──
TASK5_CODE = r"""# TASK 5: PORTFOLIO CONSTRUCTION
inv_vol = 1 / latest_vol
weights = inv_vol / inv_vol.sum()

alloc = pd.DataFrame({
    'Sector': [STOCKS[t] for t in weights.index if t in STOCKS],
    'Volatility': latest_vol,
    'Weight': weights,
    'Amount (₹)': (weights * TOTAL_CAPITAL).round(0)
})
print("\n💰 Portfolio Allocation (₹10,00,000 Capital):")
display(alloc.sort_values('Amount (₹)', ascending=False))

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(alloc['Amount (₹)'], labels=alloc.index, autopct='%1.1f%%', startangle=140)
ax.set_title('Portfolio Allocation — Volatility-Aware')
plt.tight_layout()
plt.show()
"""

# ── Fix Cell: TASK 7 & 8 ──
TASK78_CODE = r"""# TASK 7 & 8: PREDICTIONS FOR VIRTUAL TRADING
print("\n🔮 Next 2 Days Predictions (For StockGro Trading):")
pred_df = pd.DataFrame({t: forecasts_2d[t]['LSTM'] for t in TICKERS}, index=['Day 1', 'Day 2'])
display(pred_df.round(2))

print("\n📋 Recommended Trade Summary:")
for ticker in TICKERS:
    last_price = close[ticker].iloc[-1]
    day1_pred = forecasts_2d[ticker]['LSTM'][0]
    pct_change = ((day1_pred - last_price) / last_price) * 100
    action = "BUY 🟢" if pct_change > 0 else "HOLD/SELL 🔴"
    allocated = alloc.loc[ticker, 'Amount (₹)']
    shares = int(allocated / last_price)
    print(f"  {ticker:18s} | Last: ₹{last_price:>10,.2f} | Pred: ₹{day1_pred:>10,.2f} | "
          f"Chg: {pct_change:+.2f}% | {action} | Shares: {shares}")

print("\n✅ Capstone 2026 Analysis Complete!")
"""

# ── Apply fixes to cells ──
CELL_MAP = {
    "# TASK 1:": TASK1_CODE,
    "# TASK 2:": TASK2_CODE,
    "# TASK 4:": TASK4_CODE,
    "# TASK 3:": TASK3_CODE,
    "# TASK 6:": TASK6_CODE,
    "# TASK 5:": TASK5_CODE,
    "# TASK 7": TASK78_CODE,
}

fixed = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = get_cell_source(cell)
    for marker, new_code in CELL_MAP.items():
        if marker in src:
            set_cell_source(cell, new_code.strip())
            fixed += 1
            print(f"  Fixed cell: {marker}")
            break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\n[OK] Fixed {fixed} cells in notebook. Ready to execute.")
