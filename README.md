# Capstone 2026: Time Series Analysis & Stock Price Prediction 📈

## 🚀 Project Mission
The goal of this Capstone project is to build a robust, quantitative framework for forecasting stock prices and constructing an optimized portfolio. By leveraging historical daily stock data and state-of-the-art time series forecasting techniques—ranging from traditional statistical models to advanced deep learning architectures—we aim to predict short-term price movements and create a volatility-weighted portfolio designed to maximize stability and returns.

## 🏆 Key Achievements
- **End-to-End Automated Pipeline**: Successfully developed a pipeline that downloads data, cleans missing values, checks for stationarity, and trains multiple predictive models seamlessly.
- **Deep Learning Superiority**: Demonstrated that Long Short-Term Memory (LSTM) neural networks significantly outperform traditional statistical models (ARIMA) and additive models (Facebook Prophet) in capturing complex, non-linear financial time series patterns.
- **Volatility-Aware Allocation**: Implemented a dynamic risk management strategy that allocates a hypothetical capital base of ₹10,00,000 inversely proportional to each asset's volatility.
- **Actionable Trading Signals**: Generated concrete 2-day forward predictions mapped to clear BUY/HOLD/SELL signals for virtual trading execution.

## 📊 Methodology & Stock Universe

The analysis utilizes five years of daily historical data (January 2021 to December 2025). Following rigorous data validation, the final portfolio consists of four highly robust sectoral leaders in the Indian equity market:

1. **HDFC Bank (`HDFCBANK.NS`)** — Banking
2. **Tata Consultancy Services (`TCS.NS`)** — Information Technology
3. **Sun Pharmaceuticals (`SUNPHARMA.NS`)** — Pharmaceuticals
4. **ITC Limited (`ITC.NS`)** — Fast Moving Consumer Goods (FMCG)

*Note: TATAMOTORS.NS was initially considered but excluded due to data unavailability or delisting.*

### Data Preprocessing
- **Dataset**: 1,235 daily observations per stock.
- **Cleaning**: Missing values handled via forward and backward filling (0 missing values post-cleaning).
- **Stationarity**: Augmented Dickey-Fuller (ADF) tests were used, applying first-order differencing ($d=1$) where necessary.

## 📈 Results & Findings

### 1. Volatility Profile
We computed the 20-day rolling annualized volatility to assess the risk of each equity.
- **ITC (8.49%)**: Most stable, Low Risk
- **HDFC Bank (10.43%)**: Moderate Risk
- **TCS (13.84%)**: Moderate to High Risk
- **Sun Pharma (15.05%)**: Highest fluctuation, Aggressive Risk

### 2. Forecasting Model Evaluation
Three modeling paradigms were trained and evaluated on 125 days of out-of-sample test data:
1. **ARIMA** (Statistical / Autoregressive)
2. **Prophet** (Additive Trend & Seasonality)
3. **LSTM** (Deep Learning / Sequential)

**Average Performance Metrics:**

| Model | Mean Absolute Percentage Error (MAPE) | Root Mean Squared Error (RMSE) | Directional Accuracy |
|-------|---------------------------------------|--------------------------------|----------------------|
| **LSTM** | **1.42%** | **30.43** | **47.78%** |
| ARIMA | 5.25% | 113.74 | 37.90% |
| Prophet | 5.59% | 110.51 | 52.62% |

*Conclusion:* The LSTM network vastly outperformed traditional models, reducing the MAPE to an impressive 1.42%.

## 💼 Portfolio Allocation
To mitigate risk, a Volatility-Aware Allocation strategy was employed for a hypothetical capital base of **₹10,00,000**. Capital was allocated inversely proportional to each asset's volatility:

| Ticker | Weight | Allocation Amount |
|--------|--------|-------------------|
| **ITC.NS** | 33.43% | ₹3,34,325 |
| **HDFCBANK.NS** | 27.21% | ₹2,72,116 |
| **TCS.NS** | 20.49% | ₹2,04,978 |
| **SUNPHARMA.NS**| 18.85% | ₹1,88,581 |

## 🚦 Actionable Trading Signals
Based on the superior LSTM model, the following trading strategy is recommended for execution (Next 2 Days):

| Ticker | Last Close Price | Predicted (Day 1) | Expected Change | Action | Target Shares |
|--------|------------------|-------------------|-----------------|--------|---------------|
| **HDFCBANK.NS** | ₹990.90 | ₹989.04 | -0.19% | **HOLD / SELL 🔴** | 274 |
| **TCS.NS** | ₹3,188.83 | ₹3,232.95 | +1.38% | **BUY 🟢** | 64 |
| **SUNPHARMA.NS**| ₹1,709.10 | ₹1,740.09 | +1.81% | **BUY 🟢** | 110 |
| **ITC.NS** | ₹392.38 | ₹397.25 | +1.24% | **BUY 🟢** | 852 |
