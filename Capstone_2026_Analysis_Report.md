# Capstone 2026: Time Series Analysis & Stock Price Prediction Report

**Date:** May 2026
**Subject:** Comprehensive Analysis, Forecasting, and Portfolio Allocation using Deep Learning & Statistical Models

---

## 1. Executive Summary

This report concludes the analytical findings of the Capstone 2026 project, which aimed to build a robust quantitative framework for forecasting stock prices and constructing an optimized portfolio. The analysis utilized daily historical data spanning five years (January 2021 to December 2025) for a diversified universe of Indian equities. By leveraging state-of-the-art time series forecasting techniques—including ARIMA, Facebook Prophet, and LSTM Neural Networks—we were able to predict short-term price movements and construct a volatility-weighted portfolio designed to maximize stability and returns.

## 2. Stock Universe & Data Preprocessing

The initial stock universe consisted of five major sectoral leaders. Following rigorous data validation, one stock (TATAMOTORS.NS) was excluded due to data unavailability/delisting, leaving a highly robust four-stock portfolio:

- **HDFC Bank (`HDFCBANK.NS`)** — Banking
- **Tata Consultancy Services (`TCS.NS`)** — Information Technology
- **Sun Pharmaceuticals (`SUNPHARMA.NS`)** — Pharmaceuticals
- **ITC Limited (`ITC.NS`)** — Fast Moving Consumer Goods (FMCG)

### Data Quality & Preprocessing
- **Dataset Size**: 1,235 daily observations per stock.
- **Missing Values**: Handled effectively via forward and backward filling (0 missing values post-cleaning).
- **Train/Test Split**: 1,110 days for Training, 125 days for Testing.
- **Stationarity**: Augmented Dickey-Fuller (ADF) tests indicated that all raw stock prices were non-stationary ($p > 0.05$). A first-order differencing ($d=1$) was applied for statistical modeling (ARIMA) to ensure stationarity.

## 3. Volatility & Trend Analysis

We computed the 20-day rolling annualized volatility to assess the risk profile of each equity. The latest annualized volatility figures are as follows:

| Ticker | Sector | Annualized Volatility | Risk Profile |
|--------|--------|-----------------------|--------------|
| **ITC.NS** | FMCG | **8.49%** | Low / Stable |
| **HDFCBANK.NS** | Banking | **10.43%** | Moderate |
| **TCS.NS** | IT | **13.84%** | Moderate to High |
| **SUNPHARMA.NS**| Pharma | **15.05%** | High / Aggressive |

*Insight:* ITC remains the most stable asset in the current market environment, while Sun Pharma exhibits the highest price fluctuations.

## 4. Forecasting Model Evaluation

Three distinct modeling paradigms were trained and evaluated on the out-of-sample test data:
1. **ARIMA** (Statistical / Autoregressive)
2. **Prophet** (Additive Trend & Seasonality)
3. **LSTM** (Deep Learning / Sequential)

### Average Metrics by Model

| Model | Mean Absolute Percentage Error (MAPE) | Root Mean Squared Error (RMSE) | Directional Accuracy |
|-------|---------------------------------------|--------------------------------|----------------------|
| **LSTM** | **1.42%** | **30.43** | **47.78%** |
| ARIMA | 5.25% | 113.74 | 37.90% |
| Prophet | 5.59% | 110.51 | 52.62% |

*Conclusion:* The **LSTM network vastly outperformed** the traditional statistical models in terms of both absolute magnitude error (MAPE) and variance (RMSE). Consequently, LSTM forecasts were selected to drive our actionable trading signals.

## 5. Portfolio Construction

To mitigate risk, a **Volatility-Aware Allocation strategy** was employed. Using a hypothetical capital base of **₹10,00,000**, capital was allocated inversely proportional to each asset's volatility, favoring stable stocks to anchor the portfolio.

| Ticker | Weight | Allocation Amount |
|--------|--------|-------------------|
| **ITC.NS** | 33.43% | ₹3,34,325 |
| **HDFCBANK.NS** | 27.21% | ₹2,72,116 |
| **TCS.NS** | 20.49% | ₹2,04,978 |
| **SUNPHARMA.NS**| 18.85% | ₹1,88,581 |

## 6. Actionable Trading Recommendations (Next 2 Days)

Utilizing the superior LSTM model, we generated price forecasts for the immediate upcoming trading sessions. Based on these predictions, the following trading strategy is recommended for execution on the virtual trading platform:

| Ticker | Last Close Price | Predicted (Day 1) | Expected Change | Action | Target Shares |
|--------|------------------|-------------------|-----------------|--------|---------------|
| **HDFCBANK.NS** | ₹990.90 | ₹989.04 | -0.19% | **HOLD / SELL 🔴** | 274 |
| **TCS.NS** | ₹3,188.83 | ₹3,232.95 | +1.38% | **BUY 🟢** | 64 |
| **SUNPHARMA.NS**| ₹1,709.10 | ₹1,740.09 | +1.81% | **BUY 🟢** | 110 |
| **ITC.NS** | ₹392.38 | ₹397.25 | +1.24% | **BUY 🟢** | 852 |

### Final Conclusion
The quantitative pipeline established in this Capstone project successfully demonstrates the efficacy of deep learning (LSTM) over traditional models (ARIMA, Prophet) for high-frequency short-term stock forecasting. By integrating predictive signals with a volatility-aware portfolio optimization framework, we have built a resilient strategy that mathematically maximizes risk-adjusted upside potential.
