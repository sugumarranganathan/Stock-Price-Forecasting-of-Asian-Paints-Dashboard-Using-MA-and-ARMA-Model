import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Asian Paints Stock Forecasting Dashboard",
    layout="wide"
)

st.title("📈 Asian Paints Stock Price Forecasting Dashboard")
st.markdown("### Stock Price Forecasting Using AutoRegressive (AR) Model")

# ==================================================
# LOAD DATA
# ==================================================

try:
    df = pd.read_csv("ASIANPAINT.NS.csv")
except:
    st.error("Dataset file not found.")
    st.stop()

# ==================================================
# DATA CLEANING
# ==================================================

df.columns = df.columns.str.lower()

if "unnamed: 0" in df.columns:
    df.rename(columns={"unnamed: 0": "date"}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

series = df["close"]

# ==================================================
# DATASET PREVIEW
# ==================================================

st.header("📊 Dataset Preview")
st.dataframe(df.head())

# ==================================================
# DATASET STATISTICS
# ==================================================

st.header("📋 Dataset Statistics")
st.dataframe(df.describe())

# ==================================================
# LATEST CLOSE PRICE
# ==================================================

st.header("💰 Latest Close Price")

st.metric(
    "Current Close Price",
    f"₹ {series.iloc[-1]:.2f}"
)

# ==================================================
# STOCK PRICE TREND
# ==================================================

st.header("📈 Stock Price Trend")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(series.index, series.values)

ax.set_title("Asian Paints Closing Price Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.grid(True)

st.pyplot(fig)

# ==================================================
# ROLLING MEAN & STD
# ==================================================

st.header("📉 Rolling Mean & Standard Deviation")

rolling_mean = series.rolling(12).mean()
rolling_std = series.rolling(12).std()

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(series,label="Original")
ax.plot(rolling_mean,label="Rolling Mean")
ax.plot(rolling_std,label="Rolling Std")

ax.legend()
ax.grid(True)

st.pyplot(fig)

# ==================================================
# ADF TEST
# ==================================================

st.header("🧪 ADF Stationarity Test")

result = adfuller(series.dropna())

st.write("ADF Statistic :", round(result[0],4))
st.write("P Value :", round(result[1],6))

if result[1] < 0.05:
    st.success("Series is Stationary")
else:
    st.error("Series is Non-Stationary")

# ==================================================
# DIFFERENCING
# ==================================================

series_diff = series.diff().dropna()

# ==================================================
# ACF
# ==================================================

st.header("📊 ACF Plot")

fig, ax = plt.subplots(figsize=(10,4))
plot_acf(series_diff,lags=30,ax=ax)

st.pyplot(fig)

# ==================================================
# PACF
# ==================================================

st.header("📊 PACF Plot")

fig, ax = plt.subplots(figsize=(10,4))
plot_pacf(series_diff,lags=30,ax=ax)

st.pyplot(fig)

# ==================================================
# NORMALIZATION
# ==================================================

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(
    series.values.reshape(-1,1)
)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

train_size = int(len(data_scaled)*0.80)

train = data_scaled[:train_size]
test = data_scaled[train_size:]

# ==================================================
# AR MODEL COMPARISON
# ==================================================

st.header("📊 AR Model Comparison")

results = []

for lag in range(1,5):

    model = AutoReg(
        train.flatten(),
        lags=lag,
        trend="c"
    )

    fit = model.fit()

    forecast = fit.predict(
        start=len(train),
        end=len(train)+len(test)-1
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast.reshape(-1,1)
        )
    )

    results.append([
        lag,
        rmse
    ])

result_df = pd.DataFrame(
    results,
    columns=["Lag","RMSE"]
)

result_df = result_df.sort_values(
    "RMSE"
)

st.dataframe(
    result_df,
    use_container_width=True
)

best_lag = int(
    result_df.iloc[0]["Lag"]
)

st.success(
    f"Best AR Lag = {best_lag}"
)

# ==================================================
# FINAL MODEL
# ==================================================

final_model = AutoReg(
    data_scaled.flatten(),
    lags=best_lag,
    trend="c"
)

final_fit = final_model.fit()

# ==================================================
# NEXT 4 DAY FORECAST
# ==================================================

forecast_scaled = final_fit.predict(
    start=len(data_scaled),
    end=len(data_scaled)+3
)

forecast_price = scaler.inverse_transform(
    forecast_scaled.reshape(-1,1)
)

forecast_df = pd.DataFrame({

    "Day":[
        "Day 1",
        "Day 2",
        "Day 3",
        "Day 4"
    ],

    "Forecast Price (₹)":
    forecast_price.flatten().round(2)

})

st.header("🔮 Next 4-Day Forecast")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# ==================================================
# FORECAST GRAPH
# ==================================================

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    forecast_df["Day"],
    forecast_df["Forecast Price (₹)"],
    marker="o",
    linewidth=2
)

for i,value in enumerate(
    forecast_df["Forecast Price (₹)"]
):

    ax.annotate(
        f"{value:.2f}",
        (i,value),
        textcoords="offset points",
        xytext=(0,10),
        ha="center"
    )

ax.set_title(
    f"Asian Paints Forecast Using AR({best_lag})"
)

ax.set_xlabel("Future Days")
ax.set_ylabel("Forecast Price (₹)")
ax.grid(True)

st.pyplot(fig)

# ==================================================
# CONCLUSION
# ==================================================

st.header("📌 Conclusion")

st.info(
f"""
Best AR Model : AR({best_lag})

Forecast generated using the AutoRegressive model.

Lower RMSE indicates better prediction accuracy.

The model predicts the next 4 trading day closing prices of Asian Paints.
"""
)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.markdown(
    "### Developed by Sugumar Ranganathan"
)
