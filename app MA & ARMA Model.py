import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Britannia Stock Forecast Dashboard",
    layout="wide"
)

st.title("📈 Britannia Stock Price Forecasting Dashboard")
st.markdown("### Stock Price Forecasting Using MA and ARMA Models")

# ==================================================
# LOAD DATA
# ==================================================

try:
    df = pd.read_csv("ASIANPAINT.NS.csv")
except:
    st.error("Dataset file not found.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload CSV File (Optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# ==================================================
# DATA CLEANING
# ==================================================

df.columns = df.columns.str.strip().str.lower()

if "unnamed: 0" in df.columns:
    df.rename(columns={"unnamed: 0": "date"}, inplace=True)

if "date" not in df.columns:
    st.error("Date column not found in uploaded file.")
    st.stop()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df.dropna(subset=["date"], inplace=True)

df.drop_duplicates(inplace=True)

df.set_index("date", inplace=True)

if "close" not in df.columns:
    st.error("Close column not found in uploaded file.")
    st.stop()

df["close"] = pd.to_numeric(
    df["close"],
    errors="coerce"
)

df.dropna(subset=["close"], inplace=True)

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

ax.set_title("Britannia Closing Price Trend")
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

adf_result = adfuller(series.dropna())

st.write("ADF Statistic :", round(adf_result[0],4))
st.write("P Value :", round(adf_result[1],6))

if adf_result[1] < 0.05:
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
# TRAIN TEST SPLIT
# ==================================================

train_size = int(len(series_diff)*0.80)

train = series_diff[:train_size]
test = series_diff[train_size:]

# ==================================================
# MA MODEL COMPARISON
# ==================================================

st.header("📊 MA Model Comparison")

ma_results = []

for q in range(1,4):

    model = ARIMA(
        train,
        order=(0,0,q)
    )

    fit = model.fit()

    forecast = fit.forecast(
        steps=len(test)
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast
        )
    )

    ma_results.append([
        f"MA({q})",
        rmse
    ])

ma_df = pd.DataFrame(
    ma_results,
    columns=["Model","RMSE"]
)

ma_df = ma_df.sort_values("RMSE")

st.dataframe(
    ma_df,
    use_container_width=True
)

best_ma = ma_df.iloc[0]["Model"]

st.success(
    f"Best MA Model : {best_ma}"
)

# ==================================================
# ARMA MODEL COMPARISON
# ==================================================

st.header("📊 ARMA Model Comparison")

arma_results = []

for p in range(1,4):

    for q in range(1,4):

        model = ARIMA(
            train,
            order=(p,0,q)
        )

        fit = model.fit()

        forecast = fit.forecast(
            steps=len(test)
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                forecast
            )
        )

        arma_results.append([
            p,
            q,
            rmse
        ])

arma_df = pd.DataFrame(
    arma_results,
    columns=[
        "AR Order",
        "MA Order",
        "RMSE"
    ]
)

arma_df = arma_df.sort_values("RMSE")

st.dataframe(
    arma_df,
    use_container_width=True
)

best_p = int(
    arma_df.iloc[0]["AR Order"]
)

best_q = int(
    arma_df.iloc[0]["MA Order"]
)

st.success(
    f"Best ARMA Model : ARMA({best_p},{best_q})"
)

# ==================================================
# FORECAST
# ==================================================

st.header("🔮 Next 4-Day Forecast")

best_model = ARIMA(
    train,
    order=(1,0,1)
)
best_fit = best_model.fit()

future_diff = best_fit.forecast(steps=4)

future_price = []

current_price = series.iloc[-1]

for diff in future_diff:
    current_price = current_price + diff
    future_price.append(current_price)

forecast_df = pd.DataFrame({

    "Day":[
        "Day 1",
        "Day 2",
        "Day 3",
        "Day 4"
    ],

    "Forecasted Close Price":[
        round(x,2)
        for x in future_price
    ]

})

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
    forecast_df["Forecasted Close Price"],
    marker="o",
    linewidth=2
)

for i,value in enumerate(
    forecast_df["Forecasted Close Price"]
):

    ax.annotate(
        f"{value:.2f}",
        (i,value),
        textcoords="offset points",
        xytext=(0,10),
        ha="center"
    )

ax.set_title(
    f"Britannia Forecast Using ARMA({best_p},{best_q})"
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
Best MA Model : {best_ma}
Best ARMA Model : ARMA({best_p},{best_q})
Forecast for next 4 days generated using the
best ARMA model based on lowest RMSE.
Lower RMSE indicates better prediction accuracy.
"""
)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.markdown(
    "### Developed by Sugumar Ranganathan"
)
