import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Stock Price Forecast", layout="wide")

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Generate Forecast"):

    # Download last 5 years of data
    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=True
    )

    if data.empty:
        st.error("Invalid ticker or no data found.")
        st.stop()

    close_prices = data["Close"]

    st.subheader("Historical Stock Price")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(close_prices.index,
            close_prices.values,
            label="Closing Price")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    st.subheader("ARIMA Model Selection")

    auto_model = auto_arima(
        close_prices,
        seasonal=False,
        trace=False,
        suppress_warnings=True,
        stepwise=True
    )

    order = auto_model.order

    st.write(f"Selected ARIMA Order: {order}")

    model = ARIMA(close_prices, order=order)
    fitted_model = model.fit()

    # Forecast until June 2027
    last_date = close_prices.index[-1]

    target_date = pd.Timestamp("2027-06-30")

    days_to_forecast = (target_date - last_date).days

    if days_to_forecast <= 0:
        st.warning(
            "Target date already passed relative to available data."
        )
        st.stop()

    forecast = fitted_model.forecast(steps=days_to_forecast)

    predicted_price = forecast.iloc[-1]

    st.subheader("Forecast Result")

    st.metric(
        "Predicted Price on June 2027",
        f"${predicted_price:.2f}"
    )

    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_to_forecast,
        freq="D"
    )

    forecast_series = pd.Series(
        forecast.values,
        index=forecast_dates
    )

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(
        close_prices.index,
        close_prices.values,
        label="Historical"
    )

    ax2.plot(
        forecast_series.index,
        forecast_series.values,
        label="Forecast"
    )

    ax2.axvline(
        last_date,
        linestyle="--"
    )

    ax2.legend()

    st.pyplot(fig2)
