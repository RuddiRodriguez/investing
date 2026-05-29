"""Validate whether the risk assessment predicts future risk."""

import pandas as pd
import plotly.express as px


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
RISK_DAYS = 60
FUTURE_DAYS = 20


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def add_past_risk(prices):
    prices["Return"] = prices[PRICE_COLUMN].pct_change()
    prices["Past Volatility"] = prices["Return"].rolling(RISK_DAYS).std() * (252 ** 0.5)

    prices["Past High"] = prices[PRICE_COLUMN].rolling(RISK_DAYS).max()
    prices["Past Drawdown"] = prices[PRICE_COLUMN] / prices["Past High"] - 1

    prices["Past VaR"] = prices["Return"].rolling(RISK_DAYS).quantile(0.05)
    prices["Past Expected Shortfall"] = prices["Return"].rolling(RISK_DAYS).apply(
        lambda returns: returns[returns <= returns.quantile(0.05)].mean()
    )

    prices["Volatility Rank"] = prices["Past Volatility"].rank(pct=True)
    prices["Drawdown Rank"] = (-prices["Past Drawdown"]).rank(pct=True)
    prices["VaR Rank"] = (-prices["Past VaR"]).rank(pct=True)
    prices["Expected Shortfall Rank"] = (-prices["Past Expected Shortfall"]).rank(pct=True)

    prices["Risk Score"] = prices[
        ["Volatility Rank", "Drawdown Rank", "VaR Rank", "Expected Shortfall Rank"]
    ].mean(axis=1)

    prices["Risk Bucket"] = pd.qcut(
        prices["Risk Score"].rank(method="first"),
        3,
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    return prices


def add_future_risk(prices):
    prices["Next Day Return"] = prices["Return"].shift(-1)
    prices["Future Volatility"] = prices["Return"].shift(-1).rolling(FUTURE_DAYS).std().shift(-(FUTURE_DAYS - 1))
    prices["Future Volatility"] = prices["Future Volatility"] * (252 ** 0.5)

    prices["Future Low"] = prices[PRICE_COLUMN].shift(-1).rolling(FUTURE_DAYS).min().shift(-(FUTURE_DAYS - 1))
    prices["Future Worst Loss"] = prices["Future Low"] / prices[PRICE_COLUMN] - 1
    prices["VaR Breach"] = prices["Next Day Return"] < prices["Past VaR"]
    return prices


def print_validation(prices):
    summary = prices.groupby("Risk Bucket", observed=True).agg(
        {
            "Future Volatility": "mean",
            "Future Worst Loss": "mean",
            "VaR Breach": "mean",
        }
    )

    print("Future risk by estimated risk bucket:")
    print(summary.to_string())
    print()
    print(f"Overall VaR breach rate: {prices['VaR Breach'].mean():.2%}")
    print("For a decent 95% VaR, breach rate should be near 5%.")
    print("For decent risk buckets, HIGH should have worse future risk than LOW.")


def plot_validation(prices):
    summary = prices.groupby("Risk Bucket", observed=True).agg(
        {
            "Future Volatility": "mean",
            "Future Worst Loss": "mean",
            "VaR Breach": "mean",
        }
    )
    summary = summary.reset_index()

    fig = px.bar(
        summary,
        x="Risk Bucket",
        y="Future Volatility",
        title="Validation: Future Volatility By Risk Bucket",
    )
    fig.show()

    fig = px.bar(
        summary,
        x="Risk Bucket",
        y="Future Worst Loss",
        title="Validation: Future Worst Loss By Risk Bucket",
    )
    fig.show()

    fig = px.scatter(
        prices,
        x="Risk Score",
        y="Future Worst Loss",
        color="Risk Bucket",
        title="Validation: Risk Score vs Future Worst Loss",
    )
    fig.show()


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    prices = add_past_risk(prices)
    prices = add_future_risk(prices)
    prices = prices.dropna()
    print_validation(prices)
    plot_validation(prices)
