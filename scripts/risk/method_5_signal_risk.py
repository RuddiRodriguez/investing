"""Risk method 5: risk when the direction model says UP."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.direction import method_5_lgbm_graph_test as direction_model


CSV_FILE = "stock_prices.csv"
OPTUNA_TRIALS = 10


def get_max_drawdown(equity):
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def calculate_signal_risk(prices):
    direction_model.OPTUNA_TRIALS = OPTUNA_TRIALS
    checked_prices = direction_model.check_predictions(prices)
    up_signals = checked_prices[checked_prices["Signal"] == "UP"]

    average_return = up_signals["Future Return"].mean()
    worst_return = up_signals["Future Return"].min()
    var = up_signals["Future Return"].quantile(0.05)
    expected_shortfall = up_signals[up_signals["Future Return"] <= var]["Future Return"].mean()
    max_drawdown = get_max_drawdown(checked_prices["Strategy Equity"])

    return up_signals, average_return, worst_return, var, expected_shortfall, max_drawdown


if __name__ == "__main__":
    prices = direction_model.read_prices(CSV_FILE)
    up_signals, average_return, worst_return, var, expected_shortfall, max_drawdown = calculate_signal_risk(prices)

    print(f"UP signals: {len(up_signals)}")
    print(f"Average future return after UP signal: {average_return:.2%}")
    print(f"Worst future return after UP signal: {worst_return:.2%}")
    print(f"VaR after UP signal: {var:.2%}")
    print(f"Expected shortfall after UP signal: {expected_shortfall:.2%}")
    print(f"Strategy max drawdown: {max_drawdown:.2%}")
