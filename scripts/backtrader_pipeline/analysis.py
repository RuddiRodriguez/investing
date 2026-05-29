"""Advanced backtrader analysis with parameter search and out-of-sample validation."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import backtrader as bt
import numpy as np
import pandas as pd


ProgressCallback = Callable[[int, int, str], None]


def _progress(progress_callback: ProgressCallback | None, step: int, total: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(step, total, message)


def build_backtrader_ohlcv_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    symbol = ticker.upper()
    required_columns = {
        "open": f"OPEN_{symbol}",
        "high": f"HIGH_{symbol}",
        "low": f"LOW_{symbol}",
        "close": symbol,
        "volume": f"VOLUME_{symbol}",
    }
    missing = [name for name, column in required_columns.items() if column not in prices.columns]
    if missing:
        raise ValueError(f"Backtrader analysis for {symbol} requires OHLCV data. Missing: {', '.join(missing)}.")

    frame = pd.DataFrame(index=pd.to_datetime(prices.index, errors="coerce"))
    for name, column in required_columns.items():
        frame[name] = pd.to_numeric(prices[column], errors="coerce")
    frame["openinterest"] = 0.0
    frame = frame.dropna().sort_index()
    if len(frame) < 160:
        raise ValueError(f"Backtrader analysis for {symbol} needs at least 160 valid OHLCV rows. Found {len(frame)}.")
    return frame


class TradeListAnalyzer(bt.Analyzer):
    def start(self) -> None:
        self.trades: list[dict[str, Any]] = []

    def notify_trade(self, trade: bt.Trade) -> None:
        if not trade.isclosed:
            return
        self.trades.append(
            {
                "entry_date": pd.Timestamp(bt.num2date(trade.dtopen)).tz_localize(None),
                "exit_date": pd.Timestamp(bt.num2date(trade.dtclose)).tz_localize(None),
                "entry_price": float(trade.price),
                "size": float(trade.size),
                "bars_held": int(trade.barlen),
                "pnl_gross": float(trade.pnl),
                "pnl_net": float(trade.pnlcomm),
            }
        )

    def get_analysis(self) -> list[dict[str, Any]]:
        return self.trades


class AdaptiveTrendStrategy(bt.Strategy):
    params = dict(
        fast_ema=15,
        slow_ema=60,
        breakout_lookback=20,
        rsi_period=14,
        rsi_entry=55,
        adx_period=14,
        adx_min=20,
        atr_period=14,
        atr_stop_mult=3.0,
        risk_fraction=0.01,
    )

    def __init__(self) -> None:
        self.order = None
        self.trailing_stop: float | None = None
        self.signal_history: list[dict[str, Any]] = []
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.fast_ema)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.slow_ema)
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.adx = bt.ind.ADX(self.data, period=self.p.adx_period)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.macd = bt.ind.MACD(self.data.close)
        self.breakout_high = bt.ind.Highest(self.data.high, period=self.p.breakout_lookback)
        self.breakout_low = bt.ind.Lowest(self.data.low, period=self.p.breakout_lookback)
        self.minimum_history = max(self.p.slow_ema, self.p.breakout_lookback, self.p.atr_period, self.p.adx_period) + 2

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def _record_state(self, signal: str) -> None:
        self.signal_history.append(
            {
                "date": pd.Timestamp(bt.num2date(self.data.datetime[0])).tz_localize(None),
                "close": float(self.data.close[0]),
                "portfolio_value": float(self.broker.getvalue()),
                "cash": float(self.broker.getcash()),
                "position_size": float(self.position.size),
                "ema_fast": float(self.ema_fast[0]) if len(self) >= self.p.fast_ema else np.nan,
                "ema_slow": float(self.ema_slow[0]) if len(self) >= self.p.slow_ema else np.nan,
                "rsi": float(self.rsi[0]) if len(self) >= self.p.rsi_period else np.nan,
                "adx": float(self.adx[0]) if len(self) >= self.p.adx_period else np.nan,
                "atr": float(self.atr[0]) if len(self) >= self.p.atr_period else np.nan,
                "signal": signal,
                "trailing_stop": self.trailing_stop,
            }
        )

    def next(self) -> None:
        signal = "hold"
        if len(self) < self.minimum_history:
            self._record_state("warming_up")
            return
        if self.order is not None:
            self._record_state("pending_order")
            return

        close = float(self.data.close[0])
        atr_value = max(float(self.atr[0]), 1e-6)
        bullish_setup = (
            float(self.ema_fast[0]) > float(self.ema_slow[0])
            and float(self.rsi[0]) >= float(self.p.rsi_entry)
            and float(self.adx[0]) >= float(self.p.adx_min)
            and float(self.macd.macd[0]) >= float(self.macd.signal[0])
            and close >= float(self.breakout_high[-1])
        )
        bearish_exit = (
            float(self.ema_fast[0]) < float(self.ema_slow[0])
            or float(self.rsi[0]) < 50.0
            or close <= float(self.breakout_low[-1])
        )

        if self.position.size > 0:
            proposed_stop = close - float(self.p.atr_stop_mult) * atr_value
            self.trailing_stop = proposed_stop if self.trailing_stop is None else max(self.trailing_stop, proposed_stop)
            if close <= float(self.trailing_stop) or bearish_exit:
                self.order = self.close()
                signal = "exit"
            else:
                signal = "long"
        else:
            self.trailing_stop = None
            if bullish_setup:
                capital_at_risk = float(self.broker.getvalue()) * float(self.p.risk_fraction)
                risk_per_share = max(float(self.p.atr_stop_mult) * atr_value, close * 0.01)
                affordable = int(self.broker.getcash() // close)
                sized = int(capital_at_risk // risk_per_share)
                stake = max(0, min(affordable, sized))
                if stake > 0:
                    self.order = self.buy(size=stake)
                    signal = "enter"
                else:
                    signal = "no_cash"
            else:
                signal = "flat"
        self._record_state(signal)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _extract_metrics(strategy: AdaptiveTrendStrategy, data: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    returns_analysis = strategy.analyzers.returns.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    sqn_analysis = strategy.analyzers.sqn.get_analysis()
    trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()

    trade_total = int(trade_analysis.get("total", {}).get("closed", 0) or 0)
    wins = int(trade_analysis.get("won", {}).get("total", 0) or 0)
    losses = int(trade_analysis.get("lost", {}).get("total", 0) or 0)
    win_rate = wins / trade_total if trade_total else 0.0
    gross_profit = _safe_float(trade_analysis.get("won", {}).get("pnl", {}).get("total", 0.0))
    gross_loss = abs(_safe_float(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0)))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    total_return = _safe_float(returns_analysis.get("rtot"))
    annual_return = _safe_float(returns_analysis.get("rnorm"))
    annual_return_pct = _safe_float(returns_analysis.get("rnorm100"))
    max_drawdown = _safe_float(drawdown_analysis.get("max", {}).get("drawdown")) / 100.0
    sharpe_ratio = _safe_float(sharpe_analysis.get("sharperatio"), default=np.nan)
    sqn_value = _safe_float(sqn_analysis.get("sqn"), default=np.nan)
    final_value = float(strategy.broker.getvalue())
    benchmark_return = float(data["close"].iloc[-1] / data["close"].iloc[0] - 1.0)
    excess_return = total_return - benchmark_return
    return {
        "final_value": final_value,
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "annual_return": annual_return,
        "annual_return_pct": annual_return_pct,
        "benchmark_return": benchmark_return,
        "benchmark_return_pct": benchmark_return * 100.0,
        "excess_return": excess_return,
        "excess_return_pct": excess_return * 100.0,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe_ratio,
        "sqn": sqn_value,
        "trade_count": float(trade_total),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "wins": float(wins),
        "losses": float(losses),
    }


def _score_metrics(metrics: dict[str, float]) -> float:
    sharpe = metrics["sharpe_ratio"] if np.isfinite(metrics["sharpe_ratio"]) else 0.0
    sqn = metrics["sqn"] if np.isfinite(metrics["sqn"]) else 0.0
    trade_penalty = 0.10 if metrics["trade_count"] < 4 else 0.0
    return (
        metrics["excess_return"]
        - 0.75 * metrics["max_drawdown"]
        + 0.12 * sharpe
        + 0.05 * sqn
        + 0.08 * metrics["win_rate"]
        - trade_penalty
    )


def _parameter_grid(mode: str) -> list[dict[str, Any]]:
    if mode == "focused":
        fast_values = [10, 15]
        slow_values = [50, 80]
        rsi_values = [54, 58]
        adx_values = [18, 24]
    elif mode == "thorough":
        fast_values = [10, 15, 20, 25]
        slow_values = [50, 80, 120]
        rsi_values = [52, 55, 58, 62]
        adx_values = [16, 20, 24, 28]
    else:
        fast_values = [10, 15, 20]
        slow_values = [50, 80, 120]
        rsi_values = [52, 56, 60]
        adx_values = [18, 22, 26]

    parameter_sets: list[dict[str, Any]] = []
    for fast_ema, slow_ema, rsi_entry, adx_min in product(fast_values, slow_values, rsi_values, adx_values):
        if fast_ema >= slow_ema:
            continue
        parameter_sets.append(
            {
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "breakout_lookback": max(20, fast_ema + 5),
                "rsi_entry": rsi_entry,
                "adx_min": adx_min,
                "atr_stop_mult": 2.5 if slow_ema <= 80 else 3.0,
                "risk_fraction": 0.01,
            }
        )
    return parameter_sets


def _run_single_backtest(
    data: pd.DataFrame,
    *,
    strategy_params: dict[str, Any],
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_rate)
    cerebro.broker.set_slippage_perc(perc=slippage_rate, slip_open=True, slip_limit=True, slip_match=True, slip_out=False)
    cerebro.broker.set_coc(True)
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.addstrategy(AdaptiveTrendStrategy, **strategy_params)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(TradeListAnalyzer, _name="trade_list")
    strategies = cerebro.run()
    strategy = strategies[0]
    metrics = _extract_metrics(strategy, data, initial_cash)

    signal_history = pd.DataFrame(strategy.signal_history)
    if signal_history.empty:
        signal_history = pd.DataFrame(columns=["date", "close", "portfolio_value", "cash", "position_size", "ema_fast", "ema_slow", "rsi", "adx", "atr", "signal", "trailing_stop"])
    else:
        signal_history = signal_history.sort_values("date").reset_index(drop=True)
    trade_log = pd.DataFrame(strategy.analyzers.trade_list.get_analysis())

    benchmark_value = initial_cash * (data["close"] / data["close"].iloc[0])
    if not signal_history.empty:
        equity_curve = signal_history[["date", "portfolio_value", "close", "ema_fast", "ema_slow", "rsi", "adx", "atr", "signal", "trailing_stop"]].copy()
        benchmark_series = benchmark_value.reindex(pd.to_datetime(equity_curve["date"])).ffill().bfill()
        equity_curve["benchmark_value"] = benchmark_series.to_numpy()
        equity_curve["drawdown"] = equity_curve["portfolio_value"] / equity_curve["portfolio_value"].cummax() - 1.0
    else:
        equity_curve = pd.DataFrame(columns=["date", "portfolio_value", "close", "ema_fast", "ema_slow", "rsi", "adx", "atr", "signal", "trailing_stop", "benchmark_value", "drawdown"])

    return {
        "metrics": metrics,
        "signals": signal_history,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
    }


def run_advanced_backtrader_analysis(
    prices: pd.DataFrame,
    ticker: str,
    *,
    lookback_days: int = 504,
    train_ratio: float = 0.7,
    optimization_mode: str = "balanced",
    initial_cash: float = 100000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    frame = build_backtrader_ohlcv_frame(prices, ticker).tail(lookback_days).copy()
    if len(frame) < 160:
        raise ValueError(f"Not enough history is available after the lookback filter for {ticker.upper()}.")

    split_index = int(len(frame) * train_ratio)
    split_index = min(max(split_index, 120), len(frame) - 40)
    train_data = frame.iloc[:split_index].copy()
    test_data = frame.iloc[split_index:].copy()
    if len(test_data) < 40:
        raise ValueError("Out-of-sample window is too short. Increase lookback or reduce the train ratio.")

    parameter_sets = _parameter_grid(optimization_mode)
    total_steps = len(parameter_sets) + 3
    optimization_rows: list[dict[str, Any]] = []
    best_params = parameter_sets[0]
    best_score = -np.inf

    for step, params in enumerate(parameter_sets, start=1):
        _progress(progress_callback, step, total_steps, f"Optimizing backtrader strategy {step}/{len(parameter_sets)}")
        optimization_result = _run_single_backtest(
            train_data,
            strategy_params=params,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )
        metrics = optimization_result["metrics"]
        score = _score_metrics(metrics)
        row = {**params, **metrics, "score": score}
        optimization_rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params.copy()

    _progress(progress_callback, len(parameter_sets) + 1, total_steps, "Running best in-sample backtrader configuration")
    train_result = _run_single_backtest(
        train_data,
        strategy_params=best_params,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    _progress(progress_callback, len(parameter_sets) + 2, total_steps, "Running out-of-sample validation")
    test_result = _run_single_backtest(
        test_data,
        strategy_params=best_params,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    _progress(progress_callback, len(parameter_sets) + 3, total_steps, "Running full-sample advanced backtrader analysis")
    full_result = _run_single_backtest(
        frame,
        strategy_params=best_params,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    optimization_results = pd.DataFrame(optimization_rows).sort_values("score", ascending=False).reset_index(drop=True)
    metrics_table = pd.DataFrame(
        [
            {"sample": "Train", **train_result["metrics"]},
            {"sample": "Test", **test_result["metrics"]},
            {"sample": "Full", **full_result["metrics"]},
        ]
    )
    return {
        "ticker": ticker.upper(),
        "lookback_days": int(len(frame)),
        "split_date": pd.Timestamp(test_data.index[0]).date().isoformat(),
        "optimization_mode": optimization_mode,
        "optimization_results": optimization_results,
        "best_params": best_params,
        "train": train_result,
        "test": test_result,
        "full": full_result,
        "metrics_table": metrics_table,
    }