# IG Crypto Strategy Playbook

Source: IG Bank Switzerland, "The 5 crypto trading strategies that every trader needs to know", reviewed as strategy reference material.

This knowledge note adapts the concepts for a Deribit crypto-options daily trader. It is not copied trading advice and must be applied only as context. The LLM trader must still decide from live candles, option chain, Greeks, bid/ask spread, liquidity, account exposure, and current open orders.

## General Crypto Trading Context

Crypto markets are highly volatile and can move because of supply and demand, media attention, platform or payment adoption, exchange-specific events, macro sentiment, liquidity shocks, and risk appetite. A strategy should therefore distinguish between:

- a clean directional move worth option premium,
- a choppy/range-bound move where theta and spread can dominate,
- an event or news impulse that can reverse fast,
- and an overextended move where entering late has poor expectancy.

For options, a correct price direction is not enough. The expected move must overcome spread, theta decay, implied volatility, and execution cost.

## 1. Moving Average Crossovers

Moving averages smooth noisy price action into trend lines. They are lagging indicators, so they should not be used as blind entry rules.

Use the 9-period and 21-period moving averages as the primary short-term crossover context:

- Price crossing above an important moving average can support a bullish interpretation.
- Price crossing below an important moving average can support a bearish interpretation.
- SMA/EMA 9 crossing above SMA/EMA 21 is a bullish crossover or golden-cross style signal.
- SMA/EMA 9 crossing below SMA/EMA 21 is a bearish crossover or death-cross style signal.
- If 9 and 21 repeatedly cross while flat, treat the market as choppy and require stronger confirmation before paying option premium.

Option adaptation:

- Bullish crossover can support a call only if the move is early enough, option spread is acceptable, and upside can beat theta.
- Bearish crossover can support a put only if downside momentum is not already exhausted.
- If holding calls and a bearish crossover appears with confirming candles, consider reducing or closing.
- If holding puts and a bullish crossover appears with confirming candles, consider reducing or closing.

## 2. RSI Momentum And Mean-Reversion Context

RSI estimates momentum and can help identify overbought, oversold, divergence, and hidden-divergence conditions. It is most useful when interpreted with the market regime:

- In a range-bound market, high RSI can warn of overbought conditions and low RSI can warn of oversold conditions.
- In a strong trend, RSI can remain high or low for longer than expected, so do not automatically fade it.
- RSI divergence can warn that price momentum is weakening, especially near support/resistance or after a fast impulse.

Option adaptation:

- Do not buy calls only because RSI is high; high RSI may mean momentum, but it may also mean late entry.
- Do not buy puts only because RSI is low; low RSI may mean downside momentum, but it may also mean an oversold bounce risk.
- For short-dated options, use RSI to judge whether the entry is early, exhausted, or likely to chop.
- If RSI contradicts the intended trade, demand better price action and better option liquidity before entering.

## 3. Event-Driven Trading

Crypto can react sharply to news, media attention, exchange issues, regulatory headlines, ETF or institutional flow narratives, technology events, liquidation cascades, and major platform announcements.

Event-driven behavior often creates:

- consolidation before a known event,
- breakout after the market receives new information,
- false breakouts when traders overreact,
- and liquidity gaps where option spreads widen.

Option adaptation:

- Positive event context can support calls only after price confirms strength and option spreads remain executable.
- Negative event context can support puts only after price confirms weakness and liquidity is not too thin.
- Avoid entering immediately into an unknown headline spike if bid/ask spreads explode.
- If the LLM is not given reliable event/news data, it must not invent news. It should treat event-driven strategy as inactive unless the market data itself shows event-like impulse behavior.

## 4. Scalping

Scalping is very short-term trading that tries to capture small intraday movements. It requires fast entries, fast exits, and immediate loss control. It works best when volatility and liquidity are high enough to create repeated small opportunities.

Scalping behavior:

- enter with a clear short-term impulse,
- exit quickly when the trade becomes profitable,
- cut losing trades quickly,
- avoid waiting for long thesis development,
- and avoid wide spreads that erase small gains.

Option adaptation:

- Short-dated crypto options can be scalped only when spread, theta, and liquidity support fast exit.
- The LLM should prefer smaller size and tight exit logic for scalping-style trades.
- If a profitable option scalp begins giving back gains, preserving realized profit is more important than hoping for a larger move.
- If the market is fast but option spread is wide, the apparent scalp edge may be fake.

## 5. DCA And Staged Entry

Dollar-cost averaging splits capital into smaller entries over time instead of committing all at once. For this options trader, classic DCA is not directly suitable for short-dated options because theta decay and expiry risk make repeated averaging dangerous.

Useful adaptation:

- For spot crypto, DCA can support staged accumulation.
- For options, use only controlled staged entries when the thesis improves and the first position is not already invalidated.
- Do not average down losing short-dated options blindly.
- Adding to an option position should require stronger confirmation, better price, acceptable spread, and explicit exposure limits.

## Strategy Selection Guidance

The LLM trader should classify the current market before choosing a strategy:

- Trend continuation: moving average alignment, slope, higher highs/lows or lower highs/lows, supportive volume/liquidity.
- Reversal: exhaustion, failed breakout/breakdown, RSI divergence, reclaim or loss of key moving averages.
- Range/chop: flat moving averages, frequent crossovers, repeated wicks, no clean expansion.
- Event/impulse: sudden large candle, fast volatility expansion, abnormal spread/liquidity behavior.
- Scalp candidate: clean short-term impulse, tight spread, fast fill likelihood, clear exit.

If no strategy has enough evidence after accounting for option execution costs, the professional action is hold.
