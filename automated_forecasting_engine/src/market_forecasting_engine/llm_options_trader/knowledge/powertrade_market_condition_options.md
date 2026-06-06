# Market-Condition Crypto Options Strategies

Source: PowerTrade/PowerDEX on Medium, "Crypto Options Trading Strategies for Every Market Condition", reviewed as strategy reference material.

This note adapts the concepts for the Deribit crypto-options LLM trader. It is strategy context, not an instruction to trade. The current execution interface supports one option order at a time, so multi-leg strategies such as spreads, iron condors, straddles, and strangles must be treated as analytical context until a production multi-leg order manager exists.

## Core Principle

Crypto options strategy should match the market condition:

- Bullish market: prefer call-side upside exposure or put-side income only when risk controls exist.
- Bearish market: prefer put-side downside exposure or call-side income only when covered/risk-defined.
- Range-bound market: premium-selling structures can work, but only with strict risk control and multi-leg execution.
- Volatility expansion expected: long optionality can be useful if the expected move can overcome premium cost.
- Volatility contraction expected: premium-selling can be attractive, but naked short-option risk can be unacceptable.

For this agent, never confuse strategy theory with executable edge. The selected contract must have acceptable bid/ask spread, liquidity, Greeks, expiry, size, and expected move.

## Bullish Strategy Map

### Long Call

Use when the LLM has a clear bullish thesis and wants limited downside risk equal to premium paid.

Good conditions:

- price is breaking upward or continuing a confirmed trend,
- bullish moving-average structure,
- option spread is executable,
- implied volatility is not so expensive that the expected move is unlikely to beat premium,
- enough time remains before expiry for the move to develop.

Failure conditions:

- entry is late after an exhausted spike,
- bid/ask spread is too wide,
- theta is too high for expected holding time,
- resistance is close and reward does not justify premium.

Current agent support: executable as a single buy call limit order.

### Short Put / Cash-Secured Put

This can be bullish or income-oriented, but it creates obligation to buy the underlying if assigned. It requires collateral and assignment-aware risk handling.

For this project:

- do not use naked short puts in the current autonomous agent,
- treat the concept as context for why put premiums may be rich,
- implement only after margin, assignment, collateral, and multi-leg/short-option risk controls are production-ready.

Current agent support: not enabled.

### Bull Call Spread

Buy a lower-strike call and sell a higher-strike call with the same expiry. This reduces entry cost but caps upside.

Useful when:

- bullish view is moderate, not explosive,
- outright calls are too expensive,
- target price is near the short call strike,
- multi-leg execution can enter both legs together.

Current agent support: not executable yet. Use as context: if the LLM likes a call but premium is too expensive, it should mention that a call spread would be preferred, but submit only a supported single-leg order or hold.

## Bearish Strategy Map

### Long Put

Use when the LLM has a clear bearish thesis and wants limited downside risk equal to premium paid.

Good conditions:

- price is breaking downward or continuing a confirmed downtrend,
- bearish moving-average structure,
- downside target is far enough to overcome premium and spread,
- put has reasonable delta/gamma and liquidity,
- entry is not too late after an exhausted selloff.

Failure conditions:

- price is already deeply oversold and near support,
- implied volatility has expanded too much,
- spread is too wide,
- thesis depends on a move that must happen immediately while theta is severe.

Current agent support: executable as a single buy put limit order.

### Covered Call / Short Call

Selling calls can generate income when holding the underlying or expecting limited upside. Naked short calls can have extreme loss risk.

For this project:

- do not use naked short calls in the current autonomous agent,
- covered calls require verified underlying inventory, assignment handling, and margin/risk controls,
- treat call-selling context as useful for reading market maker pricing and volatility, not as an executable action yet.

Current agent support: not enabled.

### Bear Put Spread

Buy a higher-strike put and sell a lower-strike put with the same expiry. This lowers cost and caps downside profit.

Useful when:

- bearish view is moderate,
- outright puts are expensive,
- target price is near the lower strike,
- the spread can be entered as a coordinated multi-leg trade.

Current agent support: not executable yet. Use as context: if a put is directionally correct but too expensive, the LLM should recognize that a put spread would be cleaner, but submit only supported single-leg orders or hold.

## Range-Bound Strategy Map

### Iron Condor

An iron condor sells an out-of-the-money call and put while buying further out-of-the-money protection on both sides. It is designed to collect premium when price remains inside a range, with defined maximum loss.

Useful when:

- market is genuinely range-bound,
- realized volatility is falling,
- implied volatility is high enough to make premium attractive,
- support/resistance range is clear,
- multi-leg execution and max-loss controls are available.

Current agent support: not executable yet. Use as context only. If the LLM detects range-bound conditions and single-leg long options have poor expectancy, it should usually hold.

### Short Strangle

A short strangle sells an out-of-the-money call and put without protective wings. It collects more premium than an iron condor but has large or theoretically unlimited tail risk.

For this project:

- do not use autonomous naked short strangles,
- treat this as a warning that premium-selling can look attractive but has asymmetric risk,
- require explicit multi-leg risk framework before enabling.

Current agent support: not enabled.

### Straddle

A straddle uses call and put at the same strike. A long straddle buys both and profits from a large move in either direction. A short straddle sells both and profits from little movement but has high tail risk.

Long straddle context:

- useful when a large move is expected but direction is unclear,
- expensive when implied volatility is high,
- needs enough move to overcome two premiums.

Short straddle context:

- income strategy for very quiet markets,
- dangerous if the market breaks out,
- requires active delta/gamma risk management.

Current agent support: not executable yet.

## Practical Decision Rules For This Agent

The current LLM trader should apply the article as follows:

- If the market is bullish and executable edge is clear, prefer a supported long call.
- If the market is bearish and executable edge is clear, prefer a supported long put.
- If the market is range-bound, recognize that multi-leg premium-selling structures may be theoretically appropriate, but usually hold because the current agent cannot safely execute those structures.
- If the directional view is moderate but premium is expensive, recognize that spreads would be cleaner, but do not fake a spread with a single-leg order.
- If volatility is high and direction is unclear, avoid buying expensive single-leg options unless the expected move is unusually strong.
- If volatility is low and a breakout is forming, single-leg options may become more attractive because premium can be cheaper.
- Always size so maximum premium loss is acceptable.
- Always prefer no-trade over unsupported strategy execution.

## Required Future Production Work Before Multi-Leg Strategies

Before enabling spreads, condors, strangles, or straddles, implement:

- multi-leg order planning,
- simultaneous or atomic execution where supported,
- net debit/credit validation,
- max loss and payoff diagram calculation,
- margin and collateral checks,
- leg fill monitoring,
- cancel/replace for partially filled legs,
- assignment and expiry handling,
- dashboard display of strategy-level P/L, Greeks, and break-even points.
