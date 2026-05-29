from ingestion.db import init_db
from strategy_knowledge.strategy_knowledge_agent import ingest_strategy_knowledge

ONEIL_CHAPTER_2_NOTES = """
# Chapter 2 - Key Takeaways
## How to Make Money in Stocks - William O'Neil
Chapter 2 focuses mainly on one core idea:
You must buy leading growth stocks at the right time, not "cheap" stocks.
O'Neil argues that most investors fail because they:
- buy stocks after big declines,
- focus on low P/E ratios or dividends,
- or hold losing positions too long.
Instead, successful investors focus on:
- strong earnings growth,
- strong sales growth,
- market leadership,
- and proper chart patterns.
# Main Concepts
## 1. Buy Strength, Not Weakness
Most people think:
Cheap stock = good opportunity
O'Neil says:
Strong stock making new highs = higher probability winner
Why?
The biggest winning stocks in history:
- were already outperforming the market,
- often near their highs,
- before making massive moves higher.
# 2. The Market Rewards Growth
The biggest stock winners usually had:
- Explosive quarterly earnings
- Strong annual earnings growth
- New products/services
- Industry leadership
- Heavy institutional buying
This becomes part of the CAN SLIM philosophy later.
# 3. Price and Volume Matter
O'Neil emphasizes that charts are:
Supply and demand in visual form
Key idea:
- Rising price + high volume = institutional accumulation.
- Falling price + high volume = institutional selling.
Large funds move markets.
You want to:
Buy when institutions are accumulating
Avoid when institutions are distributing
# 4. Human Nature Repeats
One of the most important ideas in the chapter:
Chart patterns repeat because human psychology never changes.
Fear, greed, hope, and panic create recurring structures.
That's why historical chart studies matter.
# 5. Cut Losses Quickly
A major principle:
Never allow a small loss to become a catastrophic loss.
O'Neil strongly recommends:
Sell if a stock falls 7-8% below your purchase price.
Why?
Because:
- recovering from large losses is mathematically difficult,
- professionals protect capital aggressively.
Example:
-10% loss -> needs +11% recovery
-50% loss -> needs +100% recovery
# 6. Big Winners Often Start From Bases
Before large advances, leading stocks usually:
- pause,
- consolidate,
- and create recognizable chart bases.
Examples:
- Cup with Handle
- Double Bottom
- Flat Base
These represent:
Temporary equilibrium before a new move higher
# 7. Ignore Traditional Metrics Alone
O'Neil criticizes relying mainly on:
- low P/E ratios,
- dividends,
- book value.
Because many historic super-winners:
- looked "expensive" before huge advances.
The market pays premiums for:
Future growth expectations
# Simplified Chapter Flow
Strong company
      v
Institutional accumulation
      v
Base formation
      v
Breakout on volume
      v
Major price advance
# Core Psychological Lesson
Most people:
Buy emotionally
Average down
Hold losers
Sell winners too early
O'Neil's approach:
Buy leaders
Add to winners
Cut losers fast
Follow objective rules
# One-Sentence Summary
The stock market rewards disciplined investors who buy strong growth leaders breaking out of proper bases while controlling risk aggressively.
"""


def main() -> None:
    init_db()
    result = ingest_strategy_knowledge(
        strategy_name="oneil_growth_leadership",
        source_name="How to Make Money in Stocks - Chapter 2 Notes",
        source_type="book_notes",
        raw_text=ONEIL_CHAPTER_2_NOTES,
    )
    print(result)


if __name__ == "__main__":
    main()
