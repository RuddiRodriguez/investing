import pandas as pd
import pytest

from scripts.direction.method_5_lgbm import (
    CLASS_BY_SIGNAL,
    DOWN_SIGNAL,
    FUTURE_DAYS,
    HOLD_SIGNAL,
    UP_SIGNAL,
    add_target,
    get_signal,
    get_trade_return,
)


def test_add_target_builds_three_classes() -> None:
    prices = pd.DataFrame(
        {
            "Close": [100.0] * FUTURE_DAYS + [97.0, 100.0, 103.0],
        }
    )

    labeled = add_target(prices)

    assert labeled.loc[0, "Target"] == CLASS_BY_SIGNAL[DOWN_SIGNAL]
    assert labeled.loc[1, "Target"] == CLASS_BY_SIGNAL[HOLD_SIGNAL]
    assert labeled.loc[2, "Target"] == CLASS_BY_SIGNAL[UP_SIGNAL]


def test_get_signal_requires_confident_direction() -> None:
    assert get_signal(probability_up=0.62, probability_down=0.18) == UP_SIGNAL
    assert get_signal(probability_up=0.12, probability_down=0.58) == DOWN_SIGNAL
    assert get_signal(probability_up=0.54, probability_down=0.20) == HOLD_SIGNAL
    assert get_signal(probability_up=0.60, probability_down=0.61) == DOWN_SIGNAL


def test_get_trade_return_handles_long_only_and_hold_costs() -> None:
    assert get_trade_return(UP_SIGNAL, future_return=0.03, transaction_cost=0.001) == pytest.approx(0.029)
    assert get_trade_return(DOWN_SIGNAL, future_return=-0.03, transaction_cost=0.001) == 0.0
    assert get_trade_return(HOLD_SIGNAL, future_return=0.03, transaction_cost=0.001) == 0.0