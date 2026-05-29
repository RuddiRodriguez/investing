from portfolio.schemas import PortfolioInput, SuggestedPosition


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def get_suggested_direction(alpha_score: float) -> str:
    if alpha_score > 0.25:
        return "long"
    if alpha_score < -0.25:
        return "short"
    return "none"


def build_suggested_position(
    portfolio_input: PortfolioInput,
    max_position_size: float = 0.10,
) -> SuggestedPosition:
    direction = get_suggested_direction(portfolio_input.alpha_score)
    if direction == "none":
        return SuggestedPosition(
            suggested_direction="none",
            suggested_position_size=0.0,
        )

    alpha_strength = clamp(abs(portfolio_input.alpha_score))
    alpha_confidence = clamp(portfolio_input.alpha_confidence)
    risk_multiplier = clamp(portfolio_input.position_size_multiplier)

    suggested_position_size = (
        max_position_size
        * alpha_strength
        * alpha_confidence
        * risk_multiplier
    )
    suggested_position_size = clamp(
        suggested_position_size,
        minimum=0.0,
        maximum=max_position_size,
    )

    return SuggestedPosition(
        suggested_direction=direction,
        suggested_position_size=suggested_position_size,
    )
