from ingestion.db import init_db
from alpha.alpha_agent import run_combined_alpha_signal_agent


def main() -> None:
    init_db()
    result = run_combined_alpha_signal_agent(
        sector="semiconductors",
    )
    print(result)


if __name__ == "__main__":
    main()
