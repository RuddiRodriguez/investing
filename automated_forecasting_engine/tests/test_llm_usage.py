import json
from datetime import UTC, datetime

from market_forecasting_engine.llm_trader.responses_api import call_response as call_trader_response
from market_forecasting_engine.openai_responses import call_response as call_shared_response


class _FakeResponse:
    output_text = '{"decision":"Hold"}'

    def model_dump(self, mode="json"):
        return {
            "id": "resp_test",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 10},
            },
        }


class _FakeResponses:
    def create(self, **payload):
        self.payload = payload
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.api_key = "sk-proj-test-secret-1234567890"
        self.responses = _FakeResponses()


def test_shared_openai_response_logs_usage(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    monkeypatch.setenv("OPENAI_USAGE_PRICE_PER_1M_JSON", '{"gpt-test":{"input_per_1m":1.0,"output_per_1m":2.0}}')

    call_shared_response(
        client=_FakeClient(),
        model="gpt-test",
        system_message="system",
        user_message="{{ item.text }}",
        json_schema={"type": "json_schema", "name": "test_schema", "schema": {"type": "object"}},
        item={"text": "hello"},
        usage_context={"purpose": "unit_test"},
    )

    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert rows[0]["status"] == "ok"
    assert rows[0]["usage"]["total_tokens"] == 120
    assert rows[0]["estimated_cost_usd"] == 0.00014
    assert rows[0]["api_key"]["configured"] is True
    assert rows[0]["api_key"]["masked"].startswith("sk-proj")
    assert rows[0]["api_key"]["sha256_12"]
    assert rows[0]["context"]["purpose"] == "unit_test"
    assert rows[0]["request"]["text_format"] == "test_schema"


def test_trader_openai_response_logs_usage(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))

    call_trader_response(
        client=_FakeClient(),
        model="gpt-test",
        system_message="system",
        user_message="{{ item.text }}",
        json_schema={"type": "json_schema", "name": "trader_schema", "schema": {"type": "object"}},
        item={"text": "hello"},
        use_web_search=False,
        search_context_size="low",
        usage_context={"purpose": "trader_unit_test"},
    )

    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert rows[0]["status"] == "ok"
    assert rows[0]["usage"]["input_tokens"] == 100
    assert rows[0]["context"]["purpose"] == "trader_unit_test"
    assert rows[0]["request"]["text_format"] == "trader_schema"


def test_default_openai_pricing_estimates_gpt_5_4_mini_with_web_search(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENAI_USAGE_LOG_FILE", str(log_file))
    monkeypatch.delenv("OPENAI_USAGE_PRICE_PER_1M_JSON", raising=False)

    call_shared_response(
        client=_FakeClient(),
        model="gpt-5.4-mini-2026-03-17",
        system_message="system",
        user_message="{{ item.text }}",
        json_schema={"type": "json_schema", "name": "priced_schema", "schema": {"type": "object"}},
        item={"text": "hello"},
        tools=[{"type": "web_search"}],
    )

    row = json.loads(log_file.read_text().splitlines()[0])
    assert row["cost_breakdown"]["model_price_key"] == "gpt-5.4-mini"
    assert row["cost_breakdown"]["input_rate_per_1m"] == 0.75
    assert row["cost_breakdown"]["cached_input_rate_per_1m"] == 0.075
    assert row["cost_breakdown"]["output_rate_per_1m"] == 4.5
    assert row["cost_breakdown"]["tool_cost_usd"] == 0.01
    assert row["estimated_cost_usd"] == 0.01015825


def test_usage_log_routes_by_branch_and_process(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_USAGE_LOG_FILE", raising=False)
    monkeypatch.setenv("OPENAI_USAGE_BRANCH", "feature/daily trade")
    monkeypatch.setenv("OPENAI_USAGE_PROCESS_NAME", "daily-trade")

    call_shared_response(
        client=_FakeClient(),
        model="gpt-4o-mini",
        system_message="system",
        user_message="{{ item.text }}",
        json_schema={"type": "json_schema", "name": "routed_schema", "schema": {"type": "object"}},
        item={"text": "hello"},
        usage_context={"purpose": "daily_trade_llm_decision"},
    )

    log_files = list((tmp_path / "automated_forecasting_engine" / "runs" / "openai_usage").glob("**/*.jsonl"))
    assert len(log_files) == 1
    today = datetime.now(UTC).strftime("%Y%m%d")
    assert log_files[0].name == f"openai_usage_feature_daily_trade_daily-trade_{today}.jsonl"
    row = json.loads(log_files[0].read_text().splitlines()[0])
    assert row["routing"] == {"branch": "feature/daily trade", "process": "daily-trade"}
