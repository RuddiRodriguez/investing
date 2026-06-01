import json
import os
import runpy
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.governance import write_audit_bundle
from market_forecasting_engine.llm_trader.profiles import trader_profiles
from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL
from market_forecasting_engine.pipeline import ForecastingEngine
from market_forecasting_engine.plots import write_plot_artifacts
from market_forecasting_engine.schema import ForecastConfig
from market_forecasting_engine.security_master import resolve_security_metadata


def run_autonomous_trader(args):
    output_dir = Path(args.output_dir) if args.output_dir else None
    emit_progress(args, "FORECAST", "starting technical forecast pipeline", output_dir=output_dir or "none")
    report, prices = run_forecast(args, output_dir)
    emit_progress(
        args,
        "FORECAST",
        "technical forecast pipeline finished",
        current_price=report.get("current_price"),
        suggested_action=report.get("suggested_action"),
    )
    emit_progress(args, "PACKET", "building technical and portfolio packets")
    technical_packet = build_technical_packet(report)
    portfolio_context = build_portfolio_context(args)
    profile = trader_profiles[args.profile]
    prompt = load_prompt(args.prompt)
    item = {
        "today": datetime.now(UTC).date().isoformat(),
        "ticker": args.ticker.upper(),
        "trader_name": args.trader_name,
        "trader_profile_json": json.dumps(profile, indent=2, sort_keys=True),
        "portfolio_context_json": json.dumps(portfolio_context, indent=2, sort_keys=True),
        "technical_packet_json": json.dumps(technical_packet, indent=2, sort_keys=True),
    }
    if args.dry_run:
        emit_progress(args, "LLM", "dry-run enabled; skipping autonomous trader LLM call")
        payload = None
        raw_response = None
        decision = {
            "status": "dry_run",
            "decision": None,
            "reason": "LLM call was skipped; prompt packet was built only.",
        }
    else:
        emit_progress(
            args,
            "LLM",
            "calling autonomous trader LLM",
            model=resolve_llm_model(args.llm_model),
            web_search=not args.no_web_search,
        )
        load_env(args.llm_env_file)
        client = OpenAI(timeout=float(args.llm_timeout))
        payload, raw_response, decision = call_response(
            client=client,
            model=resolve_llm_model(args.llm_model),
            system_message=prompt["system_message"],
            user_message=prompt["user_message"],
            json_schema=prompt["json_schema"],
            reasoning_effort=args.reasoning_effort,
            item=item,
            use_web_search=not args.no_web_search,
            search_context_size=args.search_context_size,
            usage_context={"purpose": "autonomous_trader_decision", "ticker": args.ticker.upper(), "profile": args.profile},
        )
        emit_progress(
            args,
            "LLM",
            "autonomous trader LLM decision received",
            decision=decision.get("decision"),
            confidence=decision.get("confidence"),
        )
    summary_payload = None
    summary_raw_response = None
    if args.dry_run:
        emit_progress(args, "SUMMARY", "dry-run enabled; skipping non-technical summary LLM call")
        trader_summary = {
            "status": "dry_run",
            "reason": "Summary LLM call was skipped because dry-run is enabled.",
        }
    elif args.no_summary:
        emit_progress(args, "SUMMARY", "non-technical summary disabled")
        trader_summary = {
            "status": "disabled",
            "reason": "Non-technical summary call was disabled.",
        }
    else:
        emit_progress(
            args,
            "SUMMARY",
            "calling non-technical summary LLM",
            model=resolve_llm_model(args.summary_model or args.llm_model),
        )
        summary_prompt = load_prompt(args.summary_prompt)
        currency_context = build_currency_context(args)
        summary_item = {
            "ticker": args.ticker.upper(),
            "trader_name": args.trader_name,
            "trader_profile_json": json.dumps(profile, indent=2, sort_keys=True),
            "portfolio_context_json": json.dumps(portfolio_context, indent=2, sort_keys=True),
            "currency_context_json": json.dumps(currency_context, indent=2, sort_keys=True),
            "trader_decision_json": json.dumps(decision, indent=2, sort_keys=True),
            "technical_packet_json": json.dumps(technical_packet, indent=2, sort_keys=True),
        }
        summary_payload, summary_raw_response, trader_summary = call_response(
            client=client,
            model=resolve_llm_model(args.summary_model or args.llm_model),
            system_message=summary_prompt["system_message"],
            user_message=summary_prompt["user_message"],
            json_schema=summary_prompt["json_schema"],
            reasoning_effort=args.summary_reasoning_effort,
            item=summary_item,
            use_web_search=False,
            search_context_size="low",
            usage_context={"purpose": "nontechnical_trader_summary", "ticker": args.ticker.upper(), "profile": args.profile},
        )
        emit_progress(args, "SUMMARY", "non-technical summary received")
    result = {
        "ticker": args.ticker.upper(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "trader_name": args.trader_name,
        "trader_profile": profile,
        "forecast_report": report,
        "technical_packet": technical_packet,
        "portfolio_context": portfolio_context,
        "llm_decision": decision,
        "llm_prompt_payload": payload,
        "llm_raw_response": raw_response,
        "nontechnical_summary": trader_summary,
        "summary_prompt_payload": summary_payload,
        "summary_raw_response": summary_raw_response,
    }
    if output_dir:
        emit_progress(args, "OUTPUT", "writing trader artifacts", output_dir=output_dir)
        write_trader_outputs(output_dir, result)
        emit_progress(args, "OUTPUT", "trader artifacts written", output_dir=output_dir)
    return result


def resolve_llm_model(model: str | None) -> str:
    return model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def run_forecast(args, output_dir):
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    config = ForecastConfig(
        ticker=args.ticker,
        horizons=horizons,
        target_column=args.target_column.lower(),
        selection_metric=args.selection_metric,
        confidence_level=args.confidence_level,
        include_lightgbm=not args.no_lightgbm,
        include_statistical_models=not args.no_statistical_models,
        include_lstm=args.include_lstm,
        llm_env_file=args.llm_env_file,
    )
    provider = (args.provider or ("csv" if args.csv else "yahoo")).lower()
    if args.csv:
        provider = "csv"
    request = DataRequest(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target_column=config.target_column,
        interval=args.interval,
        adjustment_policy=args.adjustment_policy,
        source_path=args.csv,
    )
    provider_result = load_prices_with_provider(provider, request)
    prices = normalize_price_frame(provider_result.frame, target_column=config.target_column)
    security_metadata = resolve_security_metadata(
        ticker=args.ticker,
        prices=prices,
        provider_metadata=provider_result.metadata,
        adjustment_policy=args.adjustment_policy,
    )
    data_quality = build_data_quality_report(prices, target_column=config.target_column, calendar=args.calendar)
    data_manifest = build_data_manifest(
        prices=prices,
        ticker=args.ticker,
        target_column=config.target_column,
        provider=provider,
        source=args.csv,
        request=request.to_dict(),
        artifacts={"primary": provider_result.metadata.get("artifacts", {})},
        security_metadata=security_metadata,
        calendar_summary=summarize_calendar_alignment(prices, calendar=args.calendar),
    )
    report = ForecastingEngine(config).run(
        prices,
        data_manifest=data_manifest,
        data_quality_report=data_quality,
        security_metadata=security_metadata,
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.write_plots:
            plot_artifacts = write_plot_artifacts(
                report,
                prices,
                output_dir,
                target_column=config.target_column,
                chart_scale=args.chart_scale,
            )
            report["artifacts"] = {"plots": plot_artifacts}
        write_audit_bundle(report, output_dir)
    return report, prices


def build_technical_packet(report):
    technical = report.get("technical_view", {})
    decision = report.get("decision_view", {})
    operations = report.get("operations_view", {})
    selection = report.get("selection_view", {})
    trade_risk = report.get("trade_risk_view", {}).get("chapter_23_30_trade_risk_plan", {})
    portfolio_risk = report.get("portfolio_view", {}).get("chapter_31_42_portfolio_capital_risk", {})
    discipline = report.get("discipline_view", {}).get("chapter_39_43_discipline_governance", {})
    chapter_18 = decision.get("chapter_18_tactical_problem", {})
    chapter_19 = operations.get("chapter_19_validation", {})
    chapter_20 = selection.get("chapter_20_ticker_suitability", {})
    chapter_21 = selection.get("chapter_21_chart_selection", {})
    chapter_13 = technical.get("chapter_13_support_resistance", {})
    chapter_14 = technical.get("chapter_14_trendlines", {})
    chapter_15 = technical.get("chapter_15_major_trendlines", {})
    return {
        "ticker": report.get("ticker"),
        "as_of_date": report.get("as_of_date"),
        "current_price": report.get("current_price"),
        "suggested_action": report.get("suggested_action"),
        "part_i_suggested_action": report.get("part_i_suggested_action"),
        "risk_level": report.get("risk_level"),
        "risk_warning": report.get("risk_warning"),
        "forecasts": [
            {
                "horizon_days": item.get("horizon_days"),
                "selected_model": item.get("selected_model"),
                "expected_direction": item.get("expected_direction"),
                "expected_return": item.get("expected_return"),
                "directional_confidence": item.get("directional_confidence"),
                "predicted_price": item.get("predicted_price"),
                "lower_price": item.get("lower_price"),
                "upper_price": item.get("upper_price"),
                "validation_metrics": item.get("validation_metrics"),
            }
            for item in report.get("forecasts", [])
        ],
        "trend": {
            "trend_state": technical.get("trend_state", {}),
            "dow_theory": technical.get("dow_theory", {}).get("primary_trend", {}),
            "magee_basing": technical.get("magee_basing_points", {}).get("preferred", {}),
            "chapter_13_support": chapter_13.get("support_zones", {}).get("nearest", {}),
            "chapter_13_resistance": chapter_13.get("resistance_zones", {}).get("nearest", {}),
            "chapter_14_trendline": chapter_14.get("trendlines", {}).get("preferred", {}),
            "chapter_15_major_trend": chapter_15.get("stock_major_trend", {}),
        },
        "patterns": {
            "reversal": technical.get("reversal_patterns", {}).get("preferred", {}),
            "triangle": technical.get("triangle_patterns", {}).get("preferred", {}),
            "rectangle": technical.get("chapter_9_patterns", {}).get("rectangle_patterns", {}).get("preferred", {}),
            "multi_top_bottom": technical.get("chapter_9_patterns", {}).get("multi_top_bottom_patterns", {}).get("preferred", {}),
            "gap": technical.get("chapter_12_gaps", {}).get("classified_gaps", {}).get("preferred", {}),
        },
        "decision_governance": {
            "chapter_18_rule_based_action": chapter_18.get("rule_based_action"),
            "chapter_18_final_action": chapter_18.get("final_action"),
            "chapter_18_rule_gate": chapter_18.get("rule_gate", {}),
            "chapter_18_trade_plan": chapter_18.get("trade_plan", {}),
            "production_gate": decision.get("production_gate", {}),
            "mean_reversion_dip_buy": decision.get("mean_reversion_dip_buy", {}),
            "chapter_19_status": chapter_19.get("status"),
            "chapter_19_action_gate": chapter_19.get("action_gate", {}),
            "chapter_20_profile_fit": chapter_20.get("profile_fit", {}),
            "chapter_21_chart_selection": chapter_21.get("chart_selection", {}),
            "trade_risk_commitment": trade_risk.get("commitment", {}),
            "trade_risk_execution_summary": trade_risk.get("execution_summary", {}),
            "portfolio_capital_gate": portfolio_risk.get("portfolio_capital_gate", {}),
            "portfolio_capital_summary": portfolio_risk.get("capital_summary", {}),
            "discipline_gate": discipline.get("discipline_gate", {}),
            "discipline_status": discipline.get("status"),
        },
        "options_decision": {
            "mode": report.get("options_decision", {}).get("mode"),
            "best_trade": report.get("options_decision", {}).get("best_trade", {}),
            "policy": report.get("options_decision", {}).get("policy", {}),
        },
        "backtests": report.get("backtests", {}),
        "selection_metric": report.get("selection_metric"),
        "data_version": report.get("data_version"),
        "model_version": report.get("model_version"),
    }


def build_portfolio_context(args):
    return {
        "holding_status": args.holding_status,
        "entry_price": args.entry_price,
        "quantity": args.quantity,
        "position_value": args.position_value,
        "account_equity": args.account_equity,
        "notes": args.portfolio_notes,
    }


def build_currency_context(args):
    if args.usd_eur_rate:
        return {
            "base_currency": "USD",
            "converted_currency": "EUR",
            "usd_to_eur": float(args.usd_eur_rate),
            "source": "manual_cli_override",
            "status": "available",
            "instruction": "Show every price as dollars first and euros in parentheses, for example $1,505.22 (€1,385.00).",
        }
    if args.dry_run:
        return {
            "base_currency": "USD",
            "converted_currency": "EUR",
            "usd_to_eur": None,
            "source": "not_requested_in_dry_run",
            "status": "unavailable",
            "instruction": "Dry-run skipped live FX conversion.",
        }
    try:
        import yfinance as yf

        frame = yf.download("EURUSD=X", period="5d", interval="1d", auto_adjust=True, progress=False)
        close = frame["Close"].dropna()
        eurusd = float(close.iloc[-1])
        return {
            "base_currency": "USD",
            "converted_currency": "EUR",
            "usd_to_eur": 1.0 / eurusd,
            "eurusd": eurusd,
            "source": "yahoo_finance_EURUSD=X",
            "as_of": str(close.index[-1].date()),
            "status": "available",
            "instruction": "Show every price as dollars first and euros in parentheses, for example $1,505.22 (€1,385.00).",
        }
    except Exception as exc:
        return {
            "base_currency": "USD",
            "converted_currency": "EUR",
            "usd_to_eur": None,
            "source": "yahoo_finance_EURUSD=X",
            "status": "unavailable",
            "error": str(exc)[:300],
            "instruction": "Show dollar prices. If euro conversion is unavailable, say EUR conversion unavailable instead of inventing it.",
        }


def load_prompt(path):
    return runpy.run_path(path)


def load_env(path):
    env_path = Path(path or ".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#") and "=" in clean:
                name, value = clean.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def write_trader_outputs(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trader_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "trader_decision_only.json").write_text(
        json.dumps(result["llm_decision"], indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "trader_summary.json").write_text(
        json.dumps(result["nontechnical_summary"], indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "trader_prompt_packet.json").write_text(
        json.dumps(
            {
                "trader_profile": result["trader_profile"],
                "portfolio_context": result["portfolio_context"],
                "technical_packet": result["technical_packet"],
                "llm_prompt_payload": result["llm_prompt_payload"],
                "summary_prompt_payload": result["summary_prompt_payload"],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def emit_progress(args, stage, message, **fields):
    logger = getattr(args, "progress_logger", None)
    if logger:
        logger(stage, message, **fields)
