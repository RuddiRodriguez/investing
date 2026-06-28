import json
import os
import runpy
from datetime import UTC, datetime
from pathlib import Path

from market_forecasting_engine.calendar import summarize_calendar_alignment
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_manifest import build_data_manifest
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.data_quality import build_data_quality_report
from market_forecasting_engine.governance import write_audit_bundle
from market_forecasting_engine.llm_trader.profiles import trader_profiles
from market_forecasting_engine.llm_trader.responses_api import call_response
from market_forecasting_engine.llm_trader.source_synthesis import attach_long_term_source_synthesis, run_long_term_source_synthesis
from market_forecasting_engine.llm_handler import normalize_provider_name, resolve_llm_client_profile
from market_forecasting_engine.long_term_sources import (
    DEFAULT_LONG_TERM_SOURCE_PROVIDERS,
    LongTermSourceRequest,
    append_long_term_source_snapshot,
    collect_long_term_source_context,
    load_long_term_source_snapshot_features,
    parse_long_term_source_providers,
)
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL
from market_forecasting_engine.pipeline import ForecastingEngine
from market_forecasting_engine.plots import write_plot_artifacts
from market_forecasting_engine.schema import ForecastConfig
from market_forecasting_engine.security_master import resolve_security_metadata
from market_forecasting_engine.strategy_knowledge import (
    DEFAULT_STRATEGY_CORPUS_DIR,
    DEFAULT_STRATEGY_INDEX_PATH,
    StrategyKnowledgeRequest,
    attach_strategy_knowledge_context,
    build_strategy_knowledge_context,
)


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
    emit_progress(args, "SOURCES", "synthesizing long-term source evidence for CEO context")
    source_synthesis = run_long_term_source_synthesis(
        report=report,
        llm_provider=getattr(args, "llm_provider", None),
        llm_model=args.llm_model,
        reasoning_effort=args.reasoning_effort,
        llm_env_file=args.llm_env_file,
        timeout_seconds=float(args.llm_timeout),
        dry_run=bool(args.dry_run),
    )
    attach_long_term_source_synthesis(report, source_synthesis)
    emit_progress(args, "SOURCES", "long-term source synthesis ready", status=source_synthesis.get("status"))
    emit_progress(args, "STRATEGY", "retrieving durable strategy knowledge for CEO context")
    if not getattr(args, "disable_strategy_knowledge", False):
        strategy_context = build_strategy_knowledge_context(
            report,
            StrategyKnowledgeRequest(
                ticker=args.ticker.upper(),
                corpus_dir=getattr(args, "strategy_knowledge_corpus_dir", str(DEFAULT_STRATEGY_CORPUS_DIR)),
                index_path=getattr(args, "strategy_knowledge_index", str(DEFAULT_STRATEGY_INDEX_PATH)),
                llm_env_file=args.llm_env_file,
                max_chunks=int(getattr(args, "strategy_knowledge_max_chunks", 8)),
                rebuild_index=bool(getattr(args, "strategy_knowledge_rebuild_index", False)),
                timeout_seconds=int(args.llm_timeout),
            ),
        )
    else:
        strategy_context = {
            "status": "disabled",
            "reason": "disabled by --disable-strategy-knowledge",
            "decision_policy": {
                "feeds_ceo_llm": False,
                "overrides_model_validation": False,
                "overrides_risk_gates": False,
            },
        }
    attach_strategy_knowledge_context(report, strategy_context)
    emit_progress(args, "STRATEGY", "strategy knowledge context ready", status=strategy_context.get("status"))
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
            provider=resolve_llm_provider(getattr(args, "llm_provider", None)),
            model=resolve_llm_model(args.llm_model, provider=resolve_llm_provider(getattr(args, "llm_provider", None))),
            web_search=not args.no_web_search and resolve_llm_provider(getattr(args, "llm_provider", None)) == "openai",
        )
        load_env(args.llm_env_file)
        provider = resolve_llm_provider(getattr(args, "llm_provider", None))
        client = openai_client_for_provider(provider, timeout=float(args.llm_timeout))
        payload, raw_response, decision = call_response(
            client=client,
            provider=provider,
            model=resolve_llm_model(args.llm_model, provider=provider),
            system_message=prompt["system_message"],
            user_message=prompt["user_message"],
            json_schema=prompt["json_schema"],
            reasoning_effort=args.reasoning_effort,
            item=item,
            use_web_search=not args.no_web_search and provider == "openai",
            search_context_size=args.search_context_size,
            require_web_search=not args.no_web_search and provider == "openai",
            usage_context={"purpose": "autonomous_trader_decision", "ticker": args.ticker.upper(), "profile": args.profile, "provider": provider},
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
            provider=resolve_llm_provider(getattr(args, "summary_provider", None) or getattr(args, "llm_provider", None)),
            model=resolve_llm_model(args.summary_model or args.llm_model, provider=resolve_llm_provider(getattr(args, "summary_provider", None) or getattr(args, "llm_provider", None))),
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
        summary_provider = resolve_llm_provider(getattr(args, "summary_provider", None) or getattr(args, "llm_provider", None))
        summary_client = openai_client_for_provider(summary_provider, timeout=float(args.llm_timeout))
        summary_payload, summary_raw_response, trader_summary = call_response(
            client=summary_client,
            provider=summary_provider,
            model=resolve_llm_model(args.summary_model or args.llm_model, provider=summary_provider),
            system_message=summary_prompt["system_message"],
            user_message=summary_prompt["user_message"],
            json_schema=summary_prompt["json_schema"],
            reasoning_effort=args.summary_reasoning_effort,
            item=summary_item,
            use_web_search=False,
            search_context_size="low",
            usage_context={"purpose": "nontechnical_trader_summary", "ticker": args.ticker.upper(), "profile": args.profile, "provider": summary_provider},
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
        "long_term_source_synthesis": source_synthesis,
        "strategy_knowledge_context": strategy_context,
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


def resolve_llm_provider(provider: str | None) -> str:
    return normalize_provider_name(provider or os.environ.get("LLM_PROVIDER") or "openai")


def resolve_llm_model(model: str | None, *, provider: str | None = None) -> str:
    return resolve_llm_client_profile(provider=resolve_llm_provider(provider), model=model).model


def openai_client_for_provider(provider: str, *, timeout: float):
    if resolve_llm_provider(provider) != "openai":
        return None
    from openai import OpenAI

    return OpenAI(timeout=timeout)


def run_forecast(args, output_dir):
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    enable_long_term_sources = bool(getattr(args, "enable_long_term_sources", True))
    provider_arg = getattr(args, "long_term_source_providers", None)
    long_term_source_providers = (
        parse_long_term_source_providers(provider_arg)
        if provider_arg
        else DEFAULT_LONG_TERM_SOURCE_PROVIDERS
    )
    config = ForecastConfig(
        ticker=args.ticker,
        horizons=horizons,
        target_column=args.target_column.lower(),
        selection_metric=args.selection_metric,
        confidence_level=args.confidence_level,
        validation_workers=int(getattr(args, "validation_workers", 0)),
        include_lightgbm=not args.no_lightgbm,
        include_statistical_models=not args.no_statistical_models,
        include_lstm=args.include_lstm,
        deep_learning_profile=getattr(args, "deep_learning_profile", "off"),
        llm_env_file=args.llm_env_file,
        enable_long_term_sources=enable_long_term_sources,
        long_term_source_providers=long_term_source_providers,
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
    long_term_context = None
    if enable_long_term_sources:
        long_term_output_dir = (
            Path(getattr(args, "long_term_source_output_dir"))
            if getattr(args, "long_term_source_output_dir", None)
            else ((output_dir / "long_term_sources") if output_dir else None)
        )
        long_term_snapshot_dir = (
            Path(getattr(args, "long_term_source_snapshot_dir"))
            if getattr(args, "long_term_source_snapshot_dir", None)
            else ((output_dir / "data" / "long_term_source_snapshots") if output_dir else Path("automated_forecasting_engine/runs/long_term_source_snapshots"))
        )
        long_term_context = collect_long_term_source_context(
            LongTermSourceRequest(
                ticker=args.ticker,
                providers=long_term_source_providers,
                env_file=getattr(args, "long_term_source_env_file", None) or args.llm_env_file,
                output_dir=long_term_output_dir,
                start_date=args.start,
                end_date=args.end,
            )
        )
        snapshot_path = append_long_term_source_snapshot(long_term_context, long_term_snapshot_dir, ticker=args.ticker)
        long_term_features, snapshot_feature_metadata = load_long_term_source_snapshot_features(
            args.ticker,
            long_term_snapshot_dir,
            prices.index,
        )
        if not long_term_features.empty:
            prices = normalize_price_frame(prices.join(long_term_features, how="left"), target_column=config.target_column)
        long_term_context.setdefault("model_feature_policy", {})["snapshot_feature_metadata"] = snapshot_feature_metadata
        data_manifest["long_term_sources"] = {
            "status": long_term_context.get("status"),
            "providers_requested": long_term_context.get("providers_requested", []),
            "provider_summaries": long_term_context.get("provider_summaries", {}),
            "artifacts": long_term_context.get("artifacts", {}),
            "model_feature_policy": long_term_context.get("model_feature_policy", {}),
            "snapshot_path": str(snapshot_path),
            "snapshot_feature_metadata": snapshot_feature_metadata,
        }
    report = ForecastingEngine(config).run(
        prices,
        data_manifest=data_manifest,
        data_quality_report=data_quality,
        security_metadata=security_metadata,
        long_term_context=long_term_context,
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
            "long_term_context": decision.get("long_term_context", {}),
            "strategy_knowledge_context": decision.get("strategy_knowledge_context", {}),
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
                "strategy_knowledge_context": result.get("strategy_knowledge_context"),
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
