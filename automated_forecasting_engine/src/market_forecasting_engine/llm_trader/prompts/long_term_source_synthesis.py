system_message = """
# Role: Long-Term Source Evidence Synthesizer

You are the evidence synthesis layer before the autonomous trader CEO.

Your job is to read every populated provider section supplied in the long-term source context and convert it into a structured, non-duplicative decision brief. You do not make the final Buy/Hold/Sell decision. You prepare the evidence so the CEO can make that decision.

## Evidence Policy

- Treat every populated provider section as decision evidence.
- Do not omit a populated section just because it is noisy, duplicated, bearish, bullish, stale, or uncertain.
- Consolidate duplicate facts across providers, but preserve provider provenance.
- Separate fundamentals, valuation, analyst data, news, filings, transcripts, dividends, company profile, quote context, data quality, conflicts, and stale fields when present.
- If the evidence manifest says a scraped section was not passed, mark coverage as incomplete.
- If raw HTML was not supplied, do not claim to have read raw HTML. Work only from the extracted normalized evidence.

## Output Rules

Return exactly one JSON object matching the schema.
Use concise language, but include enough detail for the CEO to understand why the sources matter.
Every provider and every populated section in the evidence manifest must appear in `coverage.providers`.
""".strip()


user_message = """
Today:
{{ item.today }}

Ticker:
{{ item.ticker }}

Long-Term Source Context:
{{ item.long_term_context_json }}

Task:
Synthesize all populated long-term provider evidence into a structured CEO-ready context.
Do not make the final trading decision.
""".strip()


json_schema = {
    "type": "json_schema",
    "name": "long_term_source_synthesis",
    "description": "Structured synthesis of all populated long-term source evidence before final trader decision.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "coverage": {
                "type": "object",
                "properties": {
                    "all_scraped_sections_passed": {"type": "boolean"},
                    "omitted_section_count": {"type": "integer"},
                    "providers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "provider": {"type": "string"},
                                "scraped_sections": {"type": "array", "items": {"type": "string"}},
                                "passed_sections": {"type": "array", "items": {"type": "string"}},
                                "sections_used_in_synthesis": {"type": "array", "items": {"type": "string"}},
                                "coverage_status": {
                                    "type": "string",
                                    "enum": ["complete", "incomplete", "empty"],
                                },
                            },
                            "required": [
                                "provider",
                                "scraped_sections",
                                "passed_sections",
                                "sections_used_in_synthesis",
                                "coverage_status",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["all_scraped_sections_passed", "omitted_section_count", "providers"],
                "additionalProperties": False,
            },
            "company_context": {"type": "string"},
            "fundamental_read": {"type": "string"},
            "valuation_read": {"type": "string"},
            "analyst_read": {"type": "string"},
            "news_and_sentiment_read": {"type": "string"},
            "filings_and_transcripts_read": {"type": "string"},
            "dividend_and_capital_return_read": {"type": "string"},
            "data_quality_read": {"type": "string"},
            "cross_provider_conflicts": {"type": "array", "items": {"type": "string"}},
            "bullish_evidence": {"type": "array", "items": {"type": "string"}},
            "bearish_evidence": {"type": "array", "items": {"type": "string"}},
            "neutral_or_uncertain_evidence": {"type": "array", "items": {"type": "string"}},
            "decision_implications": {
                "type": "object",
                "properties": {
                    "supports_buy_now": {"type": "string"},
                    "supports_buy_lower": {"type": "string"},
                    "supports_hold": {"type": "string"},
                    "supports_sell_or_trim": {"type": "string"},
                    "confidence_adjustment": {"type": "string"},
                    "key_monitoring_triggers": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "supports_buy_now",
                    "supports_buy_lower",
                    "supports_hold",
                    "supports_sell_or_trim",
                    "confidence_adjustment",
                    "key_monitoring_triggers",
                ],
                "additionalProperties": False,
            },
            "source_provenance_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "coverage",
            "company_context",
            "fundamental_read",
            "valuation_read",
            "analyst_read",
            "news_and_sentiment_read",
            "filings_and_transcripts_read",
            "dividend_and_capital_return_read",
            "data_quality_read",
            "cross_provider_conflicts",
            "bullish_evidence",
            "bearish_evidence",
            "neutral_or_uncertain_evidence",
            "decision_implications",
            "source_provenance_notes",
        ],
        "additionalProperties": False,
    },
}
