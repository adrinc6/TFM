"""OpenAI-backed business research export.

This layer is optional and cache-first. When OpenAI credentials are not
configured, it still writes deterministic JSON so the viewer and academic audit
trail remain complete.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from environment import ENABLE_OPENAI_RESEARCH, OPENAI_API_KEY, OPENAI_MODEL, PROCESSED_DIR, RAW_DIR, Settings
from module.research.thesis import enrich_with_thesis_scores
from module.utils import read_parquet, write_json

log = logging.getLogger(__name__)


RESEARCH_FIELDS = [
    "business_summary",
    "moat",
    "catalysts",
    "risks",
    "recent_news",
    "thesis",
    "exit_thesis",
    "classification",
]


def build_openai_research(settings: Settings) -> Path:
    scored = enrich_with_thesis_scores(read_parquet(PROCESSED_DIR / "scored_universe.parquet"))
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = settings.run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for stale in (settings.run_dir / "research_ai_history.csv", settings.run_dir / "research_ai"):
        if stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
    history_cols = [
        "snapshot_date", "ticker", "business_summary", "moat_analysis", "catalyst",
        "risk_summary", "investment_thesis", "exit_thesis", "opportunity_type",
        "conviction_score", "buy_today_score", "would_buy_today",
    ]
    scored[[col for col in history_cols if col in scored.columns]].to_csv(audit_dir / "research_ai_history.csv", index=False)
    latest_date = sorted(scored["snapshot_date"].unique())[-1]
    latest = scored[scored["snapshot_date"] == latest_date].copy()
    news = _load_news()
    output_dir = audit_dir / "research_ai"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in latest.itertuples(index=False):
        payload = _research_payload(pd.Series(row._asdict()), news.get(row.ticker, []))
        path = output_dir / f"{row.ticker}.json"
        write_json(payload, path)
        rows.append({"snapshot_date": latest_date, "ticker": row.ticker, "path": str(path), **payload})

    pd.DataFrame(rows).to_csv(settings.run_dir / "research_ai.csv", index=False)
    log.info(
        "Research AI artifacts written to %s latest_rows=%s history_rows=%s",
        output_dir,
        len(rows),
        len(scored),
    )
    return output_dir


def _research_payload(row: pd.Series, news_items: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = _fallback_payload(row, news_items)
    if not ENABLE_OPENAI_RESEARCH or not OPENAI_API_KEY:
        return fallback

    try:
        generated = _call_openai(row, news_items)
        return {**fallback, **{key: generated.get(key, fallback[key]) for key in RESEARCH_FIELDS}}
    except Exception as exc:
        log.warning("[%s] OpenAI research failed; using fallback: %s", row["ticker"], exc)
        return fallback


def _call_openai(row: pd.Series, news_items: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = {
        "ticker": row["ticker"],
        "sector": row.get("sector", "Unknown"),
        "classification": row["opportunity_type"],
        "scores": {
            "quality_probability": row["quality_probability"],
            "improvement_probability": row["improvement_probability"],
            "mispricing_probability": row["mispricing_probability"],
            "alpha_probability": row["alpha_probability"],
            "business_quality_score": row["business_quality_score"],
            "expectation_gap": row["expectation_gap"],
            "valuation_score": row["valuation_score"],
        },
        "existing_thesis": row["investment_thesis"],
        "news": news_items[:3],
    }
    body = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are a long-term GARP business analyst. Return only valid JSON with "
                    "business_summary, moat, catalysts, risks, recent_news, thesis, exit_thesis, classification. "
                    "Do not give financial advice; explain the thesis evidence."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, default=str)},
        ],
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    text = _extract_text(data)
    return json.loads(text)


def _extract_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return data["output_text"]
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    if not chunks:
        raise ValueError("OpenAI response did not contain text output")
    return "".join(chunks)


def _fallback_payload(row: pd.Series, news_items: list[dict[str, Any]]) -> dict[str, Any]:
    recent_news = _news_titles(news_items)
    return {
        "ticker": row["ticker"],
        "business_summary": row["business_summary"],
        "moat": row["moat_analysis"],
        "catalysts": row["catalyst"],
        "risks": row["risk_summary"],
        "recent_news": recent_news,
        "thesis": row["investment_thesis"],
        "exit_thesis": row["exit_thesis"],
        "classification": row["opportunity_type"],
        "source": "openai" if ENABLE_OPENAI_RESEARCH and OPENAI_API_KEY else "deterministic_fallback",
    }


def _load_news() -> dict[str, list[dict[str, Any]]]:
    path = RAW_DIR / "news.parquet"
    if not path.exists():
        return {}
    news = pd.read_parquet(path)
    rows: dict[str, list[dict[str, Any]]] = {}
    for ticker, group in news.groupby("ticker"):
        rows[ticker] = group.sort_values("datetime", ascending=False).head(3).to_dict("records")
    return rows


def _news_titles(news_items: list[dict[str, Any]]) -> list[str]:
    titles = [str(item.get("headline") or item.get("summary") or "News item unavailable") for item in news_items[:3]]
    while len(titles) < 3:
        titles.append("No recent structured news item available in the data lake.")
    return titles
