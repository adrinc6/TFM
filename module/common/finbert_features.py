"""FinBERT-based sentiment feature extraction with deterministic fallback."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np


BULLISH_WORDS = {"growth", "strong", "beat", "expansion", "upside", "improve", "confidence"}
BEARISH_WORDS = {"decline", "weak", "miss", "downside", "pressure", "risk", "uncertain"}
RISK_WORDS = {"risk", "uncertainty", "volatility", "exposure", "downturn", "liquidity"}


@lru_cache(maxsize=1)
def _finbert_pipeline():
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            top_k=None,
        )
    except Exception:
        return None


def _lexicon_scores(texts: List[str]) -> Dict[str, float]:
    if not texts:
        return {
            "finbert_sentiment_polarity": 0.0,
            "finbert_uncertainty_score": 0.0,
            "finbert_risk_intensity": 0.0,
            "finbert_bullish_tone": 0.0,
        }
    joined = " ".join(texts).lower()
    tokens = [t.strip(".,;:!?()[]{}\"'") for t in joined.split() if t.strip()]
    if not tokens:
        return {
            "finbert_sentiment_polarity": 0.0,
            "finbert_uncertainty_score": 0.0,
            "finbert_risk_intensity": 0.0,
            "finbert_bullish_tone": 0.0,
        }
    n = float(len(tokens))
    bull = sum(1 for t in tokens if t in BULLISH_WORDS)
    bear = sum(1 for t in tokens if t in BEARISH_WORDS)
    risk = sum(1 for t in tokens if t in RISK_WORDS)
    polarity = (bull - bear) / max(bull + bear, 1)
    return {
        "finbert_sentiment_polarity": float(np.clip(polarity, -1.0, 1.0)),
        "finbert_uncertainty_score": float(np.clip((bear + risk) / n, 0.0, 1.0)),
        "finbert_risk_intensity": float(np.clip(risk / n, 0.0, 1.0)),
        "finbert_bullish_tone": float(np.clip(bull / n, 0.0, 1.0)),
    }


def extract_finbert_features(texts: Iterable[str], max_docs: int = 8) -> Dict[str, float]:
    docs = [str(t).strip() for t in texts if t is not None and str(t).strip()]
    docs = docs[: max(int(max_docs), 1)]
    if not docs:
        return _lexicon_scores([])

    pipe = _finbert_pipeline()
    if pipe is None:
        return _lexicon_scores(docs)

    try:
        outs = pipe(docs)
    except Exception:
        return _lexicon_scores(docs)

    pos = []
    neg = []
    neu = []
    for item in outs:
        rows = item if isinstance(item, list) else [item]
        probs = {str(r.get("label", "")).lower(): float(r.get("score", 0.0)) for r in rows}
        pos.append(probs.get("positive", 0.0))
        neg.append(probs.get("negative", 0.0))
        neu.append(probs.get("neutral", 0.0))

    pos_m = float(np.mean(pos)) if pos else 0.0
    neg_m = float(np.mean(neg)) if neg else 0.0
    neu_m = float(np.mean(neu)) if neu else 0.0

    return {
        "finbert_sentiment_polarity": float(np.clip(pos_m - neg_m, -1.0, 1.0)),
        "finbert_uncertainty_score": float(np.clip(neu_m, 0.0, 1.0)),
        "finbert_risk_intensity": float(np.clip(neg_m, 0.0, 1.0)),
        "finbert_bullish_tone": float(np.clip(pos_m, 0.0, 1.0)),
    }


def extract_text_blobs(*dfs) -> List[str]:
    """Extract candidate text fields from heterogeneous dataframes."""
    texts: List[str] = []
    cols = ["headline", "summary", "text", "transcript", "statement", "discussion", "commentary"]
    for df in dfs:
        if df is None:
            continue
        if hasattr(df, "columns"):
            for c in cols:
                if c in df.columns:
                    vals = df[c].dropna().astype(str).tolist()
                    texts.extend(vals[-10:])
    return texts
