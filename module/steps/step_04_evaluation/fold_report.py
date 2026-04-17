"""
fold_report.py — Generación del CSV consolidado de resultados por fold.

Produce dos archivos por cada fold completado, y al final un consolidado global:

1. results/agents/fold_{N}_scores.csv
   Una fila por ticker. Columnas:
     - Identificación: year_quarter, fold, ticker, sector, industry
     - Selección: selected, rank, selection_reason
     - Scores de cada agente (0-1) con interpretación en texto
     - Score final y predicción
     - Realized result (if available): actual_return, beat_benchmark
     - Explicación de cada agente: qué factores jugaron a favor y en contra,
       con los valores reales de las métricas en lenguaje natural

2. results/agents/all_folds_scores.csv
   Concatenación de todos los fold_{N}_scores.csv. Permite filtrar por quarter,
   comparar tickers a lo largo del tiempo, y auditar decisiones por agente.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module.steps.step_04_evaluation.explainability import (
    FEATURE_DESCRIPTIONS,
    _format_value,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Umbrales para interpretar cada agente en texto
# ---------------------------------------------------------------------------

AGENT_LABELS = {
    "fundamental": {
        "high":   "Salud financiera sólida",
        "medium": "Salud financiera aceptable",
        "low":    "Debilidades financieras detectadas",
    },
    "valuation": {
        "high":   "Acción atractiva por valoración",
        "medium": "Valoración razonable",
        "low":    "Posiblemente sobrevalorada",
    },
    "momentum": {
        "high":   "Tendencia técnica positiva",
        "medium": "Tendencia técnica neutra",
        "low":    "Tendencia técnica negativa",
    },
    "bear": {
        "high":   "Riesgo bajo (perfil defensivo)",
        "medium": "Riesgo moderado",
        "low":    "Riesgo elevado detectado",
    },
    "sentiment": {
        "high":   "Sentimiento externo positivo",
        "medium": "Sentimiento externo neutro",
        "low":    "Sentimiento externo negativo",
    },
    "meta_learner": {
        "high":   "Alta probabilidad de Outperform",
        "medium": "Señal mixta",
        "low":    "Alta probabilidad de Underperform",
    },
}


def _is_ratio_or_normalized_feature(feature: str) -> bool:
    f = str(feature).lower().strip()
    if not f:
        return False
    allowed_tokens = [
        "ratio", "margin", "yield", "growth", "trend", "momentum", "volatility",
        "rsi", "macd", "beta", "zscore", "zsector", "pct", "coverage",
        "score", "prior", "dispersion", "consensus", "confidence", "quality",
        "fscore", "accrual", "atr", "bb_", "vs_5y", "vs_52w", "debt_to_", "_to_",
        "revision", "surprise", "beater", "overbought", "oversold", "bullish",
        "above_sma", "cross_sma", "expansion", "decline", "losses", "risk",
    ]
    if any(tok in f for tok in allowed_tokens):
        return True
    blocked_prefixes = [
        "revenue", "net_income", "operating_income", "gross_profit", "fcf", "ebitda",
        "total_assets", "total_liabilities", "total_equity", "total_debt", "cash",
        "shares", "eps_est", "eps_reported", "market_cap", "capex", "income_tax",
        "depreciation", "operating_cash_flow",
    ]
    if any(f.startswith(p) for p in blocked_prefixes):
        return False
    return False


def _agent_text_label(agent: str, score: float) -> str:
    """Convierte el score de un agente en etiqueta de texto con su interpretación."""
    labels = AGENT_LABELS.get(agent, {
        "high": "Score alto", "medium": "Score medio", "low": "Score bajo"
    })
    if score >= 0.65:
        tier = "high"
    elif score >= 0.40:
        tier = "medium"
    else:
        tier = "low"
    return f"{score:.2f} — {labels[tier]}"


# ---------------------------------------------------------------------------
# Features clave por agente (para la explicación en texto)
# Los que más aportan contexto sin necesidad de SHAP
# ---------------------------------------------------------------------------

AGENT_KEY_FEATURES: Dict[str, List[str]] = {
    "fundamental": [
        "roe", "net_margin", "revenue_yoy_growth", "debt_to_ebitda",
        "interest_coverage", "fcf_margin", "piotroski_fscore",
        "earnings_quality", "consecutive_losses",
    ],
    "valuation": [
        "pe_ratio", "pb_ratio", "ev_to_ebitda", "fcf_yield",
        "pe_vs_5y_median", "ev_ebitda_vs_5y_median",
    ],
    "momentum": [
        "momentum_3m", "momentum_6m", "momentum_12m",
        "rsi_14", "sma_200", "volatility_60d",
    ],
    "bear": [
        "debt_to_ebitda", "consecutive_losses", "insider_sell_ratio",
        "current_ratio", "revenue_yoy_growth",
    ],
    "sentiment": [
        "analyst_buy_ratio", "analyst_consensus", "eps_surprise_pct",
        "beat_rate_4q", "mspr_3m", "insider_net_ratio_90d",
    ],
}


def _describe_feature_value(feature: str, value: float) -> str:
    """
    Genera una frase corta en lenguaje natural para un feature y su valor.
    Ej: roe=0.28 → "ROE del 28% (rentabilidad sobre fondos propios)"
    """
    if pd.isna(value):
        return ""
    desc = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())
    formatted = _format_value(feature, value)
    return f"{desc} = {formatted}"


def _build_agent_explanation(
    row: pd.Series,
    agent: str,
    agent_score: float,
    shap_drivers: Optional[List[Dict]] = None,
    top_n: int = 4,
) -> str:
    """
    Construye una explicación en lenguaje natural para un agente concreto.

    Prioridad:
    1. Si hay SHAP drivers (de explainability.py), usa los top positivos/negativos.
    2. Si no, usa las features clave del agente con sus valores reales.

    La explicación menciona los valores concretos (ej. "ROE del 28%") para que
    sea interpretable sin necesidad de conocer los números del dataset.
    """
    label = "Outperform" if agent_score >= 0.5 else "Underperform"
    score_txt = _agent_text_label(agent, agent_score)

    # -- Camino 1: SHAP disponible --
    if shap_drivers:
        positives = [d for d in shap_drivers if d.get("shap_value", 0) > 0][:top_n]
        negatives = [d for d in shap_drivers if d.get("shap_value", 0) < 0][:top_n]

        parts = [f"[{label}] {score_txt}"]

        if positives:
            favor = "; ".join(
                f"{d.get('description', d['feature'])} = "
                f"{_format_value(d['feature'], d.get('raw_value', np.nan))}"
                f" (+{d['shap_value']:.3f})"
                for d in positives
            )
            parts.append(f"In favor: {favor}")

        if negatives:
            contra = "; ".join(
                f"{d.get('description', d['feature'])} = "
                f"{_format_value(d['feature'], d.get('raw_value', np.nan))}"
                f" ({d['shap_value']:.3f})"
                for d in negatives
            )
            parts.append(f"Against: {contra}")

        return " | ".join(parts)

    # -- Camino 2: fallback con features clave del agente --
    key_features = AGENT_KEY_FEATURES.get(agent, [])
    observations: List[str] = []

    for feat in key_features:
        val = row.get(feat, np.nan)
        if pd.isna(val):
            continue
        phrase = _describe_feature_value(feat, float(val))
        if phrase:
            observations.append(phrase)

    if not observations:
        return f"[{label}] {score_txt}"

    obs_text = "; ".join(observations[:top_n])
    return f"[{label}] {score_txt} | Factores observados: {obs_text}"


# ---------------------------------------------------------------------------
# Construcción del DataFrame de scores por fold
# ---------------------------------------------------------------------------

def build_fold_scores_df(
    df_test_scored: pd.DataFrame,
    y_test: pd.Series,
    fold_id: int,
    year_quarter: str,
    agents: Dict,
    audit_df: Optional[pd.DataFrame] = None,
    actual_returns: Optional[Dict[str, float]] = None,
    benchmark_return: Optional[float] = None,
    ticker_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame de scores para un fold con:
    - Una fila por ticker
    - Scores de cada agente + interpretación en texto
    - Explicación de cada agente con los factores que influyeron
    - Resultado real si disponible

    Args:
        df_test_scored:  DataFrame de test con scores de agentes y final_score.
                         Índice: (ticker, date).
        y_test:          Labels reales del fold.
        fold_id:         Número de fold.
        year_quarter:    Ej. "2025Q3".
        agents:          Dict de agentes entrenados {nombre: agente}.
        audit_df:        DataFrame de auditoría de selección (optional).
        actual_returns:  {ticker: retorno_real} si ya ocurrió el período de test.
        benchmark_return: Retorno del S&P500 en el período de test.
    """
    tickers_index = df_test_scored.index.get_level_values("ticker")
    # Una fila por ticker: la última observación disponible en el fold
    ticker_rows: Dict[str, pd.Series] = {}
    for ticker in tickers_index.unique():
        mask = tickers_index == ticker
        ticker_rows[ticker] = df_test_scored.loc[mask].iloc[-1]

    # Mapa de auditoría
    audit_map: Dict[str, Dict] = {}
    if audit_df is not None and not audit_df.empty and "ticker" in audit_df.columns:
        audit_map = audit_df.drop_duplicates(subset="ticker").set_index("ticker").to_dict(orient="index")

    records: List[Dict] = []

    for ticker, row in ticker_rows.items():
        final_score = float(row.get("final_score", 0.5))
        audit = audit_map.get(ticker, {})

        rec: Dict = {
            # --- Identificación ---
            "year_quarter":     year_quarter,
            "fold":             fold_id,
            "ticker":           ticker,
            "sector":           row.get("sector", "Unknown"),
            "industry":         row.get("industry", "Unknown"),

            # --- Predicción ---
            "final_score":      round(final_score, 4),
            "final_score_raw":  round(float(row.get("final_score_raw", final_score)), 4),
            "prediccion":       "Outperform" if final_score >= 0.5 else "Underperform",
            "confianza":        "Alta" if abs(final_score - 0.5) > 0.25 else "Moderada",

            # --- Selección y peso en cartera ---
            "selected":         bool(audit.get("selected", False)),
            "rank":             audit.get("rank", None),
            "selection_reason": audit.get("selection_reason", "unknown"),
            "portfolio_weight": round(ticker_weights[ticker], 4) if ticker_weights and ticker in ticker_weights else None,
            "sector_peer_count": int(row.get("sector_peer_count", 0)) if pd.notna(row.get("sector_peer_count", None)) else None,
            "sector_confidence": round(float(row.get("sector_confidence", 1.0)), 4),
        }

        # --- Scores by agent + texto interpretativo ---
        for ag_name in ["fundamental", "valuation", "momentum", "bear", "sentiment"]:
            score_col = f"{ag_name}_score"
            ag_score = float(row.get(score_col, 0.5))
            rec[score_col] = round(ag_score, 4)
            rec[f"{ag_name}_interpretacion"] = _agent_text_label(ag_name, ag_score)

        # --- Explicación por agente con valores concretos ---
        for ag_name, agent in agents.items():
            if ag_name == "meta_learner":
                continue
            if ag_name == "sector_rotation":
                ag_score = float(row.get("sector_score", 0.5))
            else:
                ag_score = float(row.get(f"{ag_name}_score", 0.5))

            # Intentar obtener SHAP drivers del explainer del agente
            shap_drivers = None
            explainer = getattr(agent, "_explainer", None)
            if explainer is not None:
                try:
                    exp = explainer.explain_prediction(row, ticker, ag_score, top_n=5)
                    shap_drivers = exp.get("top_drivers", [])
                except Exception:
                    log.debug("SHAP explain falló para %s/%s", ag_name, ticker, exc_info=True)

            rec[f"{ag_name}_explicacion"] = _build_agent_explanation(
                row=row,
                agent=ag_name,
                agent_score=ag_score,
                shap_drivers=shap_drivers,
                top_n=4,
            )

        # --- Resultado real (si ya ocurrió el período) ---
        if actual_returns is not None:
            actual = actual_returns.get(ticker)
            rec["retorno_real"] = round(actual, 4) if actual is not None else None
            if actual is not None and benchmark_return is not None:
                rec["alpha_real"] = round(actual - benchmark_return, 4)
                rec["beat_benchmark"] = actual > benchmark_return
            else:
                rec["alpha_real"] = None
                rec["beat_benchmark"] = None
        else:
            rec["retorno_real"] = None
            rec["alpha_real"] = None
            rec["beat_benchmark"] = None

        # --- Label real del modelo ---
        label_idx = (ticker, row.name[1]) if isinstance(row.name, tuple) else None
        if label_idx is not None and label_idx in y_test.index:
            rec["label_real"] = int(y_test.loc[label_idx])
        else:
            rec["label_real"] = None

        records.append(rec)

    df = pd.DataFrame(records)

    # Ordenar: primero seleccionados, luego por score descendente
    df = df.sort_values(
        ["selected", "final_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Export: fold individual + acumulado global
# ---------------------------------------------------------------------------

def export_fold_scores(
    df: pd.DataFrame,
    agents_results_dir: str,
    fold_id: int | str,
) -> Path:
    """Guarda el CSV del fold y devuelve la ruta."""
    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quarter_{fold_id}_scores.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Fold {fold_id}: scores de {len(df)} tickers guardados → {path.name}")
    return path


def export_quarter_snapshot_audit(
    df_test_scored: pd.DataFrame,
    year_quarter: str,
    agents_results_dir: str,
) -> Path:
    """
    Exporta un snapshot completo por ticker para el quarter:
    - una fila por ticker
    - todas las features disponibles
    - metadatos de snapshot/reporting (carry-forward, fechas usadas)
    - scores de agentes y score final
    """
    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quarter_{year_quarter}_ticker_snapshot_audit.csv"

    if df_test_scored.empty:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8")
        return path

    tickers = df_test_scored.index.get_level_values("ticker")
    rows: List[pd.Series] = []
    for ticker in tickers.unique():
        mask = tickers == ticker
        rows.append(df_test_scored.loc[mask].iloc[-1])

    out_df = pd.DataFrame(rows).reset_index(drop=True)
    if "ticker" in out_df.columns:
        out_df["ticker"] = out_df["ticker"].astype(str)
    else:
        out_df.insert(0, "ticker", list(tickers.unique()))

    if "year_quarter" in out_df.columns:
        out_df["year_quarter"] = year_quarter
    else:
        out_df.insert(0, "year_quarter", year_quarter)
    out_df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Snapshot audit {year_quarter}: {len(out_df)} tickers → {path.name}")
    return path


def export_quarter_agent_feature_audit(
    df_test_scored: pd.DataFrame,
    agents: Dict,
    year_quarter: str,
    agents_results_dir: str,
) -> Path:
    """
    Exporta trazabilidad granular agente-feature:
    una fila por (ticker, agente, feature usada por ese agente).
    """
    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quarter_{year_quarter}_ticker_agent_feature_audit.csv"

    if df_test_scored.empty:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8")
        return path

    tickers = df_test_scored.index.get_level_values("ticker")
    rows: List[Dict] = []

    agent_order = ["fundamental", "valuation", "momentum", "bear", "sentiment", "sector_rotation", "meta_learner"]
    for ticker in tickers.unique():
        mask = tickers == ticker
        row = df_test_scored.loc[mask].iloc[-1]
        final_score = float(row.get("final_score", np.nan)) if pd.notna(row.get("final_score", np.nan)) else np.nan

        for agent_name in agent_order:
            if agent_name not in agents:
                continue
            if agent_name == "sector_rotation":
                score_col = "sector_score"
            elif agent_name == "meta_learner":
                score_col = "final_score"
            else:
                score_col = f"{agent_name}_score"

            agent_score = row.get(score_col, np.nan)
            try:
                agent_score = float(agent_score)
            except Exception:
                agent_score = np.nan

            agent_obj = agents.get(agent_name)
            feat_cols = getattr(agent_obj, "_feature_cols", None)
            if not feat_cols:
                feat_cols = []

            if not feat_cols:
                rows.append({
                    "year_quarter": year_quarter,
                    "ticker": ticker,
                    "agent": agent_name,
                    "agent_score": agent_score,
                    "final_score": final_score,
                    "feature": "__no_feature_list__",
                    "feature_value": np.nan,
                    "feature_present": False,
                })
                continue

            for feat in feat_cols:
                if not _is_ratio_or_normalized_feature(feat):
                    continue
                val = row.get(feat, np.nan)
                rows.append({
                    "year_quarter": year_quarter,
                    "ticker": ticker,
                    "agent": agent_name,
                    "agent_score": agent_score,
                    "final_score": final_score,
                    "feature": feat,
                    "feature_value": val,
                    "feature_present": bool(pd.notna(val)),
                })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Agent-feature audit {year_quarter}: {len(rows)} filas → {path.name}")
    return path


def export_all_folds_scores(
    agents_results_dir: str,
) -> Optional[Path]:
    """
    Concatena todos los fold_{N}_scores.csv en all_folds_scores.csv.
    Llamar al final del pipeline, después de todos los folds.
    """
    out_dir = Path(agents_results_dir)
    fold_files = sorted(out_dir.glob("quarter_*_scores.csv"))
    if not fold_files:
        log.warning("[FoldReport] No se encontraron quarter_*_scores.csv para consolidar.")
        return None

    dfs = []
    for f in fold_files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8"))
        except Exception as e:
            log.warning(f"[FoldReport] Error leyendo {f.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    path = out_dir / "all_folds_scores.csv"
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info(
        f"[FoldReport] CSV consolidado de todos los folds: "
        f"{len(combined)} filas | "
        f"{combined['year_quarter'].nunique()} quarters | "
        f"{combined['ticker'].nunique()} tickers únicos → {path.name}"
    )
    return path
