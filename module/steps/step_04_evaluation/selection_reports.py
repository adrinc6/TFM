"""Selection audit and explanation exports for evaluation folds."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from module.steps.step_04_evaluation.explainability import AgentExplainer

log = logging.getLogger(__name__)


def _score_label(score: float) -> str:
    return "Outperform" if score >= 0.5 else "Underperform"


def _normalise_ticker_list(tickers: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(tk) for tk in tickers if pd.notna(tk)))


def _rule_based_drivers(row: pd.Series, score: float) -> List[Dict]:
    checks = [
        ("roe", "> 0.15", lambda v: v > 0.15, "ROE solido"),
        ("net_margin", "> 0.10", lambda v: v > 0.10, "Margen neto positivo"),
        ("debt_to_ebitda", "> 6", lambda v: v > 6.0, "Deuda elevada vs EBITDA"),
        ("current_ratio", "< 1", lambda v: v < 1.0, "Riesgo de liquidez"),
        ("revenue_yoy_growth", "> 0", lambda v: v > 0, "Crecimiento de ingresos"),
        ("fcf", "< 0", lambda v: v < 0, "FCF negativo"),
        ("momentum_12m", "> 0", lambda v: v > 0, "Momentum anual positivo"),
        ("rsi_14", "> 70", lambda v: v > 70, "RSI sobrecomprado"),
        ("eps_surprise_pct", "> 0", lambda v: v > 0, "Sorpresa de EPS positiva"),
        ("beat_rate_4q", ">= 0.75", lambda v: v >= 0.75, "Beat rate consistente"),
        ("mspr_3m", "> 0", lambda v: v > 0, "MSPR positivo"),
        ("insider_net_ratio_90d", "> 0", lambda v: v > 0, "Balance neto insider positivo"),
    ]

    drivers: List[Dict] = []
    for feat, _, fn, desc in checks:
        val = row.get(feat, pd.NA)
        if pd.isna(val):
            continue
        try:
            if fn(float(val)):
                drivers.append({
                    "feature": feat,
                    "description": desc,
                    "shap_value": float(0.0),
                    "raw_value": float(val),
                    "direction": "positive" if score >= 0.5 else "negative",
                })
        except Exception:
            log.debug("Rule-based driver check failed for %s", feat, exc_info=True)
            continue

    if not drivers:
        numeric = row.select_dtypes(include="number") if hasattr(row, "select_dtypes") else pd.Series(dtype=float)
        for feat, val in numeric.abs().sort_values(ascending=False).head(3).items():
            drivers.append({
                "feature": feat,
                "description": feat.replace("_", " ").title(),
                "shap_value": float(0.0),
                "raw_value": float(row.get(feat, 0.0)),
                "direction": "positive" if score >= 0.5 else "negative",
            })

    return drivers


def _fallback_explanation(agent_name: str, row: pd.Series, ticker: str, score: float) -> tuple[str, List[Dict]]:
    text = AgentExplainer._rule_based_text(row, score, ticker)
    drivers = _rule_based_drivers(row, score)
    if not drivers:
        drivers = [{
            "feature": f"{agent_name}_score",
            "description": f"Score del agente {agent_name}",
            "shap_value": 0.0,
            "raw_value": float(score),
            "direction": "positive" if score >= 0.5 else "negative",
        }]
    return text, drivers


def _format_driver_list(drivers: List[Dict]) -> str:
    parts: List[str] = []
    for driver in drivers:
        label = driver.get("description") or driver.get("feature") or ""
        raw_value = driver.get("raw_value")
        if pd.isna(raw_value):
            value_text = "N/A"
        else:
            try:
                value_text = f"{float(raw_value):.3f}"
            except Exception:
                value_text = str(raw_value)
        parts.append(f"{label}={value_text}")
    return ", ".join(parts)


def _flatten_text(text: str) -> str:
    return " ".join(str(text).split())


def _split_driver_groups(drivers: List[Dict]) -> tuple[str, str]:
    positives = [d for d in drivers if float(d.get("shap_value", 0.0)) >= 0]
    negatives = [d for d in drivers if float(d.get("shap_value", 0.0)) < 0]
    return _format_driver_list(positives), _format_driver_list(negatives)


def build_selection_audit_df(
    df_scored: pd.DataFrame,
    selected_tickers: Sequence[str],
    score_col: str = "final_score",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a flat audit table with the reason each ticker was or was not selected."""
    if df_scored.empty:
        return pd.DataFrame()

    df = df_scored.copy()
    if "ticker" not in df.columns:
        df = df.reset_index()

    if score_col not in df.columns:
        raise KeyError(f"Score column '{score_col}' not found in scored dataframe")

    if "ticker" in df.columns and df.duplicated(subset="ticker").any():
        # Keep the latest occurrence per ticker to avoid ambiguous audit rows.
        df = df.drop_duplicates(subset="ticker", keep="last")

    selected_list = _normalise_ticker_list(selected_tickers)
    selected_set = set(selected_list)
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["selected"] = df["ticker"].isin(selected_set)
    df["common_score"] = df[score_col].astype(float)
    df["common_label"] = df["common_score"].map(_score_label)
    df["score_gap_vs_threshold"] = df["common_score"] - threshold

    selected_scores = df.loc[df["selected"], ["ticker", "common_score"]].copy()
    selected_cutoff = float(selected_scores["common_score"].min()) if not selected_scores.empty else float("nan")
    df["score_gap_vs_selected_cutoff"] = df["common_score"] - selected_cutoff
    df["selected_cutoff_score"] = selected_cutoff

    selected_rank_map = {tk: idx + 1 for idx, tk in enumerate(selected_list)}
    df["selected_rank"] = df["ticker"].map(selected_rank_map)

    def _reason(row: pd.Series) -> str:
        if row["selected"]:
            if row["common_score"] >= threshold:
                return "selected_above_threshold"
            return "selected_by_fallback"
        if row["common_score"] >= threshold:
            return "qualified_but_not_selected"
        return "below_threshold"

    df["selection_reason"] = df.apply(_reason, axis=1)
    df["above_threshold"] = df["common_score"] >= threshold
    df["distance_to_threshold"] = df["common_score"] - threshold
    df["selected_position"] = df["rank"].where(df["selected"])

    return df


def build_explanation_candidate_tickers(
    audit_df: pd.DataFrame,
    threshold: float = 0.5,
    top_extra: int = 10,
    near_margin: float = 0.05,
    max_candidates: int = 30,
) -> List[str]:
    """Choose a compact but informative set of tickers to explain."""
    if audit_df.empty:
        return []

    ordered = audit_df.sort_values("common_score", ascending=False)
    candidates: List[str] = []

    if "selected" in ordered.columns:
        candidates.extend(ordered.loc[ordered["selected"], "ticker"].tolist())

    candidates.extend(ordered.head(top_extra)["ticker"].tolist())
    candidates.extend(ordered.tail(top_extra)["ticker"].tolist())

    near_mask = ordered["common_score"].sub(threshold).abs() <= near_margin
    candidates.extend(ordered.loc[near_mask, "ticker"].tolist())

    return _normalise_ticker_list(candidates)[:max_candidates]


def export_selection_audit(
    audit_df: pd.DataFrame,
    results_dir: str,
    fold_id: Optional[int | str] = None,
    prefix: str = "fold",
) -> tuple[Path, Path]:
    """Persist the selection audit as CSV and JSON."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "selection_audit.csv"
    json_path = out_dir / "selection_audit.json"

    if audit_df.empty:
        csv_path.write_text("", encoding="utf-8")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"rows": 0, "selected": 0}, f, indent=2, ensure_ascii=False)
        return csv_path, json_path

    audit_df.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "rows": int(len(audit_df)),
        "selected": int(audit_df["selected"].sum()) if "selected" in audit_df.columns else 0,
        "qualified_but_not_selected": int((audit_df["selection_reason"] == "qualified_but_not_selected").sum())
        if "selection_reason" in audit_df.columns else 0,
        "below_threshold": int((audit_df["selection_reason"] == "below_threshold").sum())
        if "selection_reason" in audit_df.columns else 0,
        "selected_by_fallback": int((audit_df["selection_reason"] == "selected_by_fallback").sum())
        if "selection_reason" in audit_df.columns else 0,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return csv_path, json_path


def export_ticker_explanations(
    agents: Dict,
    df_test: pd.DataFrame,
    scores: pd.Series,
    fold_id: int | str,
    agents_results_dir: str,
    candidate_tickers: Sequence[str],
    audit_df: Optional[pd.DataFrame] = None,
    explanation_top_n: int = 6,
    prefix: str = "fold",
) -> tuple[Path, Path]:
    """Generate flat per-agent explanation rows for a compact candidate universe."""
    if scores.empty:
        out_dir = Path(agents_results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ticker_explanations.csv"
        json_path = out_dir / "ticker_explanations.json"
        csv_path.write_text("", encoding="utf-8")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"fold": fold_id, "tickers": {}}, f, indent=2, ensure_ascii=False)
        return csv_path, json_path

    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ticker_explanations.csv"
    json_path = out_dir / "ticker_explanations.json"

    tickers_col = df_test.index.get_level_values("ticker")
    ticker_scores = pd.Series(scores.values, index=tickers_col).groupby(level=0).last()
    selected = _normalise_ticker_list(candidate_tickers)
    audit_lookup = {}
    if audit_df is not None and not audit_df.empty:
        audit_unique = audit_df
        if audit_unique.duplicated(subset="ticker").any():
            audit_unique = audit_unique.drop_duplicates(subset="ticker", keep="last")
        audit_lookup = audit_unique.set_index("ticker").to_dict(orient="index")

    all_explanations = {"fold": fold_id, "tickers": {}}
    flat_rows: List[Dict] = []

    for ticker in selected:
        mask = tickers_col == ticker
        if not mask.any():
            continue
        row = df_test.loc[mask].iloc[-1]
        common_score = float(ticker_scores.get(ticker, 0.5))
        common_label = _score_label(common_score)
        audit_row = audit_lookup.get(ticker, {})

        ticker_exp = {
            "common_score": round(common_score, 4),
            "common_label": common_label,
            "selection_reason": audit_row.get("selection_reason", "unknown"),
            "selected": bool(audit_row.get("selected", False)),
            "agents": {},
        }

        for ag_name, ag in agents.items():
            explainer = getattr(ag, "_explainer", None)
            agent_score = common_score
            # sector_rotation stores its score in "sector_score", not "sector_rotation_score"
            if ag_name == "sector_rotation":
                agent_score_col = "sector_score"
            else:
                agent_score_col = f"{ag_name}_score"
            if agent_score_col in df_test.columns:
                try:
                    agent_score = float(row[agent_score_col])
                except Exception:
                    agent_score = common_score
            if ag_name == "sector_rotation" and (pd.isna(agent_score) or agent_score_col not in df_test.columns):
                audit_sector_score = audit_row.get("sector_score")
                if audit_sector_score is not None and not pd.isna(audit_sector_score):
                    agent_score = float(audit_sector_score)
            agent_label = _score_label(agent_score)

            flat_row = {
                "fold": fold_id,
                "ticker": ticker,
                "common_score": round(common_score, 4),
                "common_label": common_label,
                "selection_reason": audit_row.get("selection_reason", "unknown"),
                "selected": bool(audit_row.get("selected", False)),
                "rank": audit_row.get("rank"),
                "agent": ag_name,
                "agent_score": round(agent_score, 4),
                "agent_label": agent_label,
                "has_explainer": explainer is not None,
                "explanation_text": "",
                "favor_factors": "",
                "contra_factors": "",
                "top_drivers_json": "[]",
            }

            if explainer is None:
                fallback_text, fallback_drivers = _fallback_explanation(ag_name, row, ticker, agent_score)
                favor_text, contra_text = _split_driver_groups(fallback_drivers[:explanation_top_n])
                flat_row["explanation_text"] = _flatten_text(fallback_text)
                flat_row["favor_factors"] = favor_text
                flat_row["contra_factors"] = contra_text
                flat_row["top_drivers_json"] = json.dumps(fallback_drivers[:explanation_top_n], ensure_ascii=False, default=str)
                ticker_exp["agents"][ag_name] = {
                    "text": fallback_text,
                    "top_drivers": fallback_drivers[:explanation_top_n],
                }
                flat_rows.append(flat_row)
                continue

            try:
                exp = explainer.explain_prediction(row, ticker, agent_score, top_n=explanation_top_n, fold=fold_id)
                top_drivers = exp.get("top_drivers", [])[:explanation_top_n]
                favor_text, contra_text = _split_driver_groups(top_drivers)
                ticker_exp["agents"][ag_name] = {
                    "text": exp.get("text", ""),
                    "top_drivers": top_drivers,
                }
                flat_row["explanation_text"] = _flatten_text(exp.get("text", ""))
                flat_row["favor_factors"] = favor_text
                flat_row["contra_factors"] = contra_text
                flat_row["top_drivers_json"] = json.dumps(top_drivers, ensure_ascii=False, default=str)
                flat_rows.append(flat_row)
            except Exception as ex:
                log.debug(f"Explain {ag_name}/{ticker}: {ex}")
                fallback_text, fallback_drivers = _fallback_explanation(ag_name, row, ticker, agent_score)
                favor_text, contra_text = _split_driver_groups(fallback_drivers[:explanation_top_n])
                flat_row["explanation_text"] = _flatten_text(f"Error generating explanation: {ex}. {fallback_text}")
                flat_row["favor_factors"] = favor_text
                flat_row["contra_factors"] = contra_text
                flat_row["top_drivers_json"] = json.dumps(fallback_drivers[:explanation_top_n], ensure_ascii=False, default=str)
                ticker_exp["agents"][ag_name] = {
                    "text": fallback_text,
                    "top_drivers": fallback_drivers[:explanation_top_n],
                }
                flat_rows.append(flat_row)

        all_explanations["tickers"][ticker] = ticker_exp

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False, default=str)

    flat_df = pd.DataFrame(flat_rows)
    if not flat_df.empty:
        flat_df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        csv_path.write_text("", encoding="utf-8")

    log.info(
        f"[Explainer] Fold {fold_id}: explicaciones de {len(selected)} tickers "
        f"-> {csv_path.name} y {json_path.name}"
    )

    return csv_path, json_path