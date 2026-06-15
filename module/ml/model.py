"""Train, score and explain the GARP alpha model."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR, RAW_DIR, Settings
from module.common.io import read_parquet, write_json, write_parquet
from module.features.engineering import FEATURE_COLUMNS

log = logging.getLogger(__name__)


MODEL_FEATURES = [
    "quality_score",
    "growth_score",
    "valuation_score",
    "price_adjusted_valuation_score",
    "momentum_score",
    "moat_score",
    "catalyst_score",
    "risk_score",
    "expected_growth",
    "implied_growth",
    "realized_growth",
    "positive_expectation_gap",
    "quality_score_vs_sector",
    "quality_score_vs_industry",
    "quality_score_vs_universe",
    "growth_score_vs_sector",
    "growth_score_vs_industry",
    "growth_score_vs_universe",
    "valuation_score_vs_sector",
    "valuation_score_vs_industry",
    "valuation_score_vs_universe",
    "price_return_since_fundamental",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "stale_fundamental_months",
    "quality_trend_1y",
    "quality_trend_2y",
    "roic_trend",
    "margin_trend",
    "fcf_trend",
    "growth_acceleration",
    "growth_deceleration",
    "moat_trend",
]

COMPONENT_TARGETS = {
    "quality_probability": "target_quality",
    "improvement_probability": "target_improvement",
    "mispricing_probability": "target_mispricing",
    "alpha_probability": "target_future_alpha",
}


def train_and_score(settings: Settings) -> pd.DataFrame:
    features = read_parquet(PROCESSED_DIR / "features.parquet")
    prices = read_parquet(RAW_DIR / "prices.parquet")
    log.info(
        "Training/scoring universe rows=%s snapshots=%s tickers=%s features=%s",
        len(features),
        features["snapshot_date"].nunique(),
        features["ticker"].nunique(),
        len(MODEL_FEATURES),
    )
    labeled = _add_component_targets(features, prices, settings.benchmark_ticker, settings.walk_forward_label_horizon_months)
    if settings.walk_forward_scoring:
        labeled, importance, diagnostics = _walk_forward_component_scores(labeled, settings)
        settings.run_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(settings.run_dir / "model_walk_forward_diagnostics.csv", index=False)
    else:
        models, importance = _fit_component_models(labeled)
        for probability, model in models.items():
            labeled[probability] = _predict(model, labeled[MODEL_FEATURES])
        diagnostics = pd.DataFrame([{"mode": "full_sample", "training_rows": len(labeled)}])
    labeled["ml_score"] = (
        0.30 * labeled["quality_probability"]
        + 0.25 * labeled["improvement_probability"]
        + 0.25 * labeled["mispricing_probability"]
        + 0.20 * labeled["alpha_probability"]
    ).clip(0, 1)
    labeled["business_quality_score"] = (
        0.30 * labeled["quality_probability"]
        + 0.25 * labeled["quality_score_vs_sector"]
        + 0.20 * labeled["moat_score"]
        + 0.15 * labeled["growth_score_vs_sector"]
        + 0.10 * labeled["risk_score"]
    ).clip(0, 1)
    labeled["final_score"] = (
        0.30 * labeled["quality_probability"]
        + 0.25 * labeled["improvement_probability"]
        + 0.25 * labeled["mispricing_probability"]
        + 0.20 * labeled["alpha_probability"]
    ).clip(0, 1)
    labeled["opportunity_type"] = labeled.apply(_opportunity_type, axis=1)
    explain_model = _fit_component_models(labeled)[0].get("alpha_probability")
    shap_summary = _shap_summary(explain_model, labeled[MODEL_FEATURES])
    write_parquet(labeled, PROCESSED_DIR / "scored_universe.parquet")
    write_json(
        {
            "component_models": list(COMPONENT_TARGETS.keys()),
            "scoring_mode": "walk_forward" if settings.walk_forward_scoring else "full_sample",
            "walk_forward_label_horizon_months": settings.walk_forward_label_horizon_months,
            "min_walk_forward_training_rows": settings.min_walk_forward_training_rows,
            "min_walk_forward_training_years": settings.min_walk_forward_training_years,
            "max_walk_forward_training_years": settings.max_walk_forward_training_years,
            "walk_forward_training_policy": (
                "For each snapshot, train with rows from current_date minus max years through the "
                "current snapshot. Future alpha labels that would not be observable yet are replaced "
                "by the deterministic GARP fallback for the alpha component."
            ),
            "garp_score_formula": {
                "quality_probability": 0.30,
                "improvement_probability": 0.25,
                "mispricing_probability": 0.25,
                "alpha_probability": 0.20,
            },
            "feature_importance": importance,
            "shap": shap_summary,
        },
        PROCESSED_DIR / "model_explainability.json",
    )
    log.info(
        "Scored universe written rows=%s final_score_range=(%.3f, %.3f)",
        len(labeled),
        float(labeled["final_score"].min()),
        float(labeled["final_score"].max()),
    )
    return labeled


def _add_component_targets(df: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str, horizon_months: int) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    df = df.copy()
    df["snapshot_date_dt"] = pd.to_datetime(df["snapshot_date"])
    alphas = []
    for row in df.itertuples(index=False):
        stock_ret = _forward_return(prices, row.ticker, row.snapshot_date_dt, months=horizon_months)
        bench_ret = _forward_return(prices, benchmark_ticker, row.snapshot_date_dt, months=horizon_months)
        alphas.append(None if stock_ret is None or bench_ret is None else stock_ret - bench_ret)
    df["target_future_alpha"] = alphas
    trainable = df["target_future_alpha"].notna()
    if trainable.sum() < 5:
        log.warning("Few future labels available; model will lean on deterministic GARP score.")
        df["target_future_alpha"] = df["garp_score"]
    df["target_quality"] = (
        0.35 * df["quality_score"]
        + 0.25 * df["moat_score"]
        + 0.20 * df["quality_score_vs_sector"]
        + 0.20 * df["risk_score"]
    ).clip(0, 1)
    df["target_improvement"] = (
        0.35 * df["realized_growth"]
        + 0.20 * df["growth_score_vs_sector"]
        + 0.15 * df["momentum_score"]
        + 0.15 * df["catalyst_score"]
        + 0.10 * df["positive_expectation_gap"]
        + 0.05 * df["quality_trend_1y"].clip(lower=0)
    ).clip(0, 1)
    df["target_mispricing"] = (
        0.50 * df["positive_expectation_gap"]
        + 0.25 * df["valuation_score_vs_sector"]
        + 0.25 * df["price_adjusted_valuation_score"]
    ).clip(0, 1)
    return df.drop(columns=["snapshot_date_dt"])


def _walk_forward_component_scores(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    scored = df.copy()
    scored["snapshot_date_dt"] = pd.to_datetime(scored["snapshot_date"])
    for probability, target in COMPONENT_TARGETS.items():
        scored[probability] = _fallback_probability(scored, target)

    dates = sorted(scored["snapshot_date_dt"].dropna().unique())
    diagnostics = []
    latest_importance: dict = {probability: {feature: 0 for feature in MODEL_FEATURES} for probability in COMPONENT_TARGETS}
    label_offset = pd.DateOffset(months=settings.walk_forward_label_horizon_months)
    max_history = pd.DateOffset(years=settings.max_walk_forward_training_years)
    for date in dates:
        current_mask = scored["snapshot_date_dt"] == date
        train_start_cutoff = pd.Timestamp(date) - max_history
        train_mask = (
            (scored["snapshot_date_dt"] >= train_start_cutoff)
            & (scored["snapshot_date_dt"] <= pd.Timestamp(date))
        )
        train = scored[train_mask].copy()
        alpha_observable_mask = train["snapshot_date_dt"] + label_offset <= pd.Timestamp(date)
        alpha_label_rows = int((alpha_observable_mask & train["target_future_alpha"].notna()).sum())
        train.loc[~alpha_observable_mask, "target_future_alpha"] = pd.NA
        training_start = train["snapshot_date_dt"].min() if not train.empty else pd.NaT
        training_end = train["snapshot_date_dt"].max() if not train.empty else pd.NaT
        training_years = (
            (training_end - training_start).days / 365.25
            if pd.notna(training_start) and pd.notna(training_end)
            else 0.0
        )
        fallback_reasons = []
        if len(train) < settings.min_walk_forward_training_rows:
            fallback_reasons.append("not_enough_rows")
        if train.empty or train["ticker"].nunique() < 10:
            fallback_reasons.append("not_enough_tickers")
        if training_years < settings.min_walk_forward_training_years:
            fallback_reasons.append("not_enough_years")
        if fallback_reasons:
            diagnostics.append({
                "snapshot_date": pd.Timestamp(date).date().isoformat(),
                "mode": "fallback_garp",
                "fallback_reason": ";".join(fallback_reasons),
                "training_start": training_start.date().isoformat() if pd.notna(training_start) else "",
                "training_end": training_end.date().isoformat() if pd.notna(training_end) else "",
                "training_years": float(training_years),
                "training_rows": int(len(train)),
                "training_tickers": int(train["ticker"].nunique()) if not train.empty else 0,
                "alpha_label_observable_rows": alpha_label_rows,
                "alpha_label_fallback_rows": int(len(train) - alpha_label_rows),
                "uses_rows_through_current_date": True,
                "min_training_years": settings.min_walk_forward_training_years,
                "max_training_years": settings.max_walk_forward_training_years,
                "label_horizon_months": settings.walk_forward_label_horizon_months,
            })
            continue
        models, latest_importance = _fit_component_models(train)
        for probability, model in models.items():
            scored.loc[current_mask, probability] = _predict(model, scored.loc[current_mask, MODEL_FEATURES]).values
        diagnostics.append({
            "snapshot_date": pd.Timestamp(date).date().isoformat(),
            "mode": "walk_forward_model",
            "fallback_reason": "",
            "training_start": training_start.date().isoformat(),
            "training_end": training_end.date().isoformat(),
            "training_years": float(training_years),
            "training_rows": int(len(train)),
            "training_tickers": int(train["ticker"].nunique()),
            "alpha_label_observable_rows": alpha_label_rows,
            "alpha_label_fallback_rows": int(len(train) - alpha_label_rows),
            "uses_rows_through_current_date": True,
            "min_training_years": settings.min_walk_forward_training_years,
            "max_training_years": settings.max_walk_forward_training_years,
            "label_horizon_months": settings.walk_forward_label_horizon_months,
        })
    return scored.drop(columns=["snapshot_date_dt"]), latest_importance, pd.DataFrame(diagnostics)


def _fallback_probability(df: pd.DataFrame, target: str) -> pd.Series:
    if target in df.columns:
        return df[target].fillna(df["garp_score"]).clip(0, 1)
    return df["garp_score"].fillna(0.5).clip(0, 1)


def _forward_return(prices: pd.DataFrame, ticker: str, start: pd.Timestamp, months: int) -> float | None:
    series = prices[prices["ticker"] == ticker].sort_values("date")
    start_rows = series[series["date"] <= start]
    end_rows = series[series["date"] <= start + pd.DateOffset(months=months)]
    if start_rows.empty or end_rows.empty:
        return None
    start_price = float(start_rows.iloc[-1]["adj_close"])
    end_price = float(end_rows.iloc[-1]["adj_close"])
    if start_price <= 0:
        return None
    return end_price / start_price - 1


def _fit_component_models(df: pd.DataFrame):
    models = {}
    importances = {}
    for probability, target in COMPONENT_TARGETS.items():
        model, importance = _fit_model(df, target)
        models[probability] = model
        importances[probability] = importance
    return models, importances


def _fit_model(df: pd.DataFrame, target: str):
    x = df[MODEL_FEATURES].fillna(0.5)
    y = df[target].fillna(df["garp_score"])
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(n_estimators=80, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=1)
        model.fit(x, y)
        importance = dict(zip(MODEL_FEATURES, model.feature_importances_.tolist()))
        return model, importance
    except Exception as exc:
        log.warning("LightGBM unavailable or failed (%s). Falling back to sklearn.", exc)
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=1)
        model.fit(x, y)
        importance = dict(zip(MODEL_FEATURES, model.feature_importances_.tolist()))
        return model, importance


def _predict(model, x: pd.DataFrame) -> pd.Series:
    raw = pd.Series(model.predict(x.fillna(0.5)), index=x.index)
    if raw.max() == raw.min():
        return pd.Series(0.5, index=x.index)
    return (raw - raw.min()) / (raw.max() - raw.min())


def _shap_summary(model, x: pd.DataFrame) -> dict:
    try:
        import shap

        sample = x.fillna(0.5).head(200)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        values_df = pd.DataFrame(values, columns=MODEL_FEATURES)
        return {
            "available": True,
            "mean_abs_contribution": values_df.abs().mean().sort_values(ascending=False).to_dict(),
        }
    except Exception as exc:
        return {"available": False, "reason": f"SHAP unavailable or failed: {exc}"}


def _opportunity_type(row: pd.Series) -> str:
    quality = row["quality_score"]
    moat = row["moat_score"]
    growth = row["growth_score"]
    valuation = row["valuation_score"]
    adjusted_valuation = row.get("price_adjusted_valuation_score", valuation)
    catalyst = row["catalyst_score"]
    risk = row["risk_score"]

    momentum = row.get("momentum_score", 0.5)

    if quality < 0.35 or (risk < 0.20 and quality < 0.50):
        return "Avoid"
    if momentum < 0.18 and adjusted_valuation < 0.55:
        return "Avoid"
    if valuation >= 0.80 and catalyst < 0.45 and quality < 0.55:
        return "Value Trap"
    if quality >= 0.60 and moat >= 0.55 and growth >= 0.55 and adjusted_valuation >= 0.35:
        return "Quality Growth Reasonable"
    if quality >= 0.80 and moat >= 0.75 and growth >= 0.50:
        return "Compounder" if adjusted_valuation >= 0.35 else "Fully Valued Compounder"
    if growth >= 0.75 and adjusted_valuation >= 0.55:
        return "Growth Undervalued"
    if adjusted_valuation >= 0.70 and catalyst >= 0.55 and quality >= 0.45:
        return "Value with Catalyst"
    if catalyst >= 0.70 and growth >= 0.45 and quality >= 0.40:
        return "Turnaround"
    if adjusted_valuation >= 0.65 and risk >= 0.45:
        return "Cyclical Opportunity"
    if adjusted_valuation >= 0.85:
        return "Deep Value"
    if growth >= 0.70 and adjusted_valuation < 0.35:
        return "Expensive Growth"
    if quality >= 0.50 and growth >= 0.35 and adjusted_valuation >= 0.45:
        return "Fully Valued Compounder" if adjusted_valuation < 0.55 else "Quality Growth Reasonable"
    return "Avoid"
