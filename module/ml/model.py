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
    labeled = _add_component_targets(features, prices, settings.benchmark_ticker)
    models, importance = _fit_component_models(labeled)
    for probability, model in models.items():
        labeled[probability] = _predict(model, labeled[MODEL_FEATURES])
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
    shap_summary = _shap_summary(models.get("alpha_probability"), labeled[MODEL_FEATURES])
    write_parquet(labeled, PROCESSED_DIR / "scored_universe.parquet")
    write_json(
        {
            "component_models": list(COMPONENT_TARGETS.keys()),
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
    return labeled


def _add_component_targets(df: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    df = df.copy()
    df["snapshot_date_dt"] = pd.to_datetime(df["snapshot_date"])
    alphas = []
    for row in df.itertuples(index=False):
        stock_ret = _forward_return(prices, row.ticker, row.snapshot_date_dt, months=12)
        bench_ret = _forward_return(prices, benchmark_ticker, row.snapshot_date_dt, months=12)
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
        0.45 * df["realized_growth"]
        + 0.20 * df["growth_score_vs_sector"]
        + 0.15 * df["catalyst_score"]
        + 0.10 * df["positive_expectation_gap"]
        + 0.10 * df["quality_trend_1y"].clip(lower=0)
    ).clip(0, 1)
    df["target_mispricing"] = (
        0.50 * df["positive_expectation_gap"]
        + 0.25 * df["valuation_score_vs_sector"]
        + 0.25 * df["valuation_score"]
    ).clip(0, 1)
    return df.drop(columns=["snapshot_date_dt"])


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
    catalyst = row["catalyst_score"]
    risk = row["risk_score"]

    if quality < 0.35 or (risk < 0.20 and quality < 0.50):
        return "Avoid"
    if valuation >= 0.80 and catalyst < 0.45 and quality < 0.55:
        return "Value Trap"
    if quality >= 0.60 and moat >= 0.55 and growth >= 0.55 and valuation >= 0.35:
        return "Quality Growth Reasonable"
    if quality >= 0.80 and moat >= 0.75 and growth >= 0.50:
        return "Compounder" if valuation >= 0.35 else "Fully Valued Compounder"
    if growth >= 0.75 and valuation >= 0.55:
        return "Growth Undervalued"
    if valuation >= 0.70 and catalyst >= 0.55 and quality >= 0.45:
        return "Value with Catalyst"
    if catalyst >= 0.70 and growth >= 0.45 and quality >= 0.40:
        return "Turnaround"
    if valuation >= 0.65 and risk >= 0.45:
        return "Cyclical Opportunity"
    if valuation >= 0.85:
        return "Deep Value"
    if growth >= 0.70 and valuation < 0.35:
        return "Expensive Growth"
    if quality >= 0.50 and growth >= 0.35 and valuation >= 0.45:
        return "Fully Valued Compounder" if valuation < 0.55 else "Quality Growth Reasonable"
    return "Avoid"
