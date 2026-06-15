# =============================================================================
# module/explainer.py — Prediction explainability (SHAP)
# =============================================================================
# For each prediction answers: why this score?
#
# Generates three levels of explanation:
#   1. Global  → which features matter most in the trained model
#   2. Local   → why this specific ticker received that score
#   3. Text    → human-readable natural-language summary
#
# Compatible with: XGBoost, GBM, Random Forest, Logistic Regression
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("[Explainer] SHAP no instalado. Instala con: pip install shap")


# ── Human-readable descriptions for each feature ─────────────────────────────
FEATURE_DESCRIPTIONS = {
    # Profitability
    "roe":                      "Return on Equity (ROE)",
    "roa":                      "Return on Assets (ROA)",
    "roi":                      "Return on Investment (ROI)",
    "roic":                     "Return on Invested Capital (ROIC)",
    "net_margin":               "Net margin (profit / revenue)",
    "gross_margin":             "Gross margin",
    "fcf_margin":               "Free cash flow margin",
    "ebitda_margin":            "EBITDA margin",
    "operating_margin":         "Operating margin",
    # Liquidity
    "current_ratio":            "Current ratio (current assets / current liabilities)",
    "quick_ratio":              "Quick ratio",
    # Solvency
    "debt_equity":              "Debt / equity ratio",
    "debt_to_ebitda":           "Net debt / EBITDA",
    "interest_coverage":        "Interest coverage (EBIT / interest expense)",
    # Growth
    "revenue_yoy_growth":       "YoY revenue growth",
    "net_income_yoy_growth":    "YoY net income growth",
    "eps_yoy_growth":           "YoY EPS growth",
    "fcf_yoy_growth":           "YoY FCF growth",
    "total_debt_yoy_growth":    "YoY total debt growth",
    # Quality / additional growth
    "accruals_ratio":               "Accruals ratio (accounting quality, lower=better)",
    "capex_to_revenue":             "Capital intensity (CapEx / revenue)",
    "consecutive_losses":           "Consecutive quarters with losses",
    "earnings_quality":             "Earnings quality: FCF / Net Income (>1 = real cash)",
    "piotroski_fscore":             "Piotroski F-score (0-1): 8 financial health signals",
    "operating_income_yoy_growth":  "YoY operating income growth",
    "roa_change_yoy":               "YoY improvement in ROA",
    "gross_margin_change_yoy":      "YoY improvement in gross margin",
    "current_ratio_change_yoy":     "YoY improvement in current ratio",
    # Trends (normalised slope of last 8 observations)
    "roe_trend_2y":                 "ROE trend 2Y (positive = improving)",
    "roe_trend_3y":                 "ROE trend 3Y",
    "net_margin_trend_2y":          "Net margin trend 2Y",
    "net_margin_trend_3y":          "Net margin trend 3Y",
    "gross_margin_trend_3y":        "Gross margin trend 3Y",
    # Valuation
    "pe_ratio":                 "Price / EPS (P/E ratio)",
    "pb_ratio":                 "Price / Book value (P/B ratio)",
    "ps_ratio":                 "Price / Sales (P/S ratio)",
    "ev_to_ebitda":             "EV / EBITDA",
    "fcf_yield":                "FCF yield (FCF / market cap)",
    "earnings_yield":           "Earnings yield (inverse of P/E)",
    "pe_vs_5y_median":          "Current P/E vs 5Y historical median (>0 = more expensive)",
    "pb_vs_5y_median":          "Current P/B vs 5Y historical median",
    "ev_ebitda_vs_5y_median":   "Current EV/EBITDA vs 5Y historical median",
    # Technical
    "rsi_14":                   "RSI 14 days (>70 overbought, <30 oversold)",
    "rsi_28":                   "RSI 28 days",
    "macd":                     "MACD",
    "macd_hist":                "MACD histogram (trend momentum)",
    "sma_20":                   "Distance to 20-day SMA (%)",
    "sma_50":                   "Distance to 50-day SMA (%)",
    "sma_200":                  "Distance to 200-day SMA (long-term trend)",
    "bb_pct":                   "Bollinger Band position (0=low, 1=high)",
    "momentum_1m":              "1-month momentum",
    "momentum_3m":              "3-month momentum",
    "momentum_6m":              "6-month momentum",
    "momentum_12m":             "12-month momentum",
    "price_vs_52w_high":        "Distance to 52-week high",
    "volatility_20d":           "Realised volatility 20 days (annualised)",
    "volatility_60d":           "Realised volatility 60 days (annualised)",
    "vol_ratio_20_50":          "Volume ratio 20d / 50d (>1 = expansion)",
    # Insiders / Sentiment
    "insider_net_ratio_90d":        "Normalised net insider ratio (90 days)",
    "insider_sell_ratio":           "Insider sell ratio (>0.7 = red flag)",
    "insider_net_zscore":           "Z-score of net insider buys vs sector",
    "analyst_buy_ratio":            "Analyst buy recommendation ratio",
    "analyst_bearish_score":        "Analyst bearish score (negative recommendations)",
    "analyst_consensus":            "Analyst consensus (1=Strong Buy, 5=Strong Sell)",
    "analyst_dispersion":           "Analyst dispersion (high = uncertainty)",
    "analyst_strong_buy_pct":       "% of analysts with Strong Buy recommendation",
    "analyst_consensus_change":     "Recent change in analyst consensus",
    "analyst_net_bullish":          "Net analyst bullish balance",
    "mspr_3m":                      "MSPR (institutional sentiment) 3-month",
    "mspr_trend":                   "MSPR trend (improving/deteriorating)",
    "mspr_positive":                "Positive MSPR signals",
    "mspr_negative":                "Negative MSPR signals",
    "eps_surprise_pct":             "EPS surprise vs estimate (%)",
    "eps_revision":                 "Recent analyst EPS estimate revision",
    "eps_est":                      "Analyst estimated EPS",
    "eps_reported":                 "Reported EPS",
    "beat_rate_4q":                 "% of quarters where EPS beat the estimate (4Q)",
    "eps_surprise_avg_4q":          "Mean EPS surprise over last 4 quarters",
    "revenue_decline":              "FLAG: YoY revenue decline",
    # Additional technical
    "atr_14":                       "Average True Range 14 days (price volatility)",
    "price_vs_52w_low":             "Distance to 52-week low",
    "macd_signal":                  "MACD signal (EMA of MACD)",
    # Bear flags
    "debt_growth_high":         "FLAG: Debt growing >20% YoY",
    "fcf_negative":             "FLAG: Negative FCF",
    "liquidity_risk":           "FLAG: Current ratio < 1 (liquidity risk)",
    "low_coverage":             "FLAG: Interest coverage < 1.5x",
    "insider_selling":          "FLAG: Insiders selling >70% of transactions",
    # Agent scores (meta-learner)
    "quality_score":            "Quality Agent score (business quality and profitability)",
    "growth_score":             "Growth Agent score (durable growth probability)",
    "valuation_score":          "Valuation Agent score (margin of safety)",
    "fundamental_trend_score":  "Fundamental Trend Agent score (improving fundamentals)",
    "catalyst_score":           "Catalyst Agent score (re-rating probability)",
    "risk_bear_score":          "Risk/Bear Agent safety score (>0.5 is safer)",
    "technical_guardrail_score": "Technical Guardrail score (risk/timing only)",
    "sentiment_score":          "Sentiment Agent score (analysts and insiders)",
    "sector_rotation_score":    "Sector Rotation Agent score (favourable sector)",
    "mom_x_safety":             "Momentum × (1-Bear): momentum with risk filter",
}


def _describe(feature: str) -> str:
    """Returns a human-readable description of a feature, or the name if not mapped."""
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())


# =============================================================================
# AgentExplainer
# =============================================================================

class AgentExplainer:
    """
    Genera explicaciones SHAP para cualquier agente del sistema.

    Uso:
        explainer = AgentExplainer(agent, feature_cols, results_dir)
        explainer.fit_explainer(X_train)

        # Para un ticker concreto:
        explanation = explainer.explain_prediction(X_row, ticker="AAPL", score=0.73)
        print(explanation["text"])
    """

    def __init__(
        self,
        agent_name:   str,
        feature_cols: List[str],
        results_dir:  str,
        model_type:   str = "tree",   # "tree" | "linear" | "kernel"
    ):
        self.agent_name   = agent_name
        self.feature_cols = feature_cols
        base_dir = Path(results_dir)
        # Keep SHAP files directly in sector folders (..../sectors/<sector>/),
        # and avoid duplicating agent folder names (.../<agent>/<agent>/).
        if base_dir.name.lower() == str(agent_name).lower() or base_dir.parent.name.lower() == "sectors":
            self.results_dir = base_dir
        else:
            self.results_dir = base_dir / agent_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_type   = model_type
        self._explainer   = None
        self._shap_values_train: Optional[np.ndarray] = None
        self._X_train_summary:   Optional[pd.DataFrame] = None

    # ── Explainer construction ────────────────────────────────────────────────

    def fit_explainer(self, model, X_train: pd.DataFrame, max_background: int = 100):
        """
        Build the SHAP explainer on the trained model.

        model:          sklearn/xgb object already fitted
        X_train:        training data (used as SHAP background)
        max_background: rows for the background (more = more accurate, slower)
        """
        if not SHAP_AVAILABLE:
            log.warning(f"[{self.agent_name}] SHAP not available — no explanations")
            return self

        X = X_train[self.feature_cols].copy().fillna(0)

        try:
            if self.model_type == "tree":
                # TreeExplainer: fast and exact for XGB/RF/GBM
                self._explainer = shap.TreeExplainer(model)
            elif self.model_type == "linear":
                # LinearExplainer for Logistic Regression
                self._explainer = shap.LinearExplainer(model, X)
            else:
                # KernelExplainer: universal but slow — use small sample
                background = shap.sample(X, min(50, len(X)))
                self._explainer = shap.KernelExplainer(model.predict_proba, background)

            # Compute SHAP values on a train sample for global analysis
            background_data = shap.sample(X, min(max_background, len(X)))
            self._shap_values_train = self._explainer.shap_values(background_data)
            self._X_train_summary   = background_data

            # Normalise to 2D (n_samples, n_features) — positive class (index 1)
            sv = self._shap_values_train
            if isinstance(sv, list):
                # Old SHAP: list [class_0, class_1]
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                # New SHAP with RF/GBM: 3D array (n_samples, n_features, n_classes)
                sv = sv[:, :, 1]
            self._shap_values_train = sv

            log.info(f"[{self.agent_name}] SHAP explainer ready "
                     f"({type(self._explainer).__name__}, {len(background_data)} background)")

        except Exception as e:
            log.warning(f"[{self.agent_name}] Error building SHAP explainer: {e}")

        return self

    # ── Global explanation ────────────────────────────────────────────────────

    def global_importance(self) -> Optional[pd.Series]:
        """
        Global importance: mean(|SHAP|) per feature over the training set.
        More robust than model feature_importances_ because it has real units.
        """
        if self._shap_values_train is None:
            return None
        sv = np.array(self._shap_values_train)
        # Normalise to 2D (n_samples, n_features) regardless of SHAP format
        if sv.ndim == 3:
            sv = sv[:, :, 1]       # take positive class
        elif sv.ndim == 1:
            sv = sv.reshape(1, -1)
        mean_abs = np.abs(sv).mean(axis=0)   # always (n_features,)
        return pd.Series(mean_abs, index=self.feature_cols).sort_values(ascending=False)

    def save_global_explanation(self, fold: Optional[int] = None):
        """
        Save global SHAP importance to disk:
          - CSV with all features sorted by importance
          - JSON with top 20 features and human-readable descriptions
          - PNG bar chart of SHAP importance
        """
        imp = self.global_importance()
        if imp is None:
            return
        suffix = f"_{fold}" if fold is not None else ""

        # Bar chart of top 15 features by SHAP importance
        self._save_shap_bar_plot(imp.head(15), suffix)

    # ── Local explanation (per ticker) ────────────────────────────────────────

    def explain_prediction(
        self,
        X_row:  pd.Series,
        ticker: str,
        score:  float,
        top_n:  int = 8,
        fold:   Optional[int] = None,
    ) -> Dict:
        """
        Explain why the agent gave that score to a specific ticker.

        Returns:
            {
              "ticker":        "AAPL",
              "score":         0.73,
              "label":         "Outperform",
              "top_drivers":   [{"feature": "roe", "shap": 0.15, "value": 0.28, ...}, ...],
              "risk_drivers":  [...],   # features that penalise the score
              "text":          "AAPL receives score 0.73 (Outperform) mainly because..."
            }
        """
        result = {
            "ticker": ticker,
            "score":  round(score, 4),
            "label":  "Outperform" if score >= 0.5 else "Underperform",
            "agent":  self.agent_name,
        }

        if not SHAP_AVAILABLE or self._explainer is None:
            result["text"] = self._rule_based_text(X_row, score, ticker)
            return result

        try:
            x = X_row.reindex(self.feature_cols).fillna(0).values.reshape(1, -1)
            sv = self._explainer.shap_values(x)
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            sv = np.array(sv).flatten()

            # Build contribution table
            drivers = []
            for feat, shap_val, raw_val in zip(self.feature_cols, sv, x.flatten()):
                drivers.append({
                    "feature":     feat,
                    "description": _describe(feat),
                    "shap_value":  float(shap_val),
                    "raw_value":   float(raw_val),
                    "direction":   "positive" if shap_val > 0 else "negative",
                })

            drivers.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
            top_all    = drivers[:top_n]
            top_pos    = [d for d in top_all if d["shap_value"] > 0]
            top_neg    = [d for d in top_all if d["shap_value"] < 0]

            result["top_drivers"]  = top_all
            result["risk_drivers"] = top_neg
            result["text"]         = self._generate_text(ticker, score, top_pos, top_neg)

        except Exception as e:
            log.debug(f"[{self.agent_name}] SHAP local error for {ticker}: {e}")
            result["text"] = self._rule_based_text(X_row, score, ticker)

        return result

    # ── Natural-language text ─────────────────────────────────────────────────

    @staticmethod
    def _generate_text(
        ticker:   str,
        score:    float,
        positive: List[Dict],
        negative: List[Dict],
    ) -> str:
        """Generate a human-readable paragraph explaining the prediction."""
        label      = "Outperform" if score >= 0.5 else "Underperform"
        confidence = "high" if abs(score - 0.5) > 0.25 else "moderate"
        lines      = [f"{ticker} — Score: {score:.2f} ({label}, confidence {confidence})"]
        lines.append("")

        if positive:
            lines.append("Factors IN FAVOUR:")
            for d in positive[:4]:
                val_str = _format_value(d["feature"], d["raw_value"])
                lines.append(f"  + {d['description']} = {val_str}  "
                              f"[SHAP contribution: +{d['shap_value']:.3f}]")

        if negative:
            lines.append("")
            lines.append("Factors AGAINST:")
            for d in negative[:4]:
                val_str = _format_value(d["feature"], d["raw_value"])
                lines.append(f"  - {d['description']} = {val_str}  "
                              f"[SHAP contribution: {d['shap_value']:.3f}]")

        return "\n".join(lines)

    @staticmethod
    def _rule_based_text(X_row: pd.Series, score: float, ticker: str) -> str:
        """Fallback without SHAP: text based on raw value thresholds."""
        label = "Outperform" if score >= 0.5 else "Underperform"
        lines = [f"{ticker} — Score: {score:.2f} ({label})"]
        lines.append("(Rule-based explanation — install shap for full analysis)")
        lines.append("")

        checks = [
            ("roe",               lambda v: v > 0.15,  "Strong ROE", "positive"),
            ("net_margin",        lambda v: v > 0.10,  "Positive net margin", "positive"),
            ("debt_to_ebitda",    lambda v: v > 6.0,   "High debt vs EBITDA", "negative"),
            ("current_ratio",     lambda v: v < 1.0,   "Liquidity risk", "negative"),
            ("revenue_yoy_growth",lambda v: v > 0,     "Revenue growth", "positive"),
            ("fcf",               lambda v: v < 0,     "Negative FCF", "negative"),
            ("momentum_12m",      lambda v: v > 0,     "Positive annual momentum", "positive"),
            ("rsi_14",            lambda v: v > 70,    "RSI overbought", "negative"),
        ]
        for feat, fn, desc, direction in checks:
            val = X_row.get(feat, np.nan)
            if pd.notna(val) and fn(val):
                sign = "+ " if direction == "positive" else "- "
                lines.append(f"  {sign}{desc} ({feat}={val:.2f})")

        return "\n".join(lines)

    def _save_shap_bar_plot(self, imp: pd.Series, suffix: str = ""):
        """
        Save a horizontal bar chart of the top features by |SHAP|.
        The file is named shap_bar<suffix>.png.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.4)))
            colors = ["#2196F3" if v >= 0 else "#F44336" for v in imp.values]
            labels = [_describe(f) for f in imp.index]
            ax.barh(range(len(imp)), imp.values[::-1], color=colors[::-1], alpha=0.85)
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels(labels[::-1], fontsize=9)
            ax.set_xlabel("Mean SHAP importance |Δscore|")
            ax.set_title(f"[{self.agent_name}] Top features by SHAP importance{suffix}")
            ax.axvline(0, color="black", lw=0.7)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            plot_path = self.results_dir / f"shap_bar{suffix}.png"
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.debug(f"[{self.agent_name}] Could not generate shap_bar: {e}")



# ── Value formatting helper ───────────────────────────────────────────────────

def _format_value(feature: str, value: float) -> str:
    """Format a raw feature value in a human-readable way according to feature type."""
    if pd.isna(value):
        return "N/A"
    pct_features = {
        "roe","roa","roi","roic","net_margin","gross_margin","fcf_margin",
        "ebitda_margin","operating_margin","revenue_yoy_growth","net_income_yoy_growth",
        "eps_yoy_growth","fcf_yoy_growth","total_debt_yoy_growth","fcf_yield",
        "earnings_yield","momentum_1m","momentum_3m","momentum_6m","momentum_12m",
        "price_vs_52w_high","price_vs_52w_low","sma_20","sma_50","sma_200",
        "volatility_20d","volatility_60d","pe_vs_5y_median","pb_vs_5y_median",
        "roa_change_yoy","gross_margin_change_yoy","current_ratio_change_yoy",
        "fcf_margin","ebitda_margin","operating_margin",
    }
    if feature in pct_features:
        return f"{value:.1%}"
    if "ratio" in feature or feature in {"pe_ratio","pb_ratio","ps_ratio","ev_to_ebitda",
                                          "debt_to_ebitda","interest_coverage"}:
        return f"{value:.2f}x"
    if feature in {"rsi_14","rsi_28","vix"}:
        return f"{value:.1f}"
    return f"{value:.3f}"


# =============================================================================
# Convenience function for integrating into agents
# =============================================================================

def build_explainer_for_agent(
    agent_name:   str,
    model,                    # sklearn/xgb model already fitted
    feature_cols: List[str],
    X_train:      pd.DataFrame,
    results_dir:  str,
    fold:         Optional[int] = None,
    model_type:   str = "tree",
) -> AgentExplainer:
    """
    Build, fit, and save the global explanation for an agent in a single step.
    Call this at the end of each agent's fit() method.
    """
    explainer = AgentExplainer(agent_name, feature_cols, results_dir, model_type)
    explainer.fit_explainer(model, X_train)
    explainer.save_global_explanation(fold)
    return explainer
