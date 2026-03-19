# =============================================================================
# module/explainer.py — Explicabilidad de predicciones (SHAP)
# =============================================================================
# Para cada predicción responde: ¿por qué este score?
#
# Genera tres niveles de explicación:
#   1. Global  → qué features importan más en el modelo entrenado
#   2. Local   → por qué este ticker concreto recibió ese score
#   3. Texto   → resumen en lenguaje natural legible
#
# Compatible con: XGBoost, GBM, Random Forest, Logistic Regression
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("[Explainer] SHAP no instalado. Instala con: pip install shap")


# ── Descripciones legibles de cada feature ───────────────────────────────────
FEATURE_DESCRIPTIONS = {
    # Rentabilidad
    "roe":                      "Rentabilidad sobre fondos propios (ROE)",
    "roa":                      "Rentabilidad sobre activos (ROA)",
    "roi":                      "Rentabilidad sobre inversión (ROI)",
    "roic":                     "Rentabilidad sobre capital invertido (ROIC)",
    "net_margin":               "Margen neto (beneficio / ventas)",
    "gross_margin":             "Margen bruto",
    "fcf_margin":               "Margen de flujo de caja libre",
    "ebitda_margin":            "Margen EBITDA",
    "operating_margin":         "Margen operativo",
    # Liquidez
    "current_ratio":            "Ratio de liquidez corriente (activo / pasivo cte.)",
    "quick_ratio":              "Ratio de liquidez inmediata",
    # Solvencia
    "debt_equity":              "Ratio deuda / fondos propios",
    "debt_to_ebitda":           "Deuda neta / EBITDA",
    "interest_coverage":        "Cobertura de intereses (EBIT / intereses)",
    # Crecimiento
    "revenue_yoy_growth":       "Crecimiento de ingresos interanual",
    "net_income_yoy_growth":    "Crecimiento de beneficio neto interanual",
    "eps_yoy_growth":           "Crecimiento de BPA (EPS) interanual",
    "fcf_yoy_growth":           "Crecimiento de FCF interanual",
    "total_debt_yoy_growth":    "Crecimiento de la deuda total interanual",
    # Calidad
    "accruals_ratio":           "Ratio de devengos (calidad contable, menor=mejor)",
    "capex_to_revenue":         "Intensidad de capital (CapEx / ventas)",
    "consecutive_losses":       "Trimestres consecutivos con pérdidas",
    # Valoración
    "pe_ratio":                 "Precio / BPA (PER)",
    "pb_ratio":                 "Precio / Valor en libros (P/B)",
    "ps_ratio":                 "Precio / Ventas (P/S)",
    "ev_to_ebitda":             "EV / EBITDA",
    "fcf_yield":                "Rentabilidad por FCF (FCF / market cap)",
    "earnings_yield":           "Rentabilidad por beneficios (inverso del PER)",
    "pe_vs_5y_median":          "PER actual vs mediana histórica 5Y (>0 = más caro)",
    "pb_vs_5y_median":          "P/B actual vs mediana histórica 5Y",
    "ev_ebitda_vs_5y_median":   "EV/EBITDA actual vs mediana histórica 5Y",
    # Técnicos
    "rsi_14":                   "RSI 14 días (>70 sobrecompra, <30 sobreventa)",
    "rsi_28":                   "RSI 28 días",
    "macd":                     "MACD",
    "macd_hist":                "Histograma MACD (momentum de tendencia)",
    "sma_20":                   "Distancia al SMA 20 días (%)",
    "sma_50":                   "Distancia al SMA 50 días (%)",
    "sma_200":                  "Distancia al SMA 200 días (tendencia larga)",
    "bb_pct":                   "Posición en Bandas de Bollinger (0=min, 1=max)",
    "momentum_1m":              "Momentum 1 mes",
    "momentum_3m":              "Momentum 3 meses",
    "momentum_6m":              "Momentum 6 meses",
    "momentum_12m":             "Momentum 12 meses",
    "price_vs_52w_high":        "Distancia al máximo de 52 semanas",
    "volatility_20d":           "Volatilidad realizada 20 días (anualizada)",
    "volatility_60d":           "Volatilidad realizada 60 días (anualizada)",
    "vol_ratio_20_50":          "Ratio de volumen 20d / 50d (>1 = expansión)",
    # Insiders
    "insider_net_shares_90d":   "Acciones compradas netas por insiders (90 días)",
    "insider_sell_ratio":       "Proporción de ventas de insiders (>0.7 = red flag)",
    # Macro
    "vix":                      "VIX (volatilidad implícita de mercado)",
    "yield_curve":              "Pendiente curva de tipos (10Y - 2Y)",
    "sp500_momentum_3m":        "Momentum del S&P 500 a 3 meses",
    "sp500_momentum_12m":       "Momentum del S&P 500 a 12 meses",
    # Flags bear
    "debt_growth_high":         "FLAG: Deuda creciendo >20% interanual",
    "fcf_negative":             "FLAG: FCF negativo",
    "liquidity_risk":           "FLAG: Ratio corriente < 1 (riesgo liquidez)",
    "low_coverage":             "FLAG: Cobertura intereses < 1.5x",
    "insider_selling":          "FLAG: Insiders vendiendo >70% de operaciones",
    # Scores de agentes (meta-learner)
    "fundamental_score":        "Score del Agente Fundamental (salud financiera)",
    "valuation_score":          "Score del Agente de Valoración (infravaloración)",
    "momentum_score":           "Score del Agente de Momentum (tendencia técnica)",
    "bear_score":               "Score del Agente Bear (riesgo, <0.5 es bueno)",
}


def _describe(feature: str) -> str:
    """Devuelve descripción legible de un feature, o el nombre si no está mapeado."""
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
        self.results_dir  = Path(results_dir) / agent_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_type   = model_type
        self._explainer   = None
        self._shap_values_train: Optional[np.ndarray] = None
        self._X_train_summary:   Optional[pd.DataFrame] = None

    # ── Construcción del explainer ────────────────────────────────────────────

    def fit_explainer(self, model, X_train: pd.DataFrame, max_background: int = 100):
        """
        Construye el explainer SHAP sobre el modelo entrenado.

        model:       el objeto sklearn/xgb ya entrenado
        X_train:     datos de entrenamiento (para el background de SHAP)
        max_background: filas para el background (más = más preciso, más lento)
        """
        if not SHAP_AVAILABLE:
            log.warning(f"[{self.agent_name}] SHAP no disponible — sin explicaciones")
            return self

        X = X_train[self.feature_cols].copy().fillna(0)

        try:
            if self.model_type == "tree":
                # TreeExplainer: rápido y exacto para XGB/RF/GBM
                self._explainer = shap.TreeExplainer(model)
            elif self.model_type == "linear":
                # LinearExplainer para Logistic Regression
                self._explainer = shap.LinearExplainer(model, X)
            else:
                # KernelExplainer: universal pero lento — usar muestra pequeña
                background = shap.sample(X, min(50, len(X)))
                self._explainer = shap.KernelExplainer(model.predict_proba, background)

            # Calcular SHAP values sobre muestra de train para análisis global
            background_data = shap.sample(X, min(max_background, len(X)))
            self._shap_values_train = self._explainer.shap_values(background_data)
            self._X_train_summary   = background_data

            # Normalizar a 2D (n_samples, n_features) — clase positiva (índice 1)
            sv = self._shap_values_train
            if isinstance(sv, list):
                # SHAP antiguo: lista [clase_0, clase_1]
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                # SHAP nuevo con RF/GBM: array 3D (n_samples, n_features, n_classes)
                sv = sv[:, :, 1]
            self._shap_values_train = sv

            log.info(f"[{self.agent_name}] Explainer SHAP listo "
                     f"({type(self._explainer).__name__}, {len(background_data)} background)")

        except Exception as e:
            log.warning(f"[{self.agent_name}] Error construyendo explainer SHAP: {e}")

        return self

    # ── Explicación global ────────────────────────────────────────────────────

    def global_importance(self) -> Optional[pd.Series]:
        """
        Importancia global: mean(|SHAP|) por feature sobre el set de entrenamiento.
        Más robusto que feature_importances_ del modelo ya que tiene unidades reales.
        """
        if self._shap_values_train is None:
            return None
        sv = np.array(self._shap_values_train)
        # Normalizar a 2D (n_samples, n_features) sea cual sea el formato de SHAP
        if sv.ndim == 3:
            sv = sv[:, :, 1]       # tomar clase positiva
        elif sv.ndim == 1:
            sv = sv.reshape(1, -1)
        mean_abs = np.abs(sv).mean(axis=0)   # ahora siempre (n_features,)
        return pd.Series(mean_abs, index=self.feature_cols).sort_values(ascending=False)

    def save_global_explanation(self, fold: Optional[int] = None):
        """
        Guarda importancia global SHAP en disco:
          - CSV con todas las features ordenadas por importancia
          - JSON con las top 20 features y descripciones legibles
          - PNG con el gráfico de barras de importancia SHAP
        """
        imp = self.global_importance()
        if imp is None:
            return
        suffix = f"_fold{fold}" if fold is not None else ""

        # CSV completo
        path = self.results_dir / f"shap_global{suffix}.csv"
        imp.to_csv(path, header=["shap_importance"])

        # JSON top 20 con descripciones
        report = {
            feat: {
                "shap_importance": float(val),
                "description":     _describe(feat),
                "rank":            i + 1,
            }
            for i, (feat, val) in enumerate(imp.head(20).items())
        }
        json_path = self.results_dir / f"shap_global{suffix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Gráfico de barras de las top 15 features por importancia SHAP
        self._save_shap_bar_plot(imp.head(15), suffix)

        log.info(f"[{self.agent_name}] Importancia SHAP global -> {path.name}")

    # ── Explicación local (por ticker) ────────────────────────────────────────

    def explain_prediction(
        self,
        X_row:  pd.Series,
        ticker: str,
        score:  float,
        top_n:  int = 8,
        fold:   Optional[int] = None,
    ) -> Dict:
        """
        Explica por qué el agente dio ese score a un ticker concreto.

        Devuelve:
            {
              "ticker":        "AAPL",
              "score":         0.73,
              "label":         "Outperform",
              "top_drivers":   [{"feature": "roe", "shap": 0.15, "value": 0.28, ...}, ...],
              "risk_drivers":  [...],   # features que penalizan
              "text":          "AAPL recibe score 0.73 (Outperform) principalmente porque..."
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

            # Construir tabla de contribuciones
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

            # Guardar explicación individual
            self._save_local_explanation(result, fold)

        except Exception as e:
            log.debug(f"[{self.agent_name}] SHAP local error para {ticker}: {e}")
            result["text"] = self._rule_based_text(X_row, score, ticker)

        return result

    def explain_batch(
        self,
        X:      pd.DataFrame,
        scores: pd.Series,
        tickers: List[str],
        top_n:  int = 5,
        fold:   Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Explica todas las predicciones de un fold.
        Devuelve DataFrame con una fila por ticker y las top_n contribuciones.
        """
        rows = []
        for ticker, (idx, row) in zip(tickers, X.iterrows()):
            score = float(scores.loc[idx]) if idx in scores.index else 0.5
            exp   = self.explain_prediction(row, ticker, score, top_n=top_n, fold=fold)
            flat  = {
                "ticker": ticker,
                "score":  exp["score"],
                "label":  exp["label"],
            }
            for i, d in enumerate(exp.get("top_drivers", [])[:top_n], 1):
                flat[f"driver_{i}_feature"] = d["feature"]
                flat[f"driver_{i}_shap"]    = round(d["shap_value"], 4)
                flat[f"driver_{i}_value"]   = round(d["raw_value"], 4)
            rows.append(flat)

        df = pd.DataFrame(rows)
        suffix = f"_fold{fold}" if fold is not None else ""
        path   = self.results_dir / f"batch_explanations{suffix}.csv"
        df.to_csv(path, index=False)
        log.info(f"[{self.agent_name}] Batch explanations ({len(df)} tickers) -> {path.name}")
        return df

    # ── Texto en lenguaje natural ─────────────────────────────────────────────

    @staticmethod
    def _generate_text(
        ticker:   str,
        score:    float,
        positive: List[Dict],
        negative: List[Dict],
    ) -> str:
        """Genera un párrafo legible explicando la predicción."""
        label     = "Outperform" if score >= 0.5 else "Underperform"
        confianza = "alta" if abs(score - 0.5) > 0.25 else "moderada"
        lines     = [f"{ticker} — Score: {score:.2f} ({label}, confianza {confianza})"]
        lines.append("")

        if positive:
            lines.append("Factores a FAVOR:")
            for d in positive[:4]:
                val_str = _format_value(d["feature"], d["raw_value"])
                lines.append(f"  + {d['description']} = {val_str}  "
                              f"[contribucion SHAP: +{d['shap_value']:.3f}]")

        if negative:
            lines.append("")
            lines.append("Factores EN CONTRA:")
            for d in negative[:4]:
                val_str = _format_value(d["feature"], d["raw_value"])
                lines.append(f"  - {d['description']} = {val_str}  "
                              f"[contribucion SHAP: {d['shap_value']:.3f}]")

        return "\n".join(lines)

    @staticmethod
    def _rule_based_text(X_row: pd.Series, score: float, ticker: str) -> str:
        """Fallback sin SHAP: texto basado en umbrales de los valores crudos."""
        label = "Outperform" if score >= 0.5 else "Underperform"
        lines = [f"{ticker} — Score: {score:.2f} ({label})"]
        lines.append("(Explicacion basada en reglas — instala shap para analisis completo)")
        lines.append("")

        checks = [
            ("roe",               "> 0.15", lambda v: v > 0.15,  "ROE solido"),
            ("net_margin",        "> 0.10", lambda v: v > 0.10,  "Margen neto positivo"),
            ("debt_to_ebitda",    "> 6",    lambda v: v > 6.0,   "Deuda elevada vs EBITDA"),
            ("current_ratio",     "< 1",    lambda v: v < 1.0,   "Riesgo de liquidez"),
            ("revenue_yoy_growth","> 0",    lambda v: v > 0,     "Crecimiento de ingresos"),
            ("fcf",               "< 0",    lambda v: v < 0,     "FCF negativo"),
            ("momentum_12m",      "> 0",    lambda v: v > 0,     "Momentum anual positivo"),
            ("rsi_14",            "> 70",   lambda v: v > 70,    "RSI sobrecomprado"),
        ]
        for feat, cond, fn, desc in checks:
            val = X_row.get(feat, np.nan)
            if pd.notna(val) and fn(val):
                lines.append(f"  {'+ ' if score >= 0.5 else '- '}{desc} ({feat}={val:.2f})")

        return "\n".join(lines)

    def _save_shap_bar_plot(self, imp: pd.Series, suffix: str = ""):
        """
        Guarda un gráfico de barras horizontales con las top features por |SHAP|.
        El fichero se nombra shap_bar<suffix>.png.
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
            ax.set_xlabel("Importancia SHAP media |Δscore|")
            ax.set_title(f"[{self.agent_name}] Top features por importancia SHAP{suffix}")
            ax.axvline(0, color="black", lw=0.7)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            plot_path = self.results_dir / f"shap_bar{suffix}.png"
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.debug(f"[{self.agent_name}] No se pudo generar shap_bar: {e}")

    def _save_local_explanation(self, result: Dict, fold: Optional[int]):
        """
        Guarda la explicación local de un ticker en JSON con valores formateados.
        Incluye raw_value y value_formatted para cada driver.
        """
        suffix  = f"_fold{fold}" if fold is not None else ""
        ticker  = result.get("ticker", "unknown")
        path    = self.results_dir / "local_explanations"
        path.mkdir(exist_ok=True)
        out_path = path / f"{ticker}{suffix}.json"

        # Enriquecer top_drivers con valor formateado y serializar numpy scalars
        top_drivers_clean = []
        for d in result.get("top_drivers", []):
            entry = {}
            for k, v in d.items():
                if isinstance(v, (np.floating, np.integer)):
                    entry[k] = float(v)
                else:
                    entry[k] = v
            # Añadir valor formateado si no está ya
            if "raw_value" in entry and "value_formatted" not in entry:
                entry["value_formatted"] = _format_value(entry["feature"], entry["raw_value"])
            top_drivers_clean.append(entry)

        clean = {k: v for k, v in result.items() if k != "top_drivers"}
        clean["top_drivers"] = top_drivers_clean

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False, default=str)


# ── Helper de formato de valores ─────────────────────────────────────────────

def _format_value(feature: str, value: float) -> str:
    """Formatea un valor crudo de forma legible según el tipo de feature."""
    if pd.isna(value):
        return "N/A"
    pct_features = {
        "roe","roa","roi","roic","net_margin","gross_margin","fcf_margin",
        "ebitda_margin","operating_margin","revenue_yoy_growth","net_income_yoy_growth",
        "eps_yoy_growth","fcf_yoy_growth","total_debt_yoy_growth","fcf_yield",
        "earnings_yield","momentum_1m","momentum_3m","momentum_6m","momentum_12m",
        "price_vs_52w_high","price_vs_52w_low","sma_20","sma_50","sma_200",
        "volatility_20d","volatility_60d","pe_vs_5y_median","pb_vs_5y_median",
        "sp500_momentum_3m","sp500_momentum_12m",
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
# Función de conveniencia para integrar en los agentes
# =============================================================================

def build_explainer_for_agent(
    agent_name:   str,
    model,                    # modelo sklearn/xgb ya entrenado
    feature_cols: List[str],
    X_train:      pd.DataFrame,
    results_dir:  str,
    fold:         Optional[int] = None,
    model_type:   str = "tree",
) -> AgentExplainer:
    """
    Construye, fitea y guarda la explicacion global de un agente en un solo paso.
    Llama esto al final del fit() de cada agente.
    """
    explainer = AgentExplainer(agent_name, feature_cols, results_dir, model_type)
    explainer.fit_explainer(model, X_train)
    explainer.save_global_explanation(fold)
    return explainer
