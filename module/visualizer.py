# =============================================================================
# module/visualizer.py — Visualización de resultados del pipeline
# =============================================================================
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sin GUI (compatible con servidores)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

STYLE = {
    "strategy":  {"color": "#2196F3", "lw": 2.0, "label": "Estrategia ML"},
    "benchmark": {"color": "#FF5722", "lw": 1.5, "ls": "--", "label": "S&P 500 B&H"},
    "alpha":     {"color": "#4CAF50", "lw": 1.5, "label": "Alpha acumulado"},
}


class Visualizer:
    """Genera y guarda todas las gráficas comparativas del backtest."""

    def __init__(self, plots_dir: str = "results/plots", no_plot: bool = False):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.no_plot   = no_plot

    # ── Gráfica principal ─────────────────────────────────────────────────────

    def plot_full_report(
        self,
        strategy_returns:  pd.Series,
        benchmark_returns: pd.Series,
        fold_results:      List[Dict],
        agent_diagnostics: Optional[Dict] = None,
        suffix:            str = "",
    ):
        fig = plt.figure(figsize=(18, 22))
        gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, :])   # Wealth curves
        ax2 = fig.add_subplot(gs[1, 0])   # Drawdown
        ax3 = fig.add_subplot(gs[1, 1])   # Alpha acumulado
        ax4 = fig.add_subplot(gs[2, 0])   # Retornos anuales por fold
        ax5 = fig.add_subplot(gs[2, 1])   # Distribución de retornos mensuales
        ax6 = fig.add_subplot(gs[3, 0])   # Sharpe por fold
        ax7 = fig.add_subplot(gs[3, 1])   # Scores de agentes (si disponibles)

        self._wealth_curves(ax1, strategy_returns, benchmark_returns)
        self._drawdown(ax2, strategy_returns, benchmark_returns)
        self._cumulative_alpha(ax3, strategy_returns, benchmark_returns)
        self._annual_returns_by_fold(ax4, fold_results)
        self._return_distribution(ax5, strategy_returns, benchmark_returns)
        self._sharpe_by_fold(ax6, fold_results)
        if agent_diagnostics:
            self._agent_auc_bars(ax7, agent_diagnostics)
        else:
            ax7.axis("off")

        fig.suptitle("Multi-Agent ML Stock Picker — Análisis Completo",
                     fontsize=16, fontweight="bold", y=0.98)

        path = self.plots_dir / f"full_report{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if not self.no_plot:
            plt.show()
        plt.close(fig)
        log.info(f"[Visualizer] Reporte completo → {path}")

    # ── Subplots individuales ─────────────────────────────────────────────────

    def _wealth_curves(self, ax, strat: pd.Series, bench: pd.Series):
        w_strat = (1 + strat).cumprod()
        w_bench = (1 + bench).cumprod()
        ax.plot(w_strat.index, w_strat.values, **STYLE["strategy"])
        ax.plot(w_bench.index, w_bench.values, **STYLE["benchmark"])
        ax.axhline(1, color="gray", lw=0.8, ls=":")
        ax.set_title("Curva de Riqueza Acumulada", fontweight="bold")
        ax.set_ylabel("Valor de la cartera (base 1)")
        ax.legend(); ax.grid(alpha=0.3)
        # Anotar retorno final
        final_s = float(w_strat.iloc[-1] - 1) if not w_strat.empty else 0
        final_b = float(w_bench.iloc[-1] - 1) if not w_bench.empty else 0
        ax.text(0.01, 0.95,
                f"Estrategia: {final_s:+.1%}  |  S&P 500: {final_b:+.1%}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    def _drawdown(self, ax, strat: pd.Series, bench: pd.Series):
        def _dd(r):
            w  = (1 + r).cumprod()
            pk = w.cummax()
            return (w - pk) / pk * 100
        ax.fill_between(_dd(strat).index, _dd(strat).values, 0,
                        alpha=0.4, color=STYLE["strategy"]["color"], label="Estrategia")
        ax.plot(_dd(bench).index, _dd(bench).values,
                color=STYLE["benchmark"]["color"], lw=1.2, ls="--", label="S&P 500")
        ax.set_title("Drawdown (%)", fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(); ax.grid(alpha=0.3)

    def _cumulative_alpha(self, ax, strat: pd.Series, bench: pd.Series):
        common = strat.index.intersection(bench.index)
        alpha  = (strat.loc[common] - bench.loc[common]).cumsum() * 100
        ax.plot(alpha.index, alpha.values, **STYLE["alpha"])
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.fill_between(alpha.index, alpha.values, 0,
                        where=alpha > 0, alpha=0.2, color="#4CAF50")
        ax.fill_between(alpha.index, alpha.values, 0,
                        where=alpha < 0, alpha=0.2, color="#F44336")
        ax.set_title("Alpha Acumulado vs S&P 500 (%)", fontweight="bold")
        ax.set_ylabel("Alpha acumulado (pp)")
        ax.grid(alpha=0.3)

    def _annual_returns_by_fold(self, ax, fold_results: List[Dict]):
        if not fold_results:
            ax.axis("off"); return
        labels  = [f"Fold {r.get('fold','?')}\n{r.get('test_start','')[:4]}"
                   for r in fold_results]
        s_ret   = [r.get("strategy_annualized_return", 0) * 100 for r in fold_results]
        b_ret   = [r.get("benchmark_annualized_return", 0) * 100 for r in fold_results]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, s_ret, w, color=STYLE["strategy"]["color"],
               alpha=0.8, label="Estrategia")
        ax.bar(x + w/2, b_ret, w, color=STYLE["benchmark"]["color"],
               alpha=0.8, label="S&P 500")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Retorno Anualizado por Fold (%)", fontweight="bold")
        ax.set_ylabel("Retorno anualizado (%)")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.legend(); ax.grid(alpha=0.3, axis="y")

    def _return_distribution(self, ax, strat: pd.Series, bench: pd.Series):
        monthly_s = strat.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
        monthly_b = bench.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
        ax.hist(monthly_s, bins=30, alpha=0.6, color=STYLE["strategy"]["color"],
                label=f"Estrategia (μ={monthly_s.mean():.1f}%)")
        ax.hist(monthly_b, bins=30, alpha=0.6, color=STYLE["benchmark"]["color"],
                label=f"S&P 500 (μ={monthly_b.mean():.1f}%)")
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_title("Distribución de Retornos Mensuales (%)", fontweight="bold")
        ax.set_xlabel("Retorno mensual (%)"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.grid(alpha=0.3)

    def _sharpe_by_fold(self, ax, fold_results: List[Dict]):
        if not fold_results:
            ax.axis("off"); return
        labels  = [f"Fold {r.get('fold','?')}" for r in fold_results]
        s_sharp = [r.get("strategy_sharpe", 0) for r in fold_results]
        b_sharp = [r.get("benchmark_sharpe", 0) for r in fold_results]
        x = np.arange(len(labels))
        ax.plot(x, s_sharp, "o-", color=STYLE["strategy"]["color"],
                lw=2, label="Estrategia", ms=7)
        ax.plot(x, b_sharp, "s--", color=STYLE["benchmark"]["color"],
                lw=1.5, label="S&P 500", ms=6)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.axhline(1, color="green", lw=0.8, ls=":", alpha=0.5, label="Sharpe=1")
        ax.set_title("Ratio de Sharpe por Fold", fontweight="bold")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.legend(); ax.grid(alpha=0.3)

    def _agent_auc_bars(self, ax, agent_diagnostics: Dict):
        """Barras de AUC de CV de cada agente."""
        names, aucs, stds = [], [], []
        for agent, diag in agent_diagnostics.items():
            cv = diag.get("last_train_metrics", {})
            if "mean_auc" in cv:
                names.append(agent.capitalize())
                aucs.append(cv["mean_auc"])
                stds.append(cv.get("std_auc", 0))
        if not names:
            ax.axis("off"); return
        colors = ["#2196F3","#9C27B0","#FF9800","#F44336"]
        x = np.arange(len(names))
        ax.bar(x, aucs, yerr=stds, capsize=5,
               color=colors[:len(names)], alpha=0.8)
        ax.axhline(0.5, color="gray", lw=1, ls="--", label="Aleatorio")
        ax.set_title("AUC Cross-Validation por Agente", fontweight="bold")
        ax.set_ylabel("AUC-ROC"); ax.set_ylim(0, 1)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.legend(); ax.grid(alpha=0.3, axis="y")

    # ── Plots independientes ──────────────────────────────────────────────────

    def plot_feature_importances(self, importances: pd.Series,
                                 agent_name: str, fold: Optional[int] = None, top_n: int = 20):
        top  = importances.nlargest(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        top.sort_values().plot.barh(ax=ax, color="#2196F3", alpha=0.8)
        suffix = f" (Fold {fold})" if fold is not None else ""
        ax.set_title(f"Feature Importances — {agent_name.capitalize()}{suffix}",
                     fontweight="bold")
        ax.set_xlabel("Importancia")
        ax.grid(alpha=0.3, axis="x")
        path = self.plots_dir / f"feat_imp_{agent_name}{'_fold'+str(fold) if fold else ''}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        log.info(f"[Visualizer] Feature importances → {path.name}")

    def plot_score_distribution(self, scores_df: pd.DataFrame, fold: Optional[int] = None):
        """Distribución de scores de cada agente separada por label real."""
        agent_cols = [c for c in ["fundamental_score","valuation_score",
                                   "momentum_score","bear_score","final_score"]
                      if c in scores_df.columns]
        if not agent_cols or "label" not in scores_df.columns:
            return
        fig, axes = plt.subplots(1, len(agent_cols), figsize=(5 * len(agent_cols), 4))
        if len(agent_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, agent_cols):
            for lbl, color in [(1,"#2196F3"), (0,"#F44336")]:
                sub = scores_df[scores_df["label"] == lbl][col].dropna()
                ax.hist(sub, bins=25, alpha=0.6, color=color,
                        label="Outperform" if lbl else "Underperform")
            ax.set_title(col.replace("_"," ").title(), fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        suffix = f"_fold{fold}" if fold is not None else ""
        path = self.plots_dir / f"score_dist{suffix}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        log.info(f"[Visualizer] Distribución scores → {path.name}")

    def plot_sector_performance(self, results_df: pd.DataFrame):
        """Retorno medio por sector en los stocks seleccionados."""
        if "sector" not in results_df.columns: return
        if "forward_return" not in results_df.columns: return
        sec_perf = (results_df.groupby("sector")["forward_return"]
                               .agg(["mean","std","count"])
                               .sort_values("mean", ascending=False))
        fig, ax = plt.subplots(figsize=(12, 5))
        colors  = ["#4CAF50" if v >= 0 else "#F44336" for v in sec_perf["mean"]]
        ax.bar(sec_perf.index, sec_perf["mean"] * 100, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Retorno Medio por Sector (stocks seleccionados)", fontweight="bold")
        ax.set_xlabel("Sector"); ax.set_ylabel("Retorno medio (%)")
        plt.xticks(rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")
        path = self.plots_dir / "sector_performance.png"
        fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        log.info(f"[Visualizer] Rendimiento sectorial → {path.name}")
