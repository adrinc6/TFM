"""Visualization helpers for evaluation results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)

STYLE = {
	"strategy": {"color": "#2196F3", "lw": 2.0, "label": "Estrategia ML"},
	"benchmark": {"color": "#FF5722", "lw": 1.5, "ls": "--", "label": "S&P 500 B&H"},
	"alpha": {"color": "#4CAF50", "lw": 1.5, "label": "Alpha acumulado"},
}


class Visualizer:
	"""Genera y guarda todas las graficas comparativas del backtest."""

	def __init__(self, plots_dir: str = "results/plots"):
		self.plots_dir = Path(plots_dir)
		self.plots_dir.mkdir(parents=True, exist_ok=True)

	def plot_full_report(
		self,
		strategy_returns: pd.Series,
		benchmark_returns: pd.Series,
		fold_results: List[Dict],
		agent_diagnostics: Optional[Dict] = None,
		suffix: str = "",
	):
		fig = plt.figure(figsize=(18, 22))
		gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

		ax1 = fig.add_subplot(gs[0, :])
		ax2 = fig.add_subplot(gs[1, 0])
		ax3 = fig.add_subplot(gs[1, 1])
		ax4 = fig.add_subplot(gs[2, 0])
		ax5 = fig.add_subplot(gs[2, 1])
		ax6 = fig.add_subplot(gs[3, 0])
		ax7 = fig.add_subplot(gs[3, 1])

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

		fig.suptitle("Multi-Agent ML Stock Picker — Analisis Completo",
					 fontsize=16, fontweight="bold", y=0.98)

		path = self.plots_dir / f"full_report{suffix}.png"
		fig.savefig(path, dpi=150, bbox_inches="tight")
		plt.close(fig)
		log.info(f"[Visualizer] Reporte completo -> {path}")

	def _wealth_curves(self, ax, strat: pd.Series, bench: pd.Series):
		w_strat = (1 + strat).cumprod()
		w_bench = (1 + bench).cumprod()
		ax.plot(w_strat.index, w_strat.values, **STYLE["strategy"])
		ax.plot(w_bench.index, w_bench.values, **STYLE["benchmark"])
		ax.axhline(1, color="gray", lw=0.8, ls=":")
		ax.set_title("Curva de Riqueza Acumulada", fontweight="bold")
		ax.set_ylabel("Valor de la cartera (base 1)")
		ax.legend()
		ax.grid(alpha=0.3)
		final_s = float(w_strat.iloc[-1] - 1) if not w_strat.empty else 0
		final_b = float(w_bench.iloc[-1] - 1) if not w_bench.empty else 0
		ax.text(
			0.01,
			0.95,
			f"Estrategia: {final_s:+.1%}  |  S&P 500: {final_b:+.1%}",
			transform=ax.transAxes,
			fontsize=10,
			va="top",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
		)

	def _drawdown(self, ax, strat: pd.Series, bench: pd.Series):
		def _dd(r):
			w = (1 + r).cumprod()
			pk = w.cummax()
			return (w - pk) / pk * 100

		dd_strat = _dd(strat)
		dd_bench = _dd(bench)
		ax.fill_between(dd_strat.index, dd_strat.values, 0,
						alpha=0.4, color=STYLE["strategy"]["color"], label="Estrategia")
		ax.plot(dd_bench.index, dd_bench.values,
				color=STYLE["benchmark"]["color"], lw=1.2, ls="--", label="S&P 500")
		ax.set_title("Drawdown (%)", fontweight="bold")
		ax.set_ylabel("Drawdown (%)")
		ax.legend()
		ax.grid(alpha=0.3)

	def _cumulative_alpha(self, ax, strat: pd.Series, bench: pd.Series):
		common = strat.index.intersection(bench.index)
		alpha = (strat.loc[common] - bench.loc[common]).cumsum() * 100
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
			ax.axis("off")
			return
		labels = [f"Fold {r.get('fold','?')}\n{r.get('test_start','')[:4]}" for r in fold_results]
		s_ret = [r.get("strategy_annualized_return", 0) * 100 for r in fold_results]
		b_ret = [r.get("benchmark_annualized_return", 0) * 100 for r in fold_results]
		x = np.arange(len(labels))
		w = 0.35
		ax.bar(x - w / 2, s_ret, w, color=STYLE["strategy"]["color"],
			   alpha=0.8, label="Estrategia")
		ax.bar(x + w / 2, b_ret, w, color=STYLE["benchmark"]["color"],
			   alpha=0.8, label="S&P 500")
		ax.axhline(0, color="black", lw=0.8)
		ax.set_title("Retorno Anualizado por Fold (%)", fontweight="bold")
		ax.set_ylabel("Retorno anualizado (%)")
		ax.set_xticks(x)
		ax.set_xticklabels(labels, fontsize=8)
		ax.legend()
		ax.grid(alpha=0.3, axis="y")

	def _return_distribution(self, ax, strat: pd.Series, bench: pd.Series):
		if strat.empty or bench.empty:
			ax.axis("off")
			return
		if not isinstance(strat.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
			ax.axis("off")
			return
		if not isinstance(bench.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
			ax.axis("off")
			return
		monthly_s = strat.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
		monthly_b = bench.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
		ax.hist(monthly_s, bins=30, alpha=0.6, color=STYLE["strategy"]["color"],
				label=f"Estrategia (m={monthly_s.mean():.1f}%)")
		ax.hist(monthly_b, bins=30, alpha=0.6, color=STYLE["benchmark"]["color"],
				label=f"S&P 500 (m={monthly_b.mean():.1f}%)")
		ax.axvline(0, color="black", lw=0.8, ls=":")
		ax.set_title("Distribucion de Retornos Mensuales (%)", fontweight="bold")
		ax.set_xlabel("Retorno mensual (%)")
		ax.set_ylabel("Frecuencia")
		ax.legend()
		ax.grid(alpha=0.3)

	def _sharpe_by_fold(self, ax, fold_results: List[Dict]):
		if not fold_results:
			ax.axis("off")
			return
		labels = [f"Fold {r.get('fold','?')}" for r in fold_results]
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
		ax.set_xticks(x)
		ax.set_xticklabels(labels, fontsize=8)
		ax.legend()
		ax.grid(alpha=0.3)

	def _agent_auc_bars(self, ax, agent_diagnostics: Dict):
		names, aucs, stds = [], [], []
		for agent, diag in agent_diagnostics.items():
			cv = diag.get("last_train_metrics", {})
			if "mean_auc" in cv:
				names.append(agent.capitalize())
				aucs.append(cv["mean_auc"])
				stds.append(cv.get("std_auc", 0))
		if not names:
			ax.axis("off")
			return
		colors = ["#2196F3", "#9C27B0", "#FF9800", "#F44336"]
		x = np.arange(len(names))
		ax.bar(x, aucs, yerr=stds, capsize=5,
			   color=colors[:len(names)], alpha=0.8)
		ax.axhline(0.5, color="gray", lw=1, ls="--", label="Aleatorio")
		ax.set_title("AUC Cross-Validation por Agente", fontweight="bold")
		ax.set_ylabel("AUC-ROC")
		ax.set_ylim(0, 1)
		ax.set_xticks(x)
		ax.set_xticklabels(names)
		ax.legend()
		ax.grid(alpha=0.3, axis="y")

	def plot_feature_importances(self, importances: pd.Series,
								 agent_name: str, fold: Optional[int] = None, top_n: int = 20):
		top = importances.nlargest(top_n)
		fig, ax = plt.subplots(figsize=(10, 6))
		top.sort_values().plot.barh(ax=ax, color="#2196F3", alpha=0.8)
		suffix = f" (Fold {fold})" if fold is not None else ""
		ax.set_title(f"Feature Importances — {agent_name.capitalize()}{suffix}",
					 fontweight="bold")
		ax.set_xlabel("Importancia")
		ax.grid(alpha=0.3, axis="x")
		path = self.plots_dir / f"feat_imp_{agent_name}{'_fold' + str(fold) if fold else ''}.png"
		fig.savefig(path, dpi=120, bbox_inches="tight")
		plt.close(fig)
		log.info(f"[Visualizer] Feature importances -> {path.name}")

	def plot_fold_performance(self, fold_result: Dict, fold_id: int):
		"""
		Plot diario del quarter de test: cada ticker seleccionado (líneas finas),
		cartera media (línea gruesa azul) y benchmark (línea gruesa naranja).
		"""
		ticker_series: Dict[str, pd.Series] = fold_result.get("_ticker_price_series", {})
		strat_series: Optional[pd.Series] = fold_result.get("_strat_price_series")
		bench_series: Optional[pd.Series] = fold_result.get("_bench_price_series")

		if not ticker_series or strat_series is None or bench_series is None:
			return

		year_quarter = fold_result.get("year_quarter", f"Fold {fold_id}")
		ticker_weights: Dict[str, float] = fold_result.get("ticker_weights", {})
		weighting_mode: str = fold_result.get("weighting_mode", "equiponderado")

		fig, ax = plt.subplots(figsize=(12, 6))

		# Tickers individuales — líneas finas y semitransparentes
		for ticker, series in ticker_series.items():
			total_ret = float(series.iloc[-1] - 1) if len(series) > 0 else 0.0
			w = ticker_weights.get(ticker)
			w_str = f"  w={w:.1%}" if w is not None else ""
			ax.plot(
				series.index, series.values,
				lw=0.9, alpha=0.35, color="#90CAF9",
				label=f"{ticker} ({total_ret:+.1%}{w_str})",
			)

		# Benchmark — línea gruesa naranja discontinua
		bench_ret = float(bench_series.iloc[-1] - 1) if len(bench_series) > 0 else 0.0
		ax.plot(
			bench_series.index, bench_series.values,
			color=STYLE["benchmark"]["color"], lw=2.5, ls="--", zorder=4,
			label=f"S&P 500 ({bench_ret:+.1%})",
		)

		# Cartera media — línea gruesa azul
		strat_ret = float(strat_series.iloc[-1] - 1) if len(strat_series) > 0 else 0.0
		ax.plot(
			strat_series.index, strat_series.values,
			color=STYLE["strategy"]["color"], lw=2.5, zorder=5,
			label=f"Cartera ML ({strat_ret:+.1%})",
		)

		ax.axhline(1.0, color="gray", lw=0.7, ls=":")
		ax.set_title(
			f"Rendimiento diario — {year_quarter} (Fold {fold_id})  [{weighting_mode}]",
			fontweight="bold",
		)
		ax.set_ylabel("Valor (base 1 = inicio del quarter)")
		ax.legend(fontsize=7, ncol=3, loc="upper left")
		ax.grid(alpha=0.3)
		fig.tight_layout()

		path = self.plots_dir / f"fold_{fold_id:03d}_{year_quarter}_performance.png"
		fig.savefig(path, dpi=130, bbox_inches="tight")
		plt.close(fig)
		log.info(f"[Visualizer] Rendimiento fold {fold_id} -> {path.name}")

	def plot_score_distribution(self, scores_df: pd.DataFrame, fold: Optional[int] = None):
		agent_cols = [c for c in ["fundamental_score", "valuation_score",
								   "momentum_score", "bear_score", "final_score"]
					  if c in scores_df.columns]
		if not agent_cols or "label" not in scores_df.columns:
			return
		fig, axes = plt.subplots(1, len(agent_cols), figsize=(5 * len(agent_cols), 4))
		if len(agent_cols) == 1:
			axes = [axes]
		for ax, col in zip(axes, agent_cols):
			for lbl, color in [(1, "#2196F3"), (0, "#F44336")]:
				sub = scores_df[scores_df["label"] == lbl][col].dropna()
				ax.hist(sub, bins=25, alpha=0.6, color=color,
						label="Outperform" if lbl else "Underperform")
			ax.set_title(col.replace("_", " ").title(), fontsize=9)
			ax.legend(fontsize=7)
			ax.grid(alpha=0.3)
		suffix = f"_fold{fold}" if fold is not None else ""
		path = self.plots_dir / f"score_dist{suffix}.png"
		fig.savefig(path, dpi=120, bbox_inches="tight")
		plt.close(fig)
		log.info(f"[Visualizer] Distribucion scores -> {path.name}")
