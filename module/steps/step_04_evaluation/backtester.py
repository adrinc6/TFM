"""Walk-forward backtesting utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from environment import PORTFOLIO_MIN_SCORE, SCORE_WEIGHTED_PORTFOLIO
from module.steps.step_04_evaluation.metrics import compute_all_metrics

log = logging.getLogger(__name__)


class WalkForwardBacktester:
	"""Walk-forward backtesting con paso trimestral y test de 1 quarter."""

	def __init__(
		self,
		train_years: int = 3,
		test_quarters: int = 1,
		risk_free: float = 0.04,
		results_dir: str = "results/backtest",
		top_n_stocks: int = 10,
		long_only: bool = True,
		score_weighted: bool = SCORE_WEIGHTED_PORTFOLIO,
	):
		self.train_years = train_years
		self.test_quarters = test_quarters
		self.risk_free = risk_free
		self.results_dir = Path(results_dir)
		self.results_dir.mkdir(parents=True, exist_ok=True)
		self.top_n_stocks = top_n_stocks
		self.long_only = long_only
		self.score_weighted = score_weighted
		self.fold_results: List[Dict] = []
		self.all_strategy_returns = pd.Series(dtype=float)
		self.all_benchmark_returns = pd.Series(dtype=float)

	@staticmethod
	def _snap_to_quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
		"""Snap ts to the last day of the quarter it belongs to."""
		return ts + pd.offsets.QuarterEnd(0)

	@staticmethod
	def _quarter_label(ts: pd.Timestamp) -> str:
		"""Return a compact quarter label like 2024Q3."""
		period = pd.Timestamp(ts).to_period("Q")
		return f"{period.year}Q{period.quarter}"

	def _format_folds_overview(self, folds: List[tuple]) -> str:
		"""Build a readable multi-line summary for generated folds."""
		if not folds:
			return "  (sin folds)"

		lines = []
		for i, (train_start, train_end, test_end, train_years) in enumerate(folds):
			test_start = train_end + pd.offsets.Day(1)
			lines.append(
				"  "
				f"F{i:02d} | "
				f"train {train_years}Y [{self._quarter_label(train_start)} -> {self._quarter_label(train_end)}] "
				f"({train_start.date()} -> {train_end.date()}) | "
				f"test [{self._quarter_label(test_end)}] "
				f"({test_start.date()} -> {test_end.date()})"
			)
		return "\n".join(lines)

	def generate_folds(
		self,
		analysis_start_date: str,
		analysis_end_date: str,
	) -> List[tuple]:
		# start/end represent the range of ANALYSED QUARTERS (snapshot quarter).
		# The test of each fold uses the snapshot at train_end, and its actual evaluation
		# is done outside this function with configurable lag/holding.
		start = self._snap_to_quarter_end(pd.Timestamp(analysis_start_date))
		end   = self._snap_to_quarter_end(pd.Timestamp(analysis_end_date))

		folds: List[tuple] = []
		seen: set = set()

		y_offset = pd.DateOffset(years=self.train_years)
		train_end = start
		while True:
			test_end = train_end + pd.offsets.QuarterEnd(self.test_quarters)
			if train_end > end:
				break
			train_start = train_end - y_offset
			key = (train_start.date(), train_end.date(), test_end.date())
			if key not in seen:
				seen.add(key)
				folds.append((train_start, train_end, test_end, self.train_years))
			train_end += pd.offsets.QuarterEnd(self.test_quarters)

		log.info(
			f"[Backtester] {len(folds)} folds generados | test={self.test_quarters}Q | "
			f"train={self.train_years}Y | analysis_start={start.date()} ({self._quarter_label(start)}) | "
			f"analysis_end={end.date()} ({self._quarter_label(end)})"
		)
		log.info("[Backtester] Plan de folds:\n%s", self._format_folds_overview(folds))
		for i, (ts, te, tse, ny) in enumerate(folds):
			log.debug(
				f"  [{i:02d}] {ny}Y train | "
				f"{ts.date()} -> {te.date()} | test {te.date()} -> {tse.date()}"
			)
		return folds

	def simulate_portfolio(
		self,
		predictions_df,
		prices_dict,
		benchmark,
		fold_id,
		test_start,
		test_end,
		train_start=None,
		train_years_int: int = 0,
		analysis_quarter: str | None = None,
	) -> Dict:
		period_id = analysis_quarter if analysis_quarter else str(fold_id)
		min_stocks = max(1, self.top_n_stocks // 2)
		ordered = predictions_df.sort_values("score", ascending=False)
		qualified = ordered[ordered["score"] >= PORTFOLIO_MIN_SCORE]
		if len(qualified) >= min_stocks:
			# Tomar hasta top_n pero garantizar al menos min_stocks
			n_take = max(min(len(qualified), self.top_n_stocks), min_stocks)
			top_df = ordered.head(n_take)[["ticker", "score"]].copy()
			log.info(
				f"[Backtester] {period_id}: {len(qualified)} tickers superaron umbral {PORTFOLIO_MIN_SCORE:.2f} "
				f"→ selecting top {n_take} (min={min_stocks})"
			)
		else:
			# Compressed regime: keep relative selection by ranking.
			top_df = ordered.head(self.top_n_stocks)[["ticker", "score"]].copy()
			log.warning(
				f"[Backtester] {period_id}: solo {len(qualified)} tickers superaron umbral {PORTFOLIO_MIN_SCORE:.2f} "
				f"→ seleccionando top-{self.top_n_stocks} por ranking (scores: "
				f"{ordered['score'].iloc[0]:.3f} .. {ordered['score'].iloc[self.top_n_stocks-1]:.3f})"
			)
		top = top_df["ticker"].tolist()
		if not top:
			log.warning(f"[Backtester] {period_id}: no tickers available — analysis skipped.")
			return {}

		daily_returns = []
		tickers_with_prices = []
		ticker_returns = {}
		bench_period = benchmark.loc[test_start:test_end].dropna()
		if len(bench_period) < 2:
			log.warning(f"[Backtester] {period_id}: insufficient benchmark data — analysis skipped.")
			return {}
		actual_end = bench_period.index.max()
		if actual_end < test_end:
			log.info(
				f"[Backtester] {period_id}: precios parciales "
				f"{test_start.date()} -> {actual_end.date()} (teorico hasta {test_end.date()})"
			)

		for ticker in top:
			if ticker not in prices_dict:
				continue
			prices = prices_dict[ticker]
			cc = "Close" if "Close" in prices.columns else prices.columns[3]
			period = prices.loc[test_start:actual_end, cc]
			if len(period) < 2:
				continue
			ret = period.pct_change().dropna()
			daily_returns.append(ret)
			tickers_with_prices.append(ticker)
			ticker_returns[ticker] = round(float((1 + ret).prod() - 1), 6)

		if not daily_returns:
			return {}

		# Weights: real scores scaled to [w_min, w_max] or equal-weight.
		# Rule: the ticker with the highest score weighs (1 + N/10) times more than the lowest.
		# Intermediate tickers are positioned according to their real scores within that range.
		N = len(tickers_with_prices)
		if self.score_weighted and N > 1:
			scores_arr = (
				top_df.set_index("ticker")
				.loc[tickers_with_prices]["score"]
				.values.astype(float)
			)
			ratio = 1.0 + N / 10.0  # N=10 → 2.0x, N=5 → 1.5x
			s_min, s_max = scores_arr.min(), scores_arr.max()
			if s_max > s_min:
				# Escalar scores al rango [1, ratio]
				raw = 1.0 + (ratio - 1.0) * (scores_arr - s_min) / (s_max - s_min)
			else:
				raw = np.ones(N)
			weights = raw / raw.sum()
		else:
			weights = np.ones(N) / N

		ticker_weights = {t: round(float(w), 6) for t, w in zip(tickers_with_prices, weights)}

		returns_matrix = pd.concat(daily_returns, axis=1)
		returns_matrix.columns = tickers_with_prices
		strat_returns = (returns_matrix * weights).sum(axis=1).dropna()
		common_idx = strat_returns.index.intersection(bench_period.index)
		strat_aligned = strat_returns.loc[common_idx]
		bench_aligned = bench_period.loc[common_idx]

		strat_metrics = compute_all_metrics(strat_aligned, self.risk_free, "strategy")
		bench_metrics = compute_all_metrics(bench_aligned, self.risk_free, "benchmark")
		alpha = strat_metrics["strategy_cumulative_return"] - bench_metrics["benchmark_cumulative_return"]
		excess_sharpe = strat_metrics["strategy_sharpe"] - bench_metrics["benchmark_sharpe"]

		ticker_returns_sorted = dict(sorted(ticker_returns.items(), key=lambda x: x[1], reverse=True))

		# Log de pesos
		weighting_mode = "softmax(score)" if self.score_weighted else "equiponderado"
		log.info(f"[Backtester] {period_id} — cartera final ({weighting_mode}, {len(tickers_with_prices)} stocks):")
		for t in tickers_with_prices:
			log.info(f"    {t:<8}  peso={ticker_weights[t]:.3f}  score={top_df.set_index('ticker').loc[t,'score']:.3f}  ret={ticker_returns[t]:+.2%}")

		# Etiqueta del quarter analizado (quarter del snapshot usado para decidir cartera)
		if analysis_quarter:
			year_quarter = analysis_quarter
		else:
			test_q_ts = pd.Timestamp(test_start)
			year_quarter = f"{test_q_ts.year}Q{test_q_ts.quarter}"

		# Series diarias de precio normalizado (base 1) para el plot de fold
		ticker_price_series: Dict[str, pd.Series] = {}
		for ticker, ret_series in zip(tickers_with_prices, daily_returns):
			ticker_price_series[ticker] = (1 + ret_series).cumprod()

		price_days = int(len(bench_period.loc[test_start:actual_end]))
		fold_result = {
			"fold": fold_id,
			"year_quarter": year_quarter,
			"train_years": train_years_int,
			"train_start": str(train_start.date()) if train_start is not None else None,
			"test_start": str(test_start.date()),
			"test_end": str(actual_end.date()),
			"test_end_target": str(test_end.date()),
			"price_days": price_days,
			"selected_tickers": top,
			"n_stocks": len(top),
			**strat_metrics,
			**bench_metrics,
			"alpha": alpha,
			"excess_sharpe": excess_sharpe,
			"ticker_returns": ticker_returns_sorted,
			"ticker_weights": ticker_weights,
			"weighting_mode": weighting_mode,
			"_ticker_price_series": ticker_price_series,
			"_strat_price_series": (1 + strat_aligned).cumprod(),
			"_bench_price_series": (1 + bench_aligned).cumprod(),
		}

		self.all_strategy_returns = pd.concat([self.all_strategy_returns, strat_aligned])
		self.all_benchmark_returns = pd.concat([self.all_benchmark_returns, bench_aligned])

		path = self.results_dir / f"metrics_{period_id}_{train_years_int}Y.json"
		with open(path, "w") as f:
			json.dump(fold_result, f, indent=2, default=str)

		log.info(
			f"[Backtester] {period_id} — {len(top)} stocks | "
			f"Cartera={strat_metrics['strategy_cumulative_return']:+.2%}  "
			f"Benchmark={bench_metrics['benchmark_cumulative_return']:+.2%}  "
			f"Alpha={alpha:+.2%}  Sharpe={strat_metrics['strategy_sharpe']:.2f}"
		)
		return fold_result

	def summarize(self) -> Dict:
		if not self.fold_results:
			log.warning("[Backtester] Sin folds completados.")
			return {}

		global_strat = compute_all_metrics(self.all_strategy_returns, self.risk_free, "global_strategy")
		global_bench = compute_all_metrics(self.all_benchmark_returns, self.risk_free, "global_benchmark")
		folds_df = pd.DataFrame(self.fold_results)
		mean_alpha = float(folds_df["alpha"].mean()) if "alpha" in folds_df else 0.0
		pct_alpha_pos = float((folds_df["alpha"] > 0).mean()) if "alpha" in folds_df else 0.0

		by_train_years: Dict = {}
		if "train_years" in folds_df.columns:
			for ny, grp in folds_df.groupby("train_years"):
				by_train_years[int(ny)] = {
					"n_folds": int(len(grp)),
					"mean_alpha": float(grp["alpha"].mean()),
					"pct_positive_alpha": float((grp["alpha"] > 0).mean()),
					"mean_strategy_sharpe": float(grp["strategy_sharpe"].mean())
					if "strategy_sharpe" in grp else 0.0,
					"mean_strategy_return": float(grp["strategy_cumulative_return"].mean())
					if "strategy_cumulative_return" in grp else 0.0,
				}

		summary = {
			"timestamp": datetime.now().isoformat(),
			"n_folds": len(self.fold_results),
			"train_years_min": self.train_years,
			"test_quarters": self.test_quarters,
			"top_n_stocks": self.top_n_stocks,
			"mean_alpha": mean_alpha,
			"pct_folds_positive_alpha": pct_alpha_pos,
			"by_train_years": by_train_years,
			**global_strat,
			**global_bench,
		}

		path = self.results_dir / "backtest_summary.json"
		with open(path, "w") as f:
			json.dump(summary, f, indent=2, default=str)

		pd.DataFrame({
			"strategy": self.all_strategy_returns,
			"benchmark": self.all_benchmark_returns,
		}).to_csv(self.results_dir / "returns_series.csv")

		log.info("=" * 65)
		log.info("  RESUMEN WALK-FORWARD BACKTEST")
		log.info(f"  Folds completados : {len(self.fold_results)}")
		for ny, stats in sorted(by_train_years.items()):
			log.info(
				f"  [{ny}Y train]  n={stats['n_folds']:2d} folds | "
				f"ret={stats['mean_strategy_return']:+.2%} | "
				f"alpha={stats['mean_alpha']:+.2%} | "
				f"alpha>0={stats['pct_positive_alpha']:.0%} | "
				f"Sharpe={stats['mean_strategy_sharpe']:.3f}"
			)
		log.info(f"  Alpha medio       : {mean_alpha:+.2%}  ({pct_alpha_pos:.0%} de folds positivos)")
		log.info(f"  Sharpe Estrategia : {global_strat.get('global_strategy_sharpe', 0):.3f}")
		log.info(f"  Sharpe Benchmark  : {global_bench.get('global_benchmark_sharpe', 0):.3f}")
		log.info(f"  Max Drawdown      : {global_strat.get('global_strategy_max_drawdown', 0):.2%}")
		log.info("=" * 65)
		return summary

	def save_folds_summary(self, plots_dir: str = "results/plots"):
		if not self.fold_results:
			log.warning("[Backtester] save_folds_summary: sin resultados todavia.")
			return

		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		import matplotlib.patches as mpatches

		plots_path = Path(plots_dir)
		plots_path.mkdir(parents=True, exist_ok=True)

		df = pd.DataFrame(self.fold_results)

		col_order = [
			"fold", "train_years", "train_start", "test_start", "test_end",
			"strategy_cumulative_return", "benchmark_cumulative_return", "alpha",
			"strategy_sharpe", "benchmark_sharpe", "excess_sharpe",
			"strategy_sortino", "strategy_max_drawdown", "strategy_calmar",
			"strategy_volatility", "n_stocks",
		]
		existing_cols = [c for c in col_order if c in df.columns]
		df_csv = df[existing_cols].copy()

		if "test_start" in df_csv.columns and "test_end" in df_csv.columns:
			df_csv.insert(
				df_csv.columns.get_loc("test_start") + 1,
				"test_period",
				df_csv["test_start"].astype(str) + " -> " + df_csv["test_end"].astype(str),
			)

		csv_path = self.results_dir / "folds_results.csv"
		df_csv.to_csv(csv_path, index=False, float_format="%.4f")
		log.info(f"[Backtester] Folds results CSV -> {csv_path}")

		all_train_years = sorted(df["train_years"].unique()) if "train_years" in df.columns else []
		palette = plt.cm.tab10.colors
		color_map = {ny: palette[i % len(palette)] for i, ny in enumerate(all_train_years)}

		def fold_label(row):
			ts = pd.Timestamp(row["test_start"])
			q = (ts.month - 1) // 3 + 1
			return f"Q{q}'{str(ts.year)[2:]} ({int(row.get('train_years', 0))}Y)"

		df["label"] = df.apply(fold_label, axis=1)
		bar_colors = [color_map.get(int(ny), "steelblue") for ny in df.get("train_years", ["?"] * len(df))]
		alpha_vals = df["alpha"].values if "alpha" in df.columns else []
		strat_ret = df["strategy_cumulative_return"].values if "strategy_cumulative_return" in df.columns else []
		bench_ret = df["benchmark_cumulative_return"].values if "benchmark_cumulative_return" in df.columns else []
		strat_sharpe = df["strategy_sharpe"].values if "strategy_sharpe" in df.columns else []
		bench_sharpe = df["benchmark_sharpe"].values if "benchmark_sharpe" in df.columns else []
		x = np.arange(len(df))

		fig, axes = plt.subplots(2, 2, figsize=(18, 11))
		fig.suptitle(
			f"Resultados Walk-Forward — {len(df)} folds | "
			f"a medio={float(df['alpha'].mean()):+.2%} | "
			f"a positivo={float((df['alpha'] > 0).mean()):.0%}",
			fontsize=13,
			fontweight="bold",
			y=0.99,
		)
		ax_a, ax_b, ax_c, ax_d = axes.flat

		bars = ax_a.bar(x, alpha_vals * 100, color=bar_colors, alpha=0.85, width=0.7)
		ax_a.axhline(0, color="black", lw=0.8)
		ax_a.axhline(float(np.mean(alpha_vals) * 100), color="crimson",
					 lw=1.2, ls="--", label=f"Media {float(np.mean(alpha_vals)) * 100:+.1f}%")
		for bar, val in zip(bars, alpha_vals):
			bar.set_edgecolor("darkgreen" if val >= 0 else "darkred")
			bar.set_linewidth(0.7)
		ax_a.set_title("Alpha por Fold (Estrategia − Benchmark)", fontweight="bold")
		ax_a.set_ylabel("Alpha anualizado (%)")
		ax_a.set_xticks(x)
		ax_a.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7)
		ax_a.legend(fontsize=8)
		ax_a.grid(axis="y", alpha=0.3)

		w = 0.35
		ax_b.bar(
			x - w / 2,
			strat_ret * 100,
			w,
			label="Estrategia",
			color=[color_map.get(int(ny), "steelblue") for ny in df.get("train_years", [0] * len(df))],
			alpha=0.85,
		)
		ax_b.bar(x + w / 2, bench_ret * 100, w, label="Benchmark", color="#FF5722", alpha=0.65)
		ax_b.axhline(0, color="black", lw=0.8)
		ax_b.set_title("Retorno Acumulado por Fold (%)", fontweight="bold")
		ax_b.set_ylabel("Retorno acumulado (%)")
		ax_b.set_xticks(x)
		ax_b.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7)
		ax_b.legend(fontsize=8)
		ax_b.grid(axis="y", alpha=0.3)

		ax_c.plot(x, strat_sharpe, "o-", color="#2196F3", lw=2, ms=6, label="Estrategia")
		ax_c.plot(x, bench_sharpe, "s--", color="#FF5722", lw=1.5, ms=5, label="Benchmark")
		ax_c.axhline(0, color="black", lw=0.8)
		ax_c.axhline(1, color="green", lw=0.8, ls=":", alpha=0.6, label="Sharpe=1")
		ax_c.fill_between(
			x,
			strat_sharpe,
			bench_sharpe,
			where=np.array(strat_sharpe) >= np.array(bench_sharpe),
			alpha=0.12,
			color="green",
			interpolate=True,
		)
		ax_c.fill_between(
			x,
			strat_sharpe,
			bench_sharpe,
			where=np.array(strat_sharpe) < np.array(bench_sharpe),
			alpha=0.12,
			color="red",
			interpolate=True,
		)
		ax_c.set_title("Sharpe Ratio por Fold", fontweight="bold")
		ax_c.set_ylabel("Sharpe Ratio")
		ax_c.set_xticks(x)
		ax_c.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7)
		ax_c.legend(fontsize=8)
		ax_c.grid(alpha=0.3)

		if all_train_years and "train_years" in df.columns:
			box_data = [df[df["train_years"] == ny]["alpha"].values * 100 for ny in all_train_years]
			bp = ax_d.boxplot(box_data, patch_artist=True, widths=0.5,
							  medianprops=dict(color="black", lw=2))
			for patch, ny in zip(bp["boxes"], all_train_years):
				patch.set_facecolor(color_map[ny])
				patch.set_alpha(0.7)
			ax_d.axhline(0, color="black", lw=0.8, ls="--")
			ax_d.set_xticklabels(
				[f"{ny}Y train\n(n={len(df[df['train_years'] == ny])})" for ny in all_train_years],
				fontsize=9,
			)
			ax_d.set_title("Distribucion de Alpha por Longitud de Train", fontweight="bold")
			ax_d.set_ylabel("Alpha anualizado (%)")
			ax_d.grid(axis="y", alpha=0.3)
		else:
			ax_d.axis("off")

		patches = [mpatches.Patch(color=color_map[ny], label=f"{ny}Y train") for ny in all_train_years]
		fig.legend(handles=patches, loc="lower center", ncol=len(all_train_years),
				   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

		fig.tight_layout(rect=[0, 0.03, 1, 0.98])
		plot_path = plots_path / "folds_results.png"
		fig.savefig(plot_path, dpi=150, bbox_inches="tight")
		plt.close(fig)
		log.info(f"[Backtester] Folds results plot -> {plot_path}")

		return csv_path, plot_path
