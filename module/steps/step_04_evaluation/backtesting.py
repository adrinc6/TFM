"""Consolidated evaluation backtesting utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from environment import (
    PORTFOLIO_MIN_SCORE,
    SCORE_WEIGHTED_PORTFOLIO,
    PORTFOLIO_MAX_STOCKS_PER_SECTOR,
    PORTFOLIO_MAX_STOCK_WEIGHT,
)
from module.common.performance_metrics import compute_all_metrics
from module.common.portfolio_optimization import hrp_weights, risk_parity_weights, robust_markowitz_weights
from module.steps.step_04_evaluation.strategy import simulate_tp_sl
log = logging.getLogger(__name__)


class WalkForwardBacktester:
	"""Walk-forward backtesting con paso trimestral y test de 1 quarter."""

	def __init__(
		self,
		train_years: int = 3,
		test_quarters: int = 1,
		risk_free: float = 0.04,
		results_dir: str = "results/backtest",
		strategy_dir: str | None = None,
		top_n_stocks: int = 10,
		long_only: bool = True,
		score_weighted: bool = SCORE_WEIGHTED_PORTFOLIO,
		portfolio_optimizer: str = "hrp",
	):
		self.train_years = train_years
		self.test_quarters = test_quarters
		self.risk_free = risk_free
		self.results_dir = Path(results_dir)
		self.results_dir.mkdir(parents=True, exist_ok=True)
		if strategy_dir is None:
			self.strategy_dir = self.results_dir.parent / "strategy"
		else:
			self.strategy_dir = Path(strategy_dir)
		self.strategy_dir.mkdir(parents=True, exist_ok=True)
		self.top_n_stocks = top_n_stocks
		self.long_only = long_only
		self.score_weighted = score_weighted
		self.portfolio_optimizer = str(portfolio_optimizer).strip().lower()
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

	@staticmethod
	def _select_with_sector_cap(
		ordered: pd.DataFrame,
		target_n: int,
		sector_cap: int,
		min_stocks: int,
	) -> pd.DataFrame:
		"""Build a ranked shortlist enforcing a max tickers-per-sector cap."""
		if ordered.empty or target_n <= 0:
			return ordered.head(0).copy()
		if sector_cap <= 0 or "sector" not in ordered.columns:
			return ordered.head(target_n).copy()

		selected_rows = []
		sector_counts: Dict[str, int] = {}
		for _, row in ordered.iterrows():
			sector = str(row.get("sector", "Unknown"))
			if sector_counts.get(sector, 0) >= sector_cap:
				continue
			selected_rows.append(row)
			sector_counts[sector] = sector_counts.get(sector, 0) + 1
			if len(selected_rows) >= target_n:
				break

		# If constraints are too tight, relax to keep investability floor.
		if len(selected_rows) < min_stocks:
			existing = {str(r.get("ticker")) for r in selected_rows}
			for _, row in ordered.iterrows():
				tk = str(row.get("ticker"))
				if tk in existing:
					continue
				selected_rows.append(row)
				existing.add(tk)
				if len(selected_rows) >= min_stocks:
					break

		if not selected_rows:
			return ordered.head(min(target_n, len(ordered))).copy()
		return pd.DataFrame(selected_rows).head(target_n).copy()

	@staticmethod
	def _apply_weight_cap(weights: np.ndarray, cap: float) -> np.ndarray:
		"""Apply per-position max weight and renormalize."""
		if len(weights) == 0:
			return weights
		cap = float(cap)
		if cap <= 0.0:
			return weights
		# If cap is infeasible (e.g., cap=0.10 with 8 names => max total 0.80 < 1.00),
		# fall back to equal weights to keep a valid invested portfolio.
		if cap * len(weights) < 1.0:
			log.warning(
				"[Backtester] Weight cap %.3f is infeasible for %d positions; falling back to equal weights.",
				cap,
				len(weights),
			)
			return np.ones(len(weights)) / len(weights)

		w = weights.astype(float).copy()
		w = w / max(w.sum(), 1e-12)

		for _ in range(20):
			over = w > cap
			if not over.any():
				break
			excess = float((w[over] - cap).sum())
			w[over] = cap
			under = ~over
			under_sum = float(w[under].sum())
			if under_sum <= 0:
				w = np.ones(len(w)) / len(w)
				break
			w[under] += excess * (w[under] / under_sum)

		w = np.minimum(w, cap)
		w = w / max(w.sum(), 1e-12)
		return w

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
		ranking_col = "ev" if "ev" in predictions_df.columns else "score"
		selection_threshold = 0.0 if ranking_col == "ev" else PORTFOLIO_MIN_SCORE
		ordered = predictions_df.sort_values(ranking_col, ascending=False).copy()
		if "score" not in ordered.columns:
			# Keep downstream interfaces stable: "score" is the normalized ranking field.
			log.info("[Backtester] score column missing; using %s as ranking alias.", ranking_col)
			ordered["score"] = pd.to_numeric(ordered[ranking_col], errors="coerce")
		sector_cap = int(PORTFOLIO_MAX_STOCKS_PER_SECTOR)
		qualified = ordered[pd.to_numeric(ordered[ranking_col], errors="coerce") >= float(selection_threshold)]
		if len(qualified) >= min_stocks:
			# Tomar hasta top_n pero garantizar al menos min_stocks
			n_take = max(min(len(qualified), self.top_n_stocks), min_stocks)
			top_df = self._select_with_sector_cap(
				ordered=ordered,
				target_n=n_take,
				sector_cap=sector_cap,
				min_stocks=min_stocks,
			)[["ticker", "score"] + (["sector"] if "sector" in ordered.columns else [])].copy()
			log.info(
				f"[Backtester] {period_id}: {len(qualified)} tickers superaron umbral {selection_threshold:.2f} "
				f"â†’ selecting target={n_take}, final={len(top_df)} (min={min_stocks}, sector_cap={sector_cap})"
			)
		else:
			# Compressed regime: keep relative selection by ranking.
			top_df = self._select_with_sector_cap(
				ordered=ordered,
				target_n=self.top_n_stocks,
				sector_cap=sector_cap,
				min_stocks=min_stocks,
			)[["ticker", "score"] + (["sector"] if "sector" in ordered.columns else [])].copy()
			top_score = float(ordered["score"].iloc[0]) if len(ordered) > 0 else float("nan")
			bottom_idx = min(self.top_n_stocks, len(ordered)) - 1
			bottom_score = float(ordered["score"].iloc[bottom_idx]) if bottom_idx >= 0 else float("nan")
			log.warning(
				f"[Backtester] {period_id}: solo {len(qualified)} tickers superaron umbral {selection_threshold:.2f} "
				f"â†’ seleccionando top-{len(top_df)} por ranking con restricciones (sector_cap={sector_cap}) "
				f"(scores: {top_score:.3f} .. {bottom_score:.3f})"
			)
		top = top_df["ticker"].tolist()
		if not top:
			log.warning(f"[Backtester] {period_id}: no tickers available â€” analysis skipped.")
			return {}

		daily_returns = []
		tickers_with_prices = []
		ticker_returns = {}
		bench_period = benchmark.loc[test_start:test_end].dropna()
		if len(bench_period) < 2:
			log.warning(f"[Backtester] {period_id}: insufficient benchmark data â€” analysis skipped.")
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
			cc = _get_close_column(prices)
			period = prices.loc[test_start:actual_end, cc]
			if len(period) < 2:
				continue
			ret = period.pct_change().dropna()
			daily_returns.append(ret)
			tickers_with_prices.append(ticker)
			ticker_returns[ticker] = round(float((1 + ret).prod() - 1), 6)

		if not daily_returns:
			return {}

		# Portfolio optimization from pre-entry trailing returns (no look-ahead).
		N = len(tickers_with_prices)
		weights = np.ones(N) / N
		if N > 1:
			trail_start = pd.Timestamp(test_start) - pd.DateOffset(days=252)
			trail_mat = {}
			for tk in tickers_with_prices:
				px = prices_dict.get(tk)
				if px is None or px.empty:
					continue
				cc = _get_close_column(px)
				s = px.loc[trail_start:test_start, cc].pct_change().dropna()
				if len(s) >= 20:
					trail_mat[tk] = s

			if len(trail_mat) >= 2:
				trail_df = pd.concat(trail_mat, axis=1).dropna(how="all")
				if not trail_df.empty:
					if self.portfolio_optimizer == "hrp":
						w_map = hrp_weights(trail_df)
					elif self.portfolio_optimizer == "risk_parity":
						w_map = risk_parity_weights(trail_df)
					elif self.portfolio_optimizer == "markowitz":
						w_map = robust_markowitz_weights(trail_df)
					else:
						w_map = {}

					if w_map:
						w_arr = np.array([float(w_map.get(tk, 0.0)) for tk in tickers_with_prices], dtype=float)
						if np.isfinite(w_arr).all() and w_arr.sum() > 0:
							weights = w_arr / w_arr.sum()
							log.info(f"[Backtester] {period_id}: optimizer={self.portfolio_optimizer} pre-entry trailing window applied")

			# Fallback to score-weighted/equal if optimizer did not produce valid weights.
			if (not np.isfinite(weights).all()) or float(weights.sum()) <= 0:
				weights = np.ones(N) / N
			if np.allclose(weights, np.ones(N) / N) and self.score_weighted and N > 1:
				scores_arr = (
					top_df.set_index("ticker")
					.loc[tickers_with_prices]["score"]
					.values.astype(float)
				)
				ratio = 1.0 + N / 10.0
				s_min, s_max = scores_arr.min(), scores_arr.max()
				if s_max > s_min:
					raw = 1.0 + (ratio - 1.0) * (scores_arr - s_min) / (s_max - s_min)
				else:
					raw = np.ones(N)
				weights = raw / raw.sum()
		weights = self._apply_weight_cap(weights, float(PORTFOLIO_MAX_STOCK_WEIGHT))

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
		weighting_mode = f"{self.portfolio_optimizer}" if self.portfolio_optimizer else ("softmax(score)" if self.score_weighted else "equiponderado")
		if float(PORTFOLIO_MAX_STOCK_WEIGHT) > 0:
			weighting_mode = f"{weighting_mode}+cap({float(PORTFOLIO_MAX_STOCK_WEIGHT):.2f})"
		log.info(f"[Backtester] {period_id} â€” cartera final ({weighting_mode}, {len(tickers_with_prices)} stocks):")
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
			f"[Backtester] {period_id} â€” {len(top)} stocks | "
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

		path = self.strategy_dir / "backtest_summary.json"
		with open(path, "w") as f:
			json.dump(summary, f, indent=2, default=str)

		pd.DataFrame({
			"strategy": self.all_strategy_returns,
			"benchmark": self.all_benchmark_returns,
		}).to_csv(self.strategy_dir / "returns_series.csv")

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

		csv_path = self.strategy_dir / "folds_results.csv"
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
			f"Resultados Walk-Forward â€” {len(df)} folds | "
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
		ax_a.set_title("Alpha por Fold (Estrategia âˆ’ Benchmark)", fontweight="bold")
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

@dataclass
class Position:
    ticker: str
    shares: float


def _get_close_column(prices: pd.DataFrame, fallback_col_idx: int = 3) -> str:
    """Return the close price column name for a price DataFrame.

    Uses 'Close' if present, otherwise falls back to the column at *fallback_col_idx*
    (default 3, corresponding to the C in standard OHLCV layout).
    """
    if "Close" in prices.columns:
        # Canonical path: explicit close column.
        return "Close"
    if len(prices.columns) > fallback_col_idx:
        # OHLCV fallback: index 3 corresponds to "Close" in standard
        # layouts (Open=0, High=1, Low=2, Close=3, Volume=4).
        return str(prices.columns[fallback_col_idx])
    if len(prices.columns) == 1:
        # Single-column inputs are treated as pre-extracted close series.
        return str(prices.columns[0])
    # Defensive fallback for non-standard multi-column inputs.
    return str(prices.columns[-1])


def _extract_close_series(price_obj) -> pd.Series:
    if price_obj is None:
        return pd.Series(dtype=float)
    if isinstance(price_obj, pd.Series):
        s = pd.to_numeric(price_obj, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    if isinstance(price_obj, pd.DataFrame) and not price_obj.empty:
        close_col = _get_close_column(price_obj)
        s = price_obj[close_col]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    return pd.Series(dtype=float)


def _resolve_exec_date(price_series: pd.Series, requested: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the first available trading date on or after *requested*."""
    if price_series is None or price_series.empty:
        return None
    ts = pd.Timestamp(requested)
    candidates = price_series.index[price_series.index >= ts]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[0])


def _resolve_exec_date_on_or_before(price_series: pd.Series, requested: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the last available trading date on or before *requested*.

    Used for exit resolution so that a position can be closed at the last
    available price even when the data does not extend all the way to the
    requested exit date (e.g. recent folds whose price download is not yet
    complete, or tickers that were delisted shortly before exit_req).
    """
    if price_series is None or price_series.empty:
        return None
    ts = pd.Timestamp(requested)
    candidates = price_series.index[price_series.index <= ts]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[-1])


def _price_on_or_before(price_series: pd.Series, date: pd.Timestamp) -> Optional[float]:
    if price_series is None or price_series.empty:
        return None
    subset = price_series.loc[price_series.index <= pd.Timestamp(date)]
    if subset.empty:
        return None
    px = float(subset.iloc[-1])
    if not np.isfinite(px) or px <= 0:
        return None
    return px


def _build_weights(selected_tickers: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not selected_tickers:
        return {}
    if not weights:
        eq = 1.0 / len(selected_tickers)
        return {t: eq for t in selected_tickers}

    raw = {t: float(weights.get(t, 0.0)) for t in selected_tickers}
    total = float(sum(max(v, 0.0) for v in raw.values()))
    if total <= 0:
        eq = 1.0 / len(selected_tickers)
        return {t: eq for t in selected_tickers}
    return {t: max(v, 0.0) / total for t, v in raw.items()}


def simulate_fold_usd(
    *,
    fold_id: str,
    prices_dict: Dict[str, object],
    selected_tickers: List[str],
    weights: Optional[Dict[str, float]],
    entry_date_requested,
    exit_date_requested,
    starting_cash_usd: float,
    transaction_fee_usd: float,
    slippage_pct: float,
    allow_fractional_shares: bool = True,
    tp_sl_plan_by_ticker: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Simulate one long-only fold in USD with deterministic trade rules."""
    entry_req = pd.Timestamp(entry_date_requested)
    exit_req = pd.Timestamp(exit_date_requested)

    tickers = [str(t) for t in selected_tickers]
    tickers = [t for t in tickers if t in prices_dict]

    per_ticker_close: Dict[str, pd.Series] = {t: _extract_close_series(prices_dict.get(t)) for t in tickers}
    # Entry: first trading day ON or AFTER entry_req (do not buy before the entry date).
    entry_candidates = {t: _resolve_exec_date(s, entry_req) for t, s in per_ticker_close.items() if not s.empty}
    # Exit: last trading day ON or BEFORE exit_req (default date-based path).
    exit_candidates = {t: _resolve_exec_date_on_or_before(s, exit_req) for t, s in per_ticker_close.items() if not s.empty}

    valid_tickers = [
        t for t in tickers
        if (
            entry_candidates.get(t) is not None
            and exit_candidates.get(t) is not None
            # Exit must be strictly after entry so the holding period is positive.
            and exit_candidates[t] > entry_candidates[t]
        )
    ]

    if not valid_tickers:
        empty_trades = pd.DataFrame(columns=[
            "fold_id", "datetime", "action", "ticker", "raw_price", "exec_price", "shares",
            "notional_usd", "fee_usd", "slippage_pct", "entry_date_requested", "entry_date_used",
            "exit_date_requested", "exit_date_used", "reason",
        ])
        empty_equity = pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
        return {
            "trades_df": empty_trades,
            "equity_curve_df": empty_equity,
            "fold_summary": {
                "fold_id": fold_id,
                "starting_capital_usd": float(starting_cash_usd),
                "ending_capital_usd": float(starting_cash_usd),
                "pnl_usd": 0.0,
                "pnl_pct": 0.0,
                "total_fees_usd": 0.0,
                "n_buys": 0,
                "n_sells": 0,
                "n_ffill_days": 0,
                "entry_date_used": None,
                "exit_date_used": None,
                "entry_gap_days": None,
                "exit_gap_days": None,
                "n_selected_tickers": 0,
                "leakage_tainted": False,
            },
            "selected_tickers_used": [],
            "weights_used": {},
            "missing_tickers": tickers,
            "missing_reasons": {t: "missing_entry_or_exit_price" for t in tickers},
        }

    # Use the LATEST first-available entry date so that every valid ticker
    # actually has a price on (or before) entry_used.  With min() a ticker whose
    # first available date is later than entry_used would be bought at a stale
    # pre-entry_req price returned by _price_on_or_before.
    entry_used = max(entry_candidates[t] for t in valid_tickers)

    # Default common exit date (used when TP/SL plan is not provided).
    exit_used = min(exit_candidates[t] for t in valid_tickers)
    if exit_used < entry_used:
        exit_used = max(exit_candidates[t] for t in valid_tickers)

    entry_gap_days = int((entry_used - entry_req).days)
    exit_gap_days = int((exit_used - exit_req).days)

    # Per-ticker TP/SL exits (main strategy path). Baselines keep date-based exit.
    per_ticker_exit_date: Dict[str, pd.Timestamp] = {}
    per_ticker_exit_reason: Dict[str, str] = {}
    per_ticker_outcome: Dict[str, str] = {}
    per_ticker_days_to_outcome: Dict[str, int] = {}
    per_ticker_tp: Dict[str, float] = {}
    per_ticker_sl: Dict[str, float] = {}

    if tp_sl_plan_by_ticker:
        max_exit_seen = entry_used
        for ticker in valid_tickers:
            plan = tp_sl_plan_by_ticker.get(ticker, {}) if tp_sl_plan_by_ticker else {}
            tp_pct = float(plan.get("tp_pct", np.nan))
            sl_pct = float(plan.get("sl_pct", np.nan))
            max_holding_days = int(plan.get("max_holding_days", 90))
            per_ticker_tp[ticker] = tp_pct
            per_ticker_sl[ticker] = sl_pct

            s = per_ticker_close.get(ticker, pd.Series(dtype=float))
            if (not np.isfinite(tp_pct)) or (not np.isfinite(sl_pct)) or s.empty:
                dt_exit = exit_candidates.get(ticker, exit_used)
                per_ticker_exit_date[ticker] = pd.Timestamp(dt_exit)
                per_ticker_exit_reason[ticker] = "time_exit"
                per_ticker_outcome[ticker] = "NONE"
                per_ticker_days_to_outcome[ticker] = int(max((pd.Timestamp(dt_exit) - entry_used).days, 0))
                max_exit_seen = max(max_exit_seen, pd.Timestamp(dt_exit))
                continue

            sim = simulate_tp_sl(
                ticker=ticker,
                prices=s,
                entry_date=entry_used,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                max_holding_days=max_holding_days,
            )
            out = str(sim.get("outcome", "NONE")).upper()
            out_dt = sim.get("outcome_date")
            if pd.isna(out_dt):
                dt_exit = exit_candidates.get(ticker, exit_used)
            else:
                dt_exit = _resolve_exec_date_on_or_before(s, pd.Timestamp(out_dt))
                if dt_exit is None:
                    dt_exit = exit_candidates.get(ticker, exit_used)
            dt_exit = pd.Timestamp(dt_exit)
            if dt_exit <= entry_used:
                dt_exit = exit_candidates.get(ticker, exit_used)
                dt_exit = pd.Timestamp(dt_exit)

            per_ticker_exit_date[ticker] = dt_exit
            per_ticker_outcome[ticker] = out
            per_ticker_days_to_outcome[ticker] = int(sim.get("days_to_outcome", max((dt_exit - entry_used).days, 0)))
            if out == "TP":
                per_ticker_exit_reason[ticker] = "tp_hit"
            elif out == "SL":
                per_ticker_exit_reason[ticker] = "sl_hit"
            else:
                per_ticker_exit_reason[ticker] = "time_exit"
            max_exit_seen = max(max_exit_seen, dt_exit)

        exit_used = max_exit_seen

    weights_used = _build_weights(valid_tickers, weights)
    cash = float(starting_cash_usd)
    # Snapshot initial capital so every ticker's allocation is based on the
    # same starting amount, not on the cash remaining after prior purchases.
    initial_capital = float(starting_cash_usd)
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, object]] = []
    total_fees = 0.0
    n_ffill_days = 0

    for ticker in valid_tickers:
        s = per_ticker_close[ticker]
        raw_price = _price_on_or_before(s, entry_used)
        if raw_price is None:
            continue
        exec_price = raw_price * (1.0 + float(slippage_pct))
        allocated_cash = float(initial_capital * weights_used.get(ticker, 0.0))

        if allocated_cash < float(transaction_fee_usd):
            trades.append({
                "fold_id": fold_id,
                "datetime": entry_used,
                "action": "BUY",
                "ticker": ticker,
                "raw_price": raw_price,
                "exec_price": exec_price,
                "shares": 0.0,
                "notional_usd": 0.0,
                "fee_usd": float(transaction_fee_usd),
                "slippage_pct": float(slippage_pct),
                "entry_date_requested": entry_req,
                "entry_date_used": entry_used,
                "exit_date_requested": exit_req,
                "exit_date_used": exit_used,
                "reason": "insufficient_cash_for_fee",
            })
            continue

        if not allow_fractional_shares:
            shares = max(0.0, float(np.floor((allocated_cash - float(transaction_fee_usd)) / exec_price)))
        else:
            shares = max(0.0, float((allocated_cash - float(transaction_fee_usd)) / exec_price))

        notional = float(shares * exec_price)
        if shares > 0:
            cash -= (notional + float(transaction_fee_usd))
            total_fees += float(transaction_fee_usd)
            positions[ticker] = Position(ticker=ticker, shares=shares)

        trades.append({
            "fold_id": fold_id,
            "datetime": entry_used,
            "action": "BUY",
            "ticker": ticker,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "shares": shares,
            "notional_usd": notional,
            "fee_usd": float(transaction_fee_usd),
            "slippage_pct": float(slippage_pct),
            "entry_date_requested": entry_req,
            "entry_date_used": entry_used,
            "exit_date_requested": exit_req,
            "exit_date_used": exit_used,
            "reason": "rebalance_entry",
            "tp_pct": float(per_ticker_tp.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
            "sl_pct": float(per_ticker_sl.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
            "tp_sl_outcome": str(per_ticker_outcome.get(ticker, "NONE")) if tp_sl_plan_by_ticker else "",
            "days_to_outcome": float(per_ticker_days_to_outcome.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
        })

    all_dates = sorted(set().union(*[
        [d for d in s.index if (d >= entry_used and d <= exit_used)] for s in per_ticker_close.values() if not s.empty
    ]))
    if not all_dates:
        all_dates = [entry_used, exit_used]

    last_prices: Dict[str, float] = {}
    equity_rows: List[Dict[str, object]] = []

    for dt in all_dates:
        dt_ts = pd.Timestamp(dt)

        # Execute sells scheduled for this date before marking end-of-day equity.
        for ticker, pos in list(positions.items()):
            sell_dt = per_ticker_exit_date.get(ticker, exit_used) if tp_sl_plan_by_ticker else exit_used
            if pd.Timestamp(sell_dt) != dt_ts:
                continue
            s = per_ticker_close.get(ticker, pd.Series(dtype=float))
            raw_price = _price_on_or_before(s, dt_ts)
            if raw_price is None:
                continue
            exec_price = raw_price * (1.0 - float(slippage_pct))
            notional = float(pos.shares * exec_price)
            cash += (notional - float(transaction_fee_usd))
            total_fees += float(transaction_fee_usd)
            trades.append({
                "fold_id": fold_id,
                "datetime": dt_ts,
                "action": "SELL",
                "ticker": ticker,
                "raw_price": raw_price,
                "exec_price": exec_price,
                "shares": float(pos.shares),
                "notional_usd": notional,
                "fee_usd": float(transaction_fee_usd),
                "slippage_pct": float(slippage_pct),
                "entry_date_requested": entry_req,
                "entry_date_used": entry_used,
                "exit_date_requested": exit_req,
                "exit_date_used": dt_ts,
                "reason": per_ticker_exit_reason.get(ticker, "rebalance_exit") if tp_sl_plan_by_ticker else "rebalance_exit",
                "tp_pct": float(per_ticker_tp.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
                "sl_pct": float(per_ticker_sl.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
                "tp_sl_outcome": str(per_ticker_outcome.get(ticker, "NONE")) if tp_sl_plan_by_ticker else "",
                "days_to_outcome": float(per_ticker_days_to_outcome.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
            })
            positions.pop(ticker, None)

        pos_value = 0.0
        for ticker, pos in positions.items():
            s = per_ticker_close.get(ticker, pd.Series(dtype=float))
            px = _price_on_or_before(s, dt_ts)
            if px is None:
                px = last_prices.get(ticker)
                if px is not None:
                    n_ffill_days += 1
            else:
                last_prices[ticker] = px
            if px is not None:
                pos_value += float(pos.shares * px)

        equity_rows.append({
            "date": dt_ts,
            "equity_usd": float(cash + pos_value),
            "cash_usd": float(cash),
            "positions_value_usd": float(pos_value),
        })

    # Safety close: if any position remained unsold, close at the common exit date.
    for ticker, pos in list(positions.items()):
        s = per_ticker_close.get(ticker, pd.Series(dtype=float))
        raw_price = _price_on_or_before(s, exit_used)
        if raw_price is None:
            continue
        exec_price = raw_price * (1.0 - float(slippage_pct))
        notional = float(pos.shares * exec_price)
        cash += (notional - float(transaction_fee_usd))
        total_fees += float(transaction_fee_usd)
        trades.append({
            "fold_id": fold_id,
            "datetime": exit_used,
            "action": "SELL",
            "ticker": ticker,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "shares": float(pos.shares),
            "notional_usd": notional,
            "fee_usd": float(transaction_fee_usd),
            "slippage_pct": float(slippage_pct),
            "entry_date_requested": entry_req,
            "entry_date_used": entry_used,
            "exit_date_requested": exit_req,
            "exit_date_used": exit_used,
            "reason": "rebalance_exit_safety",
            "tp_pct": float(per_ticker_tp.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
            "sl_pct": float(per_ticker_sl.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
            "tp_sl_outcome": str(per_ticker_outcome.get(ticker, "NONE")) if tp_sl_plan_by_ticker else "",
            "days_to_outcome": float(per_ticker_days_to_outcome.get(ticker, np.nan)) if tp_sl_plan_by_ticker else np.nan,
        })
        positions.pop(ticker, None)

    final_equity = float(cash)
    if equity_rows:
        equity_rows[-1]["cash_usd"] = float(cash)
        equity_rows[-1]["positions_value_usd"] = 0.0
        equity_rows[-1]["equity_usd"] = float(cash)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["datetime", "action", "ticker"]).reset_index(drop=True)

    equity_curve_df = pd.DataFrame(equity_rows)
    if not equity_curve_df.empty:
        equity_curve_df = equity_curve_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    fold_summary = {
        "fold_id": fold_id,
        "starting_capital_usd": float(starting_cash_usd),
        "ending_capital_usd": float(final_equity),
        "pnl_usd": float(final_equity - float(starting_cash_usd)),
        "pnl_pct": float((final_equity / float(starting_cash_usd) - 1.0) if float(starting_cash_usd) > 0 else 0.0),
        "total_fees_usd": float(total_fees),
        "n_buys": int((trades_df["action"] == "BUY").sum()) if not trades_df.empty else 0,
        "n_sells": int((trades_df["action"] == "SELL").sum()) if not trades_df.empty else 0,
        "n_ffill_days": int(n_ffill_days),
        "entry_date_used": str(entry_used.date()),
        "exit_date_used": str(exit_used.date()),
        "entry_gap_days": int(entry_gap_days),
        "exit_gap_days": int(exit_gap_days),
        "n_selected_tickers": int(len(valid_tickers)),
        "leakage_tainted": False,
    }

    missing_tickers = [t for t in tickers if t not in valid_tickers]
    missing_reasons = {t: "missing_entry_or_exit_price" for t in missing_tickers}

    return {
        "trades_df": trades_df,
        "equity_curve_df": equity_curve_df,
        "fold_summary": fold_summary,
        "selected_tickers_used": valid_tickers,
        "weights_used": weights_used,
        "missing_tickers": missing_tickers,
        "missing_reasons": missing_reasons,
    }


def compute_max_drawdown_from_equity(equity_curve_df: pd.DataFrame) -> float:
    if equity_curve_df is None or equity_curve_df.empty or "equity_usd" not in equity_curve_df.columns:
        return 0.0
    s = pd.to_numeric(equity_curve_df["equity_usd"], errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = (s - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def to_daily_returns_from_equity(equity_curve_df: pd.DataFrame) -> pd.Series:
    if equity_curve_df is None or equity_curve_df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric(equity_curve_df["equity_usd"], errors="coerce").values,
        index=pd.to_datetime(equity_curve_df["date"]),
    ).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.pct_change().dropna()

