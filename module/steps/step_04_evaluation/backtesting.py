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
    TP_SL_GRACE_PERIOD_FRACTION,
    TP_SL_TRAILING_REVIEW_DAYS,
    ENABLE_BUY_HOLD_COUNTERFACTUAL,
    BUY_HOLD_EXIT_ON_LAST_AVAILABLE_PRICE,
    EXPORT_TP_SL_VS_BUY_HOLD,
    ENABLE_TP_SL_RESEARCH_VARIANTS,
    TP_SL_VARIANT_MODE,
    TP_SL_HYBRID_TRAILING_MIN_PCT,
    TP_SL_HYBRID_TRAILING_MAX_PCT,
    TP_SL_HYBRID_PROFIT_REVIEW_DAYS,
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
		self.counterfactual_fold_rows: List[Dict] = []
		self.counterfactual_ticker_rows: List[Dict] = []
		self.hybrid_vs_base_fold_rows: List[Dict] = []
		self.hybrid_vs_base_ticker_rows: List[Dict] = []
		self.trailing_dynamics_rows: List[Dict] = []
		self.all_strategy_returns = pd.Series(dtype=float)
		self.all_buy_hold_returns = pd.Series(dtype=float)
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


	@staticmethod
	def _counterfactual_return_series(period: pd.Series) -> pd.Series:
		"""Daily Buy & Hold returns for one ticker without TP/SL/trailing exits."""
		if period is None or len(period) < 2:
			return pd.Series(dtype=float)
		return pd.to_numeric(period, errors="coerce").dropna().pct_change().dropna()

	@staticmethod
	def _counterfactual_exit_date(period: pd.Series, target_exit: pd.Timestamp) -> pd.Timestamp | None:
		"""Resolve Buy & Hold exit date as target date or last available price."""
		if period is None or period.empty:
			return None
		target = pd.Timestamp(target_exit)
		on_or_before = period.index[period.index <= target]
		if len(on_or_before) == 0:
			return None
		if BUY_HOLD_EXIT_ON_LAST_AVAILABLE_PRICE:
			return pd.Timestamp(on_or_before[-1])
		# Strict mode still resolves to the last tradable date on/before target if
		# the target is a weekend/holiday, but skips tickers ending before target.
		if pd.Timestamp(on_or_before[-1]) < target and period.index.max() < target:
			return None
		return pd.Timestamp(on_or_before[-1])

	@staticmethod
	def _variant_mode_label() -> str:
		mode = str(TP_SL_VARIANT_MODE).strip().lower()
		if not ENABLE_TP_SL_RESEARCH_VARIANTS:
			return "base"
		return mode if mode in {"base", "vol_adjusted", "momentum_adjusted", "regime_adjusted", "hybrid_learned"} else "base"


	@staticmethod
	def _hybrid_exit_from_close(
		close_series: pd.Series,
		*,
		ticker: str,
		entry_date: pd.Timestamp,
		fallback_exit_date: pd.Timestamp,
		tp_pct: float,
		sl_pct: float,
		max_holding_days: int,
		trailing_stop_pct: float,
		momentum: float = 0.0,
		volatility: float = 0.12,
		regime: str = "Neutral",
		trail_events: Optional[List[Dict]] = None,
	) -> tuple[pd.Timestamp, str, int, float, float]:
		"""Hybrid learned TP/SL exit with dynamic profit-protection trailing."""
		fallback = pd.Timestamp(fallback_exit_date)
		entry_ts = pd.Timestamp(entry_date)
		if close_series is None or close_series.empty:
			return fallback, "time_exit", int(max((fallback - entry_ts).days, 0)), float("nan"), float("nan")
		prices = pd.to_numeric(close_series, errors="coerce").dropna().sort_index()
		prices.index = pd.to_datetime(prices.index)
		entry_candidates = prices.index[prices.index >= entry_ts]
		if len(entry_candidates) == 0:
			return fallback, "time_exit", int(max((fallback - entry_ts).days, 0)), float("nan"), float("nan")
		actual_entry = pd.Timestamp(entry_candidates[0])
		entry_price = float(prices.loc[actual_entry])
		if not np.isfinite(entry_price) or entry_price <= 0:
			return fallback, "time_exit", int(max((fallback - entry_ts).days, 0)), float("nan"), float("nan")
		tp_price = entry_price * (1.0 + float(tp_pct))
		initial_sl_price = entry_price * (1.0 - float(sl_pct))
		expiry_ts = min(actual_entry + pd.Timedelta(days=int(max_holding_days)), fallback)
		window = prices.loc[(prices.index > actual_entry) & (prices.index <= expiry_ts)]
		grace = int(max_holding_days * float(TP_SL_GRACE_PERIOD_FRACTION))
		regime_l = str(regime).lower()
		is_risk_on = ("risk-on" in regime_l) or ("bull" in regime_l)
		is_risk_off = ("risk-off" in regime_l) or ("bear" in regime_l)
		volatility = float(volatility) if np.isfinite(float(volatility)) and float(volatility) > 0 else 0.12
		base_trail = float(trailing_stop_pct) if np.isfinite(float(trailing_stop_pct)) and float(trailing_stop_pct) > 0 else 0.12
		base_trail = float(np.clip(base_trail, float(TP_SL_HYBRID_TRAILING_MIN_PCT), float(TP_SL_HYBRID_TRAILING_MAX_PCT)))
		peak_price = entry_price
		trailing_active = False
		trailing_stop = initial_sl_price
		trailing_initial = float("nan")
		last_review_day = 0

		def _dynamic_distance(days_elapsed: int, price: float) -> float:
			profit = max(price / entry_price - 1.0, 0.0)
			dist = base_trail
			dist *= 1.0 + 0.35 * max(min(volatility - 0.12, 0.20), -0.06)
			if momentum > 0 and is_risk_on:
				dist *= 1.08
			if is_risk_off:
				dist *= 0.88
			# As profit gets very large, gradually tighten but never below min.
			if profit >= 0.40:
				dist *= 0.88
			elif profit >= 0.25:
				dist *= 0.95
			return float(np.clip(dist, float(TP_SL_HYBRID_TRAILING_MIN_PCT), float(TP_SL_HYBRID_TRAILING_MAX_PCT)))

		def _record(event_type: str, dt: pd.Timestamp, price: float, stop: float, distance: float) -> None:
			if trail_events is None:
				return
			trail_events.append({
				"ticker": ticker,
				"entry_date": pd.Timestamp(actual_entry).date(),
				"entry_price": round(entry_price, 4),
				"tp_price": round(tp_price, 4),
				"sl_price_original": round(initial_sl_price, 4),
				"event_type": event_type,
				"event_date": pd.Timestamp(dt).date(),
				"days_from_entry": int((pd.Timestamp(dt) - actual_entry).days),
				"price": round(price, 4),
				"peak_price": round(peak_price, 4),
				"trailing_stop": round(stop, 4),
				"trailing_distance_pct": round(distance, 6),
				"return_pct": round((price / entry_price - 1.0) * 100.0, 2),
			})

		for dt, px in window.items():
			price = float(px)
			days_elapsed = int((pd.Timestamp(dt) - actual_entry).days)
			if price > peak_price:
				peak_price = price
			if days_elapsed < grace:
				continue
			if not trailing_active:
				if price >= tp_price:
					trailing_active = True
					last_review_day = days_elapsed
					dist = _dynamic_distance(days_elapsed, price)
					trailing_stop = max(price * (1.0 - dist), initial_sl_price)
					trailing_initial = trailing_stop
					_record("HYBRID_PROFIT_PROTECTION_ACTIVATED", dt, price, trailing_stop, dist)
				elif price <= initial_sl_price:
					_record("HYBRID_EXIT_SL", dt, price, trailing_stop, base_trail)
					return pd.Timestamp(dt), "sl_hit", days_elapsed, trailing_initial, trailing_stop
			else:
				dist = _dynamic_distance(days_elapsed, price)
				candidate_stop = peak_price * (1.0 - dist)
				if candidate_stop > trailing_stop and (days_elapsed - last_review_day) >= int(TP_SL_HYBRID_PROFIT_REVIEW_DAYS):
					trailing_stop = max(trailing_stop, candidate_stop, initial_sl_price)
					last_review_day = days_elapsed
					_record("HYBRID_TRAILING_RECALCULATED", dt, price, trailing_stop, dist)
				if price <= trailing_stop:
					_record("HYBRID_EXIT_TRAIL", dt, price, trailing_stop, dist)
					return pd.Timestamp(dt), "tp_hit", days_elapsed, trailing_initial, trailing_stop
		if not window.empty:
			last_dt = pd.Timestamp(window.index[-1])
			return last_dt, "time_exit", int((last_dt - actual_entry).days), trailing_initial, trailing_stop
		return actual_entry, "time_exit", 0, trailing_initial, trailing_stop

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
		if "selection_score" in predictions_df.columns:
			ranking_col = "selection_score"
		elif "ev" in predictions_df.columns:
			ranking_col = "ev"
		else:
			ranking_col = "score"
		selection_threshold = 0.0 if ranking_col in {"ev", "selection_score"} else PORTFOLIO_MIN_SCORE
		ordered = predictions_df.sort_values(ranking_col, ascending=False).copy()
		if "score" not in ordered.columns:
			# Keep downstream interfaces stable: "score" is the normalized ranking field.
			log.info("[Backtester] score column missing; using %s as ranking alias.", ranking_col)
			ordered["score"] = pd.to_numeric(ordered[ranking_col], errors="coerce")
		sector_cap = int(PORTFOLIO_MAX_STOCKS_PER_SECTOR)
		selection_cols = ["ticker", "score"] + [c for c in ["sector", "regime_state", "regime"] if c in ordered.columns]
		qualified = ordered[pd.to_numeric(ordered[ranking_col], errors="coerce") >= float(selection_threshold)]
		if len(qualified) >= min_stocks:
			# Tomar hasta top_n pero garantizar al menos min_stocks
			n_take = max(min(len(qualified), self.top_n_stocks), min_stocks)
			top_df = self._select_with_sector_cap(
				ordered=ordered,
				target_n=n_take,
				sector_cap=sector_cap,
				min_stocks=min_stocks,
			)[selection_cols].copy()
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
			)[selection_cols].copy()
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
		base_daily_returns = []
		hybrid_daily_returns = []
		tickers_with_prices = []
		ticker_returns = {}
		base_ticker_returns: Dict[str, float] = {}
		hybrid_ticker_returns: Dict[str, float] = {}
		base_ticker_exit_reasons: Dict[str, str] = {}
		hybrid_ticker_exit_reasons: Dict[str, str] = {}
		base_ticker_exit_dates: Dict[str, str] = {}
		hybrid_ticker_exit_dates: Dict[str, str] = {}
		hybrid_trailing_initial: Dict[str, float] = {}
		hybrid_trailing_final: Dict[str, float] = {}
		ticker_exit_dates: Dict[str, str] = {}
		ticker_exit_reasons: Dict[str, str] = {}
		ticker_days_to_outcome: Dict[str, int] = {}
		fold_trail_events: List[Dict] = []
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

		if "max_holding_days" in predictions_df.columns:
			mh_series = pd.to_numeric(predictions_df.get("max_holding_days"), errors="coerce").dropna()
			if len(mh_series) > 0 and float(mh_series.median()) > 0:
				max_holding_days_default = int(float(mh_series.median()))
			else:
				max_holding_days_default = int(max((pd.Timestamp(actual_end) - pd.Timestamp(test_start)).days, 1))
		else:
			max_holding_days_default = int(max((pd.Timestamp(actual_end) - pd.Timestamp(test_start)).days, 1))

		plan_by_ticker = (
			predictions_df.set_index("ticker")
			if "ticker" in predictions_df.columns else pd.DataFrame()
		)
		last_exit_dt = pd.Timestamp(test_start)

		for ticker in top:
			if ticker not in prices_dict:
				continue
			prices = prices_dict[ticker]
			cc = _get_close_column(prices)
			period = pd.to_numeric(prices.loc[test_start:actual_end, cc], errors="coerce").dropna()
			if len(period) < 2:
				continue

			tp_pct = np.nan
			sl_pct = np.nan
			trailing_stop_pct_ticker = 0.0
			hybrid_tp_pct = np.nan
			hybrid_sl_pct = np.nan
			hybrid_trailing_pct = np.nan
			hybrid_momentum = 0.0
			hybrid_volatility = 0.12
			hybrid_regime = "Neutral"
			max_holding_days = max_holding_days_default
			if not plan_by_ticker.empty and ticker in plan_by_ticker.index:
				row = plan_by_ticker.loc[ticker]
				if isinstance(row, pd.DataFrame):
					row = row.iloc[-1]
				tp_pct = float(pd.to_numeric(row.get("tp_pct", np.nan), errors="coerce"))
				sl_pct = float(pd.to_numeric(row.get("sl_pct", np.nan), errors="coerce"))
				trailing_val = pd.to_numeric(row.get("trailing_stop_pct", 0.0), errors="coerce")
				trailing_stop_pct_ticker = float(trailing_val) if np.isfinite(float(trailing_val if trailing_val is not None else 0.0)) else 0.0
				hybrid_tp_pct = float(pd.to_numeric(row.get("hybrid_tp_pct", tp_pct), errors="coerce"))
				hybrid_sl_pct = float(pd.to_numeric(row.get("hybrid_sl_pct", sl_pct), errors="coerce"))
				hybrid_trailing_pct = float(pd.to_numeric(row.get("hybrid_trailing_stop_pct", trailing_stop_pct_ticker), errors="coerce"))
				hybrid_momentum = float(pd.to_numeric(row.get("hybrid_momentum_used", 0.0), errors="coerce")) if "hybrid_momentum_used" in row.index else 0.0
				hybrid_volatility = float(pd.to_numeric(row.get("hybrid_volatility_used", 0.12), errors="coerce")) if "hybrid_volatility_used" in row.index else 0.12
				hybrid_regime = str(row.get("hybrid_regime", row.get("regime_state", "Neutral")))
				mh = pd.to_numeric(row.get("max_holding_days", max_holding_days_default), errors="coerce")
				if np.isfinite(mh) and float(mh) > 0:
					max_holding_days = int(float(mh))
			if not np.isfinite(hybrid_tp_pct):
				hybrid_tp_pct = tp_pct
			if not np.isfinite(hybrid_sl_pct):
				hybrid_sl_pct = sl_pct
			if not np.isfinite(hybrid_trailing_pct):
				hybrid_trailing_pct = trailing_stop_pct_ticker

			base_trail_events: List[Dict] = []
			base_exit_dt, base_exit_reason, base_days_to_outcome = _tp_sl_exit_from_close(
				period,
				entry_date=pd.Timestamp(test_start),
				fallback_exit_date=pd.Timestamp(actual_end),
				tp_pct=float(tp_pct),
				sl_pct=float(sl_pct),
				max_holding_days=max_holding_days,
				trailing_stop_pct=trailing_stop_pct_ticker,
				trail_events=base_trail_events,
			)
			hybrid_trail_events: List[Dict] = []
			hybrid_exit_dt, hybrid_exit_reason, hybrid_days_to_outcome, hybrid_initial_stop, hybrid_final_stop = self._hybrid_exit_from_close(
				period,
				ticker=ticker,
				entry_date=pd.Timestamp(test_start),
				fallback_exit_date=pd.Timestamp(actual_end),
				tp_pct=float(hybrid_tp_pct),
				sl_pct=float(hybrid_sl_pct),
				max_holding_days=max_holding_days,
				trailing_stop_pct=hybrid_trailing_pct,
				momentum=hybrid_momentum,
				volatility=hybrid_volatility,
				regime=hybrid_regime,
				trail_events=hybrid_trail_events,
			)
			use_hybrid_as_main = self._variant_mode_label() == "hybrid_learned"
			exit_dt = hybrid_exit_dt if use_hybrid_as_main else base_exit_dt
			exit_reason = hybrid_exit_reason if use_hybrid_as_main else base_exit_reason
			days_to_outcome = hybrid_days_to_outcome if use_hybrid_as_main else base_days_to_outcome
			ticker_trail_events = hybrid_trail_events if use_hybrid_as_main else base_trail_events
			for ev in ticker_trail_events:
				ev["ticker"] = ticker
				ev["fold_id"] = period_id
				fold_trail_events.append(ev)
			exit_slice = period.loc[period.index <= pd.Timestamp(exit_dt)]
			if len(exit_slice) < 2:
				continue

			ret = exit_slice.pct_change().dropna()
			base_slice = period.loc[period.index <= pd.Timestamp(base_exit_dt)]
			hybrid_slice = period.loc[period.index <= pd.Timestamp(hybrid_exit_dt)]
			base_ret = base_slice.pct_change().dropna() if len(base_slice) >= 2 else pd.Series(dtype=float)
			hybrid_ret = hybrid_slice.pct_change().dropna() if len(hybrid_slice) >= 2 else pd.Series(dtype=float)
			daily_returns.append(ret)
			base_daily_returns.append(base_ret)
			hybrid_daily_returns.append(hybrid_ret)
			tickers_with_prices.append(ticker)
			ticker_returns[ticker] = round(float((1 + ret).prod() - 1), 6)
			base_ticker_returns[ticker] = round(float((1 + base_ret).prod() - 1), 6) if len(base_ret) else 0.0
			hybrid_ticker_returns[ticker] = round(float((1 + hybrid_ret).prod() - 1), 6) if len(hybrid_ret) else 0.0
			base_ticker_exit_reasons[ticker] = str(base_exit_reason)
			hybrid_ticker_exit_reasons[ticker] = str(hybrid_exit_reason)
			base_ticker_exit_dates[ticker] = str(pd.Timestamp(base_exit_dt).date())
			hybrid_ticker_exit_dates[ticker] = str(pd.Timestamp(hybrid_exit_dt).date())
			hybrid_trailing_initial[ticker] = float(hybrid_initial_stop) if np.isfinite(hybrid_initial_stop) else np.nan
			hybrid_trailing_final[ticker] = float(hybrid_final_stop) if np.isfinite(hybrid_final_stop) else np.nan
			ticker_exit_dates[ticker] = str(pd.Timestamp(exit_dt).date())
			ticker_exit_reasons[ticker] = str(exit_reason)
			ticker_days_to_outcome[ticker] = int(days_to_outcome)
			if pd.Timestamp(exit_dt) > last_exit_dt:
				last_exit_dt = pd.Timestamp(exit_dt)

		if not daily_returns:
			return {}

		# Write per-fold trailing stop evolution CSV
		if fold_trail_events:
			trail_cols = [
				"fold_id", "ticker", "entry_date", "entry_price",
				"tp_price", "sl_price_original",
				"event_type", "event_date", "days_from_entry",
				"price", "peak_price", "trailing_stop", "return_pct",
			]
			trail_df = pd.DataFrame(fold_trail_events)
			for col in trail_cols:
				if col not in trail_df.columns:
					trail_df[col] = None
			trail_df = trail_df[trail_cols]
			csv_path = self.results_dir / f"portfolio_trail_{period_id}.csv"
			trail_df.to_csv(csv_path, index=False, float_format="%.4f")
			log.info("[Trail] %s: %d events → %s", period_id, len(trail_df), csv_path.name)

		bench_eval = benchmark.loc[test_start:last_exit_dt].dropna()
		if len(bench_eval) >= 2:
			bench_period = bench_eval
		actual_eval_end = pd.Timestamp(bench_period.index.max())

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
		top_indexed = top_df.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
		sector_by_ticker = {t: str(top_indexed.loc[t, "sector"]) for t in tickers_with_prices if "sector" in top_indexed.columns and t in top_indexed.index}
		regime_col = "regime_state" if "regime_state" in top_indexed.columns else ("regime" if "regime" in top_indexed.columns else None)
		regime_by_ticker = {t: str(top_indexed.loc[t, regime_col]) for t in tickers_with_prices if regime_col is not None and t in top_indexed.index}

		buy_hold_daily_returns: List[pd.Series] = []
		buy_hold_ticker_returns: Dict[str, float] = {}
		buy_hold_exit_dates: Dict[str, str] = {}
		buy_hold_days_held: Dict[str, int] = {}
		if ENABLE_BUY_HOLD_COUNTERFACTUAL:
			for ticker in tickers_with_prices:
				prices = prices_dict.get(ticker)
				if prices is None:
					buy_hold_daily_returns.append(pd.Series(dtype=float))
					continue
				cc = _get_close_column(prices)
				full_period = pd.to_numeric(prices.loc[test_start:test_end, cc], errors="coerce").dropna()
				exit_dt_bh = self._counterfactual_exit_date(full_period, pd.Timestamp(test_end))
				if exit_dt_bh is None:
					buy_hold_daily_returns.append(pd.Series(dtype=float))
					continue
				bh_period = full_period.loc[full_period.index <= exit_dt_bh]
				bh_ret = self._counterfactual_return_series(bh_period)
				buy_hold_daily_returns.append(bh_ret)
				buy_hold_ticker_returns[ticker] = round(float((1 + bh_ret).prod() - 1), 6) if len(bh_ret) else 0.0
				buy_hold_exit_dates[ticker] = str(pd.Timestamp(exit_dt_bh).date())
				buy_hold_days_held[ticker] = int(max((pd.Timestamp(exit_dt_bh) - pd.Timestamp(test_start)).days, 0))

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

		base_tp_sl_aligned = pd.Series(dtype=float)
		hybrid_tp_sl_aligned = pd.Series(dtype=float)
		base_tp_sl_metrics: Dict[str, float] = {}
		hybrid_tp_sl_metrics: Dict[str, float] = {}
		if base_daily_returns and hybrid_daily_returns:
			base_mat = pd.concat(base_daily_returns, axis=1)
			base_mat.columns = tickers_with_prices
			hybrid_mat = pd.concat(hybrid_daily_returns, axis=1)
			hybrid_mat.columns = tickers_with_prices
			base_tp_sl = (base_mat * weights).sum(axis=1).dropna()
			hybrid_tp_sl = (hybrid_mat * weights).sum(axis=1).dropna()
			base_idx = base_tp_sl.index.intersection(benchmark.loc[test_start:test_end].dropna().index)
			hybrid_idx = hybrid_tp_sl.index.intersection(benchmark.loc[test_start:test_end].dropna().index)
			base_tp_sl_aligned = base_tp_sl.loc[base_idx]
			hybrid_tp_sl_aligned = hybrid_tp_sl.loc[hybrid_idx]
			if len(base_tp_sl_aligned):
				base_tp_sl_metrics = compute_all_metrics(base_tp_sl_aligned, self.risk_free, "base_tp_sl")
			if len(hybrid_tp_sl_aligned):
				hybrid_tp_sl_metrics = compute_all_metrics(hybrid_tp_sl_aligned, self.risk_free, "hybrid_tp_sl")

		buy_hold_returns = pd.Series(dtype=float)
		buy_hold_aligned = pd.Series(dtype=float)
		buy_hold_bench_aligned = pd.Series(dtype=float)
		buy_hold_metrics: Dict[str, float] = {}
		buy_hold_alpha = float("nan")
		tp_sl_minus_buy_hold = float("nan")
		tickers_tp_sl_better: List[str] = []
		tickers_tp_sl_worse: List[str] = []
		if ENABLE_BUY_HOLD_COUNTERFACTUAL and buy_hold_daily_returns:
			bh_matrix = pd.concat(buy_hold_daily_returns, axis=1)
			bh_matrix.columns = tickers_with_prices
			buy_hold_returns = (bh_matrix * weights).sum(axis=1).dropna()
			bh_common_idx = buy_hold_returns.index.intersection(benchmark.loc[test_start:test_end].dropna().index)
			buy_hold_aligned = buy_hold_returns.loc[bh_common_idx]
			buy_hold_bench_aligned = benchmark.loc[bh_common_idx].dropna()
			buy_hold_aligned = buy_hold_aligned.loc[buy_hold_aligned.index.intersection(buy_hold_bench_aligned.index)]
			buy_hold_bench_aligned = buy_hold_bench_aligned.loc[buy_hold_aligned.index]
			if len(buy_hold_aligned) >= 1:
				buy_hold_metrics = compute_all_metrics(buy_hold_aligned, self.risk_free, "buy_hold")
				bh_bench_metrics = compute_all_metrics(buy_hold_bench_aligned, self.risk_free, "buy_hold_benchmark")
				buy_hold_alpha = (
					buy_hold_metrics.get("buy_hold_cumulative_return", 0.0)
					- bh_bench_metrics.get("buy_hold_benchmark_cumulative_return", 0.0)
				)
				tp_sl_minus_buy_hold = strat_metrics["strategy_cumulative_return"] - buy_hold_metrics.get("buy_hold_cumulative_return", 0.0)
			for ticker in tickers_with_prices:
				delta = float(ticker_returns.get(ticker, np.nan)) - float(buy_hold_ticker_returns.get(ticker, np.nan))
				if np.isfinite(delta) and delta > 0:
					tickers_tp_sl_better.append(ticker)
				elif np.isfinite(delta) and delta < 0:
					tickers_tp_sl_worse.append(ticker)

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

		price_days = int(len(bench_period))
		exit_reason_counts = {k: int(v) for k, v in pd.Series(ticker_exit_reasons).value_counts().to_dict().items()}
		avg_holding_days_tp_sl = float(np.mean(list(ticker_days_to_outcome.values()))) if ticker_days_to_outcome else 0.0
		avg_holding_days_buy_hold = float(np.mean(list(buy_hold_days_held.values()))) if buy_hold_days_held else 0.0
		return_buy_hold = buy_hold_metrics.get("buy_hold_cumulative_return", float("nan"))
		return_base_tp_sl = base_tp_sl_metrics.get("base_tp_sl_cumulative_return", float("nan"))
		return_hybrid_tp_sl = hybrid_tp_sl_metrics.get("hybrid_tp_sl_cumulative_return", float("nan"))
		hybrid_minus_base = return_hybrid_tp_sl - return_base_tp_sl if np.isfinite(return_hybrid_tp_sl) and np.isfinite(return_base_tp_sl) else float("nan")
		fold_result = {
			"fold": fold_id,
			"year_quarter": year_quarter,
			"train_years": train_years_int,
			"train_start": str(train_start.date()) if train_start is not None else None,
			"test_start": str(test_start.date()),
			"test_end": str(actual_eval_end.date()),
			"test_end_target": str(test_end.date()),
			"price_days": price_days,
			"selected_tickers": top,
			"buy_hold_selected_tickers": list(tickers_with_prices),
			"n_stocks": len(top),
			**strat_metrics,
			**bench_metrics,
			"alpha": alpha,
			"excess_sharpe": excess_sharpe,
			"return_tp_sl": strat_metrics.get("strategy_cumulative_return", 0.0),
			"return_buy_hold": return_buy_hold,
			"return_base_tp_sl": return_base_tp_sl,
			"return_hybrid_tp_sl": return_hybrid_tp_sl,
			"hybrid_minus_base": hybrid_minus_base,
			"alpha_tp_sl_vs_benchmark": alpha,
			"alpha_buy_hold_vs_benchmark": buy_hold_alpha,
			"tp_sl_minus_buy_hold": tp_sl_minus_buy_hold,
			"sharpe_tp_sl": strat_metrics.get("strategy_sharpe", 0.0),
			"sharpe_buy_hold": buy_hold_metrics.get("buy_hold_sharpe", float("nan")),
			"max_drawdown_tp_sl": strat_metrics.get("strategy_max_drawdown", 0.0),
			"max_drawdown_buy_hold": buy_hold_metrics.get("buy_hold_max_drawdown", float("nan")),
			"win_rate_tp_sl": strat_metrics.get("strategy_win_rate", 0.0),
			"win_rate_buy_hold": buy_hold_metrics.get("buy_hold_win_rate", float("nan")),
			"avg_holding_days_tp_sl": avg_holding_days_tp_sl,
			"avg_holding_days_buy_hold": avg_holding_days_buy_hold,
			"exit_reason_counts": exit_reason_counts,
			"tickers_tp_sl_better_than_buy_hold": tickers_tp_sl_better,
			"tickers_tp_sl_worse_than_buy_hold": tickers_tp_sl_worse,
			"tp_sl_variant_mode": self._variant_mode_label(),
			**buy_hold_metrics,
			**base_tp_sl_metrics,
			**hybrid_tp_sl_metrics,
			"ticker_returns": ticker_returns_sorted,
			"ticker_weights": ticker_weights,
			"ticker_exit_dates": ticker_exit_dates,
			"ticker_exit_reasons": ticker_exit_reasons,
			"ticker_days_to_outcome": ticker_days_to_outcome,
			"base_ticker_returns": dict(sorted(base_ticker_returns.items(), key=lambda x: x[1], reverse=True)),
			"hybrid_ticker_returns": dict(sorted(hybrid_ticker_returns.items(), key=lambda x: x[1], reverse=True)),
			"base_ticker_exit_reasons": base_ticker_exit_reasons,
			"hybrid_ticker_exit_reasons": hybrid_ticker_exit_reasons,
			"base_ticker_exit_dates": base_ticker_exit_dates,
			"hybrid_ticker_exit_dates": hybrid_ticker_exit_dates,
			"hybrid_trailing_initial": hybrid_trailing_initial,
			"hybrid_trailing_final": hybrid_trailing_final,
			"buy_hold_ticker_returns": dict(sorted(buy_hold_ticker_returns.items(), key=lambda x: x[1], reverse=True)),
			"buy_hold_exit_dates": buy_hold_exit_dates,
			"buy_hold_days_held": buy_hold_days_held,
			"weighting_mode": weighting_mode,
			"_ticker_price_series": ticker_price_series,
			"_strat_price_series": (1 + strat_aligned).cumprod(),
			"_buy_hold_price_series": (1 + buy_hold_aligned).cumprod() if len(buy_hold_aligned) else pd.Series(dtype=float),
			"_bench_price_series": (1 + bench_aligned).cumprod(),
		}

		self.all_strategy_returns = pd.concat([self.all_strategy_returns, strat_aligned])
		if len(buy_hold_aligned):
			self.all_buy_hold_returns = pd.concat([self.all_buy_hold_returns, buy_hold_aligned])
		self.all_benchmark_returns = pd.concat([self.all_benchmark_returns, bench_aligned])

		if ENABLE_BUY_HOLD_COUNTERFACTUAL:
			self._record_counterfactual_rows(
				fold_result=None,
				period_id=period_id,
				fold_id=fold_id,
				year_quarter=year_quarter,
				train_years_int=train_years_int,
				test_start=test_start,
				actual_eval_end=actual_eval_end,
				test_end=test_end,
				tickers=tickers_with_prices,
				weights=ticker_weights,
				tp_sl_returns=ticker_returns,
				buy_hold_returns=buy_hold_ticker_returns,
				tp_sl_exit_dates=ticker_exit_dates,
				buy_hold_exit_dates=buy_hold_exit_dates,
				tp_sl_days=ticker_days_to_outcome,
				buy_hold_days=buy_hold_days_held,
				exit_reasons=ticker_exit_reasons,
				sector_by_ticker=sector_by_ticker,
				regime_by_ticker=regime_by_ticker,
				fold_metrics=fold_result,
			)

		self._record_hybrid_vs_base_rows(
			period_id=period_id,
			fold_id=fold_id,
			year_quarter=year_quarter,
			train_years_int=train_years_int,
			test_start=test_start,
			tickers=tickers_with_prices,
			weights=ticker_weights,
			base_returns=base_ticker_returns,
			hybrid_returns=hybrid_ticker_returns,
			buy_hold_returns=buy_hold_ticker_returns,
			base_exit_reasons=base_ticker_exit_reasons,
			hybrid_exit_reasons=hybrid_ticker_exit_reasons,
			base_exit_dates=base_ticker_exit_dates,
			hybrid_exit_dates=hybrid_ticker_exit_dates,
			hybrid_trailing_initial=hybrid_trailing_initial,
			hybrid_trailing_final=hybrid_trailing_final,
			sector_by_ticker=sector_by_ticker,
			regime_by_ticker=regime_by_ticker,
			fold_metrics=fold_result,
		)

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


	def _record_counterfactual_rows(
		self,
		*,
		fold_result: Optional[Dict],
		period_id: str,
		fold_id,
		year_quarter: str,
		train_years_int: int,
		test_start,
		actual_eval_end,
		test_end,
		tickers: List[str],
		weights: Dict[str, float],
		tp_sl_returns: Dict[str, float],
		buy_hold_returns: Dict[str, float],
		tp_sl_exit_dates: Dict[str, str],
		buy_hold_exit_dates: Dict[str, str],
		tp_sl_days: Dict[str, int],
		buy_hold_days: Dict[str, int],
		exit_reasons: Dict[str, str],
		sector_by_ticker: Optional[Dict[str, str]] = None,
		regime_by_ticker: Optional[Dict[str, str]] = None,
		fold_metrics: Dict | None = None,
	) -> None:
		"""Append export-ready TP/SL vs Buy & Hold rows without touching ML scores."""
		fm = fold_metrics or fold_result or {}
		self.counterfactual_fold_rows.append({
			"fold": fold_id,
			"fold_id": period_id,
			"year_quarter": year_quarter,
			"train_years": train_years_int,
			"entry_date": str(pd.Timestamp(test_start).date()),
			"tp_sl_exit_end": str(pd.Timestamp(actual_eval_end).date()),
			"buy_hold_target_exit": str(pd.Timestamp(test_end).date()),
			"n_tickers": int(len(tickers)),
			"return_tp_sl": fm.get("return_tp_sl", fm.get("strategy_cumulative_return", np.nan)),
			"return_buy_hold": fm.get("return_buy_hold", np.nan),
			"benchmark_return": fm.get("benchmark_cumulative_return", np.nan),
			"alpha_tp_sl_vs_benchmark": fm.get("alpha_tp_sl_vs_benchmark", fm.get("alpha", np.nan)),
			"alpha_buy_hold_vs_benchmark": fm.get("alpha_buy_hold_vs_benchmark", np.nan),
			"tp_sl_minus_buy_hold": fm.get("tp_sl_minus_buy_hold", np.nan),
			"sharpe_tp_sl": fm.get("sharpe_tp_sl", fm.get("strategy_sharpe", np.nan)),
			"sharpe_buy_hold": fm.get("sharpe_buy_hold", np.nan),
			"max_drawdown_tp_sl": fm.get("max_drawdown_tp_sl", fm.get("strategy_max_drawdown", np.nan)),
			"max_drawdown_buy_hold": fm.get("max_drawdown_buy_hold", np.nan),
			"win_rate_tp_sl": fm.get("win_rate_tp_sl", fm.get("strategy_win_rate", np.nan)),
			"win_rate_buy_hold": fm.get("win_rate_buy_hold", np.nan),
			"avg_holding_days_tp_sl": fm.get("avg_holding_days_tp_sl", np.nan),
			"avg_holding_days_buy_hold": fm.get("avg_holding_days_buy_hold", np.nan),
			"exit_reason_counts": json.dumps(fm.get("exit_reason_counts", {}), sort_keys=True),
			"tickers_tp_sl_better_than_buy_hold": ",".join(fm.get("tickers_tp_sl_better_than_buy_hold", [])),
			"tickers_tp_sl_worse_than_buy_hold": ",".join(fm.get("tickers_tp_sl_worse_than_buy_hold", [])),
			"tp_sl_variant_mode": fm.get("tp_sl_variant_mode", self._variant_mode_label()),
		})

		for ticker in tickers:
			tp_ret = float(tp_sl_returns.get(ticker, np.nan))
			bh_ret = float(buy_hold_returns.get(ticker, np.nan))
			delta = tp_ret - bh_ret if np.isfinite(tp_ret) and np.isfinite(bh_ret) else np.nan
			self.counterfactual_ticker_rows.append({
				"fold": fold_id,
				"fold_id": period_id,
				"year_quarter": year_quarter,
				"ticker": ticker,
				"sector": (sector_by_ticker or {}).get(ticker),
				"regime": (regime_by_ticker or {}).get(ticker),
				"entry_date": str(pd.Timestamp(test_start).date()),
				"weight": float(weights.get(ticker, np.nan)),
				"return_tp_sl": tp_ret,
				"return_buy_hold": bh_ret,
				"tp_sl_minus_buy_hold": delta,
				"tp_sl_exit_date": tp_sl_exit_dates.get(ticker),
				"buy_hold_exit_date": buy_hold_exit_dates.get(ticker),
				"tp_sl_exit_reason": exit_reasons.get(ticker),
				"buy_hold_exit_reason": "annual_horizon_or_last_available",
				"holding_days_tp_sl": int(tp_sl_days.get(ticker, 0)),
				"holding_days_buy_hold": int(buy_hold_days.get(ticker, 0)),
				"tp_sl_improved": bool(np.isfinite(delta) and delta > 0),
			})



	def _record_hybrid_vs_base_rows(
		self,
		*,
		period_id: str,
		fold_id,
		year_quarter: str,
		train_years_int: int,
		test_start,
		tickers: List[str],
		weights: Dict[str, float],
		base_returns: Dict[str, float],
		hybrid_returns: Dict[str, float],
		buy_hold_returns: Dict[str, float],
		base_exit_reasons: Dict[str, str],
		hybrid_exit_reasons: Dict[str, str],
		base_exit_dates: Dict[str, str],
		hybrid_exit_dates: Dict[str, str],
		hybrid_trailing_initial: Dict[str, float],
		hybrid_trailing_final: Dict[str, float],
		sector_by_ticker: Optional[Dict[str, str]] = None,
		regime_by_ticker: Optional[Dict[str, str]] = None,
		fold_metrics: Dict | None = None,
	) -> None:
		"""Append export-ready TP/SL hybrid vs base comparison rows."""
		self.hybrid_vs_base_fold_rows.append({
			"fold": fold_id,
			"fold_id": period_id,
			"year_quarter": year_quarter,
			"train_years": train_years_int,
			"entry_date": str(pd.Timestamp(test_start).date()),
			"return_base": fold_metrics.get("return_base_tp_sl", np.nan),
			"return_hybrid": fold_metrics.get("return_hybrid_tp_sl", np.nan),
			"return_buy_hold": fold_metrics.get("return_buy_hold", np.nan),
			"hybrid_minus_base": fold_metrics.get("hybrid_minus_base", np.nan),
			"hybrid_minus_buy_hold": (fold_metrics.get("return_hybrid_tp_sl", np.nan) - fold_metrics.get("return_buy_hold", np.nan)) if np.isfinite(fold_metrics.get("return_hybrid_tp_sl", np.nan)) and np.isfinite(fold_metrics.get("return_buy_hold", np.nan)) else np.nan,
			"sharpe_base": fold_metrics.get("base_tp_sl_sharpe", np.nan),
			"sharpe_hybrid": fold_metrics.get("hybrid_tp_sl_sharpe", np.nan),
			"max_drawdown_base": fold_metrics.get("base_tp_sl_max_drawdown", np.nan),
			"max_drawdown_hybrid": fold_metrics.get("hybrid_tp_sl_max_drawdown", np.nan),
			"tp_sl_variant_mode": fold_metrics.get("tp_sl_variant_mode", self._variant_mode_label()),
		})
		for ticker in tickers:
			base_ret = float(base_returns.get(ticker, np.nan))
			hybrid_ret = float(hybrid_returns.get(ticker, np.nan))
			bh_ret = float(buy_hold_returns.get(ticker, np.nan))
			delta = hybrid_ret - base_ret if np.isfinite(hybrid_ret) and np.isfinite(base_ret) else np.nan
			self.hybrid_vs_base_ticker_rows.append({
				"fold": fold_id,
				"fold_id": period_id,
				"year_quarter": year_quarter,
				"ticker": ticker,
				"sector": (sector_by_ticker or {}).get(ticker),
				"regime": (regime_by_ticker or {}).get(ticker),
				"entry_date": str(pd.Timestamp(test_start).date()),
				"weight": float(weights.get(ticker, np.nan)),
				"return_base": base_ret,
				"return_hybrid": hybrid_ret,
				"return_buy_hold": bh_ret,
				"hybrid_minus_base": delta,
				"hybrid_minus_buy_hold": hybrid_ret - bh_ret if np.isfinite(hybrid_ret) and np.isfinite(bh_ret) else np.nan,
				"exit_reason_base": base_exit_reasons.get(ticker),
				"exit_reason_hybrid": hybrid_exit_reasons.get(ticker),
				"exit_date_base": base_exit_dates.get(ticker),
				"exit_date_hybrid": hybrid_exit_dates.get(ticker),
				"trailing_stop_initial": hybrid_trailing_initial.get(ticker),
				"trailing_stop_final": hybrid_trailing_final.get(ticker),
			})

	def _export_counterfactual_artifacts(self, summary_payload: Dict) -> None:
		"""Export TP/SL vs Buy & Hold CSV, JSON and comparison charts."""
		if not EXPORT_TP_SL_VS_BUY_HOLD and not self.hybrid_vs_base_fold_rows:
			return

		self.strategy_dir.mkdir(parents=True, exist_ok=True)
		fold_df = pd.DataFrame(self.counterfactual_fold_rows)
		ticker_df = pd.DataFrame(self.counterfactual_ticker_rows)
		if EXPORT_TP_SL_VS_BUY_HOLD and not fold_df.empty:
			fold_csv = self.strategy_dir / "tp_sl_vs_buy_hold_by_fold.csv"
			ticker_csv = self.strategy_dir / "tp_sl_vs_buy_hold_by_ticker.csv"
			fold_df.to_csv(fold_csv, index=False, float_format="%.6f")
			ticker_df.to_csv(ticker_csv, index=False, float_format="%.6f")
		if self.hybrid_vs_base_fold_rows:
			pd.DataFrame(self.hybrid_vs_base_fold_rows).to_csv(self.strategy_dir / "tp_sl_hybrid_vs_base_by_fold.csv", index=False, float_format="%.6f")
		if self.hybrid_vs_base_ticker_rows:
			pd.DataFrame(self.hybrid_vs_base_ticker_rows).to_csv(self.strategy_dir / "tp_sl_hybrid_vs_base_by_ticker.csv", index=False, float_format="%.6f")


		if self.hybrid_vs_base_fold_rows or self.hybrid_vs_base_ticker_rows:
			robustness: Dict[str, object] = {}
			hf = pd.DataFrame(self.hybrid_vs_base_fold_rows)
			ht = pd.DataFrame(self.hybrid_vs_base_ticker_rows)
			if not hf.empty and "hybrid_minus_base" in hf.columns:
				delta = pd.to_numeric(hf["hybrid_minus_base"], errors="coerce")
				robustness["folds_where_hybrid_wins"] = hf.loc[delta > 0, [c for c in ["fold_id", "year_quarter", "hybrid_minus_base", "return_base", "return_hybrid"] if c in hf.columns]].to_dict(orient="records")
				robustness["folds_where_hybrid_loses"] = hf.loc[delta < 0, [c for c in ["fold_id", "year_quarter", "hybrid_minus_base", "return_base", "return_hybrid"] if c in hf.columns]].to_dict(orient="records")
			if not ht.empty and "hybrid_minus_base" in ht.columns:
				delta_t = pd.to_numeric(ht["hybrid_minus_base"], errors="coerce")
				robustness["tickers_where_hybrid_wins"] = ht.loc[delta_t > 0, [c for c in ["fold_id", "ticker", "sector", "regime", "hybrid_minus_base", "return_base", "return_hybrid"] if c in ht.columns]].to_dict(orient="records")
				robustness["tickers_where_hybrid_loses"] = ht.loc[delta_t < 0, [c for c in ["fold_id", "ticker", "sector", "regime", "hybrid_minus_base", "return_base", "return_hybrid"] if c in ht.columns]].to_dict(orient="records")
				for group_col, key in [("sector", "by_sector"), ("regime", "by_regime")]:
					if group_col in ht.columns:
						grp_rows = []
						for name, grp in ht.groupby(group_col, dropna=False):
							gdelta = pd.to_numeric(grp["hybrid_minus_base"], errors="coerce")
							grp_rows.append({
								group_col: str(name),
								"n": int(len(grp)),
								"hybrid_wins": int((gdelta > 0).sum()),
								"hybrid_loses": int((gdelta < 0).sum()),
								"mean_hybrid_minus_base": float(gdelta.mean()) if gdelta.notna().any() else np.nan,
							})
						robustness[key] = grp_rows
			with open(self.strategy_dir / "tp_sl_hybrid_robustness_summary.json", "w", encoding="utf-8") as f:
				json.dump(robustness, f, indent=2, default=str)

		trail_files = sorted(self.results_dir.glob("portfolio_trail_*.csv"))
		trail_parts = []
		for trail_file in trail_files:
			try:
				trail_parts.append(pd.read_csv(trail_file))
			except Exception:
				continue
		if trail_parts:
			pd.concat(trail_parts, ignore_index=True).to_csv(self.strategy_dir / "trailing_dynamics_by_ticker.csv", index=False)

		summary_json = self.strategy_dir / "tp_sl_vs_buy_hold_summary.json"
		with open(summary_json, "w", encoding="utf-8") as f:
			json.dump(summary_payload, f, indent=2, default=str)

		if fold_df.empty:
			return

		try:
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt

			labels = fold_df["year_quarter"].astype(str).tolist() if "year_quarter" in fold_df else fold_df["fold_id"].astype(str).tolist()
			x = np.arange(len(fold_df))

			# Equity curve: concatenated daily return streams.
			fig, ax = plt.subplots(figsize=(12, 6))
			if not self.all_strategy_returns.empty:
				(1 + self.all_strategy_returns).cumprod().plot(ax=ax, label="TP/SL", lw=2)
			if not self.all_buy_hold_returns.empty:
				(1 + self.all_buy_hold_returns).cumprod().plot(ax=ax, label="Buy & Hold 12M", lw=2)
			if not self.all_benchmark_returns.empty:
				(1 + self.all_benchmark_returns).cumprod().plot(ax=ax, label="Benchmark", lw=1.6, alpha=0.8)
			ax.set_title("Equity Curve: TP/SL vs Buy & Hold vs Benchmark")
			ax.set_ylabel("Growth of $1")
			ax.grid(alpha=0.3)
			ax.legend()
			fig.tight_layout()
			fig.savefig(self.strategy_dir / "equity_curve_tp_sl_vs_buy_hold_vs_benchmark.png", dpi=150, bbox_inches="tight")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(12, 6))
			w = 0.38
			ax.bar(x - w / 2, fold_df["alpha_tp_sl_vs_benchmark"].astype(float) * 100, width=w, label="TP/SL alpha")
			ax.bar(x + w / 2, fold_df["alpha_buy_hold_vs_benchmark"].astype(float) * 100, width=w, label="Buy & Hold alpha")
			ax.axhline(0, color="black", lw=0.8)
			ax.set_title("Alpha por fold: TP/SL vs Buy & Hold")
			ax.set_ylabel("Alpha (%)")
			ax.set_xticks(x)
			ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
			ax.grid(axis="y", alpha=0.3)
			ax.legend()
			fig.tight_layout()
			fig.savefig(self.strategy_dir / "alpha_by_fold_tp_sl_vs_buy_hold.png", dpi=150, bbox_inches="tight")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(12, 5))
			delta = fold_df["tp_sl_minus_buy_hold"].astype(float)
			colors = ["#2E7D32" if v >= 0 else "#C62828" for v in delta]
			ax.bar(x, delta * 100, color=colors, alpha=0.85)
			ax.axhline(0, color="black", lw=0.8)
			ax.set_title("Diferencia por fold: TP/SL - Buy & Hold")
			ax.set_ylabel("Diferencia de retorno (%)")
			ax.set_xticks(x)
			ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
			ax.grid(axis="y", alpha=0.3)
			fig.tight_layout()
			fig.savefig(self.strategy_dir / "tp_sl_minus_buy_hold_by_fold.png", dpi=150, bbox_inches="tight")
			plt.close(fig)

			if not ticker_df.empty and "tp_sl_exit_reason" in ticker_df.columns:
				fig, ax = plt.subplots(figsize=(8, 5))
				counts = ticker_df["tp_sl_exit_reason"].fillna("unknown").value_counts()
				counts.plot(kind="bar", ax=ax, color="#1976D2", alpha=0.85)
				ax.set_title("Distribución de exits TP/SL")
				ax.set_ylabel("Número de posiciones")
				ax.set_xlabel("Exit reason")
				ax.grid(axis="y", alpha=0.3)
				fig.tight_layout()
				fig.savefig(self.strategy_dir / "tp_sl_exit_distribution.png", dpi=150, bbox_inches="tight")
				plt.close(fig)
		except Exception as ex:
			log.warning("[Counterfactual] Could not export comparison plots (%s)", ex)

	def summarize(self) -> Dict:
		if not self.fold_results:
			log.warning("[Backtester] Sin folds completados.")
			return {}

		global_strat = compute_all_metrics(self.all_strategy_returns, self.risk_free, "global_strategy")
		global_bench = compute_all_metrics(self.all_benchmark_returns, self.risk_free, "global_benchmark")
		global_buy_hold = (
			compute_all_metrics(self.all_buy_hold_returns, self.risk_free, "global_buy_hold")
			if ENABLE_BUY_HOLD_COUNTERFACTUAL and not self.all_buy_hold_returns.empty
			else {}
		)
		folds_df = pd.DataFrame(self.fold_results)
		mean_alpha = float(folds_df["alpha"].mean()) if "alpha" in folds_df else 0.0
		pct_alpha_pos = float((folds_df["alpha"] > 0).mean()) if "alpha" in folds_df else 0.0
		tp_sl_wins_vs_bh = int((pd.to_numeric(folds_df.get("tp_sl_minus_buy_hold", pd.Series(dtype=float)), errors="coerce") > 0).sum())
		mean_alpha_bh = float(pd.to_numeric(folds_df.get("alpha_buy_hold_vs_benchmark", pd.Series(dtype=float)), errors="coerce").mean()) if "alpha_buy_hold_vs_benchmark" in folds_df else float("nan")
		mean_tp_sl_minus_bh = float(pd.to_numeric(folds_df.get("tp_sl_minus_buy_hold", pd.Series(dtype=float)), errors="coerce").mean()) if "tp_sl_minus_buy_hold" in folds_df else float("nan")
		mean_hybrid_minus_base = float(pd.to_numeric(folds_df.get("hybrid_minus_base", pd.Series(dtype=float)), errors="coerce").mean()) if "hybrid_minus_base" in folds_df else float("nan")
		hybrid_wins_vs_base = int((pd.to_numeric(folds_df.get("hybrid_minus_base", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if "hybrid_minus_base" in folds_df else 0

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
			"tp_sl_hybrid_vs_base": {
				"variant_mode": self._variant_mode_label(),
				"n_folds": int(len(folds_df)),
				"folds_hybrid_wins": hybrid_wins_vs_base,
				"folds_base_wins": int(len(folds_df) - hybrid_wins_vs_base),
				"mean_hybrid_minus_base": mean_hybrid_minus_base,
				"mean_return_base": float(pd.to_numeric(folds_df.get("return_base_tp_sl", pd.Series(dtype=float)), errors="coerce").mean()) if "return_base_tp_sl" in folds_df else float("nan"),
				"mean_return_hybrid": float(pd.to_numeric(folds_df.get("return_hybrid_tp_sl", pd.Series(dtype=float)), errors="coerce").mean()) if "return_hybrid_tp_sl" in folds_df else float("nan"),
			},
			"tp_sl_vs_buy_hold": {
				"enabled": bool(ENABLE_BUY_HOLD_COUNTERFACTUAL),
				"variant_mode": self._variant_mode_label(),
				"n_folds": int(len(folds_df)),
				"folds_tp_sl_wins": tp_sl_wins_vs_bh,
				"folds_buy_hold_wins": int(len(folds_df) - tp_sl_wins_vs_bh),
				"mean_alpha_tp_sl": mean_alpha,
				"mean_alpha_buy_hold": mean_alpha_bh,
				"mean_tp_sl_minus_buy_hold": mean_tp_sl_minus_bh,
				"global_return_tp_sl": global_strat.get("global_strategy_cumulative_return", float("nan")),
				"global_return_buy_hold": global_buy_hold.get("global_buy_hold_cumulative_return", float("nan")),
				"global_sharpe_buy_hold": global_buy_hold.get("global_buy_hold_sharpe", float("nan")),
			},
			"by_train_years": by_train_years,
			**global_strat,
			**global_buy_hold,
			**global_bench,
		}

		path = self.strategy_dir / "backtest_summary.json"
		with open(path, "w") as f:
			json.dump(summary, f, indent=2, default=str)

		self._export_counterfactual_artifacts(summary.get("tp_sl_vs_buy_hold", {}))

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
			"return_tp_sl", "return_buy_hold", "alpha_tp_sl_vs_benchmark",
			"alpha_buy_hold_vs_benchmark", "tp_sl_minus_buy_hold",
			"strategy_sharpe", "benchmark_sharpe", "excess_sharpe",
			"sharpe_tp_sl", "sharpe_buy_hold", "max_drawdown_tp_sl", "max_drawdown_buy_hold",
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


def _tp_sl_exit_from_close(
	close_series: pd.Series,
	*,
	entry_date: pd.Timestamp,
	fallback_exit_date: pd.Timestamp,
	tp_pct: float,
	sl_pct: float,
	max_holding_days: int,
	trailing_stop_pct: float = 0.0,
	trail_events: Optional[List[Dict]] = None,
) -> tuple[pd.Timestamp, str, int]:
	"""Resolve TP/SL-first exit date for a ticker close-price series."""
	fallback = pd.Timestamp(fallback_exit_date)
	entry_ts = pd.Timestamp(entry_date)

	if close_series is None or close_series.empty:
		return fallback, "time_exit", int(max((fallback - entry_ts).days, 0))

	if (not np.isfinite(tp_pct)) or (not np.isfinite(sl_pct)) or tp_pct <= 0.0 or sl_pct <= 0.0:
		return fallback, "time_exit", int(max((fallback - entry_ts).days, 0))

	sim = simulate_tp_sl(
		ticker="__SIM__",
		prices=close_series,
		entry_date=entry_ts,
		tp_pct=float(tp_pct),
		sl_pct=float(sl_pct),
		max_holding_days=int(max_holding_days),
		min_holding_days=int(max_holding_days * float(TP_SL_GRACE_PERIOD_FRACTION)),
		trailing_stop_pct=float(trailing_stop_pct),
		trailing_review_days=int(TP_SL_TRAILING_REVIEW_DAYS),
		trail_events=trail_events,
	)

	outcome = str(sim.get("outcome", "NONE")).upper()
	out_dt = sim.get("outcome_date")
	if pd.isna(out_dt):
		dt_exit = fallback
	else:
		resolved = _resolve_exec_date_on_or_before(close_series, pd.Timestamp(out_dt))
		dt_exit = pd.Timestamp(resolved) if resolved is not None else fallback

	if dt_exit <= entry_ts:
		dt_exit = fallback
		outcome = "NONE"

	if outcome == "TP":
		reason = "tp_hit"
	elif outcome == "SL":
		reason = "sl_hit"
	else:
		reason = "time_exit"

	days = int(sim.get("days_to_outcome", max((dt_exit - entry_ts).days, 0)))
	return pd.Timestamp(dt_exit), reason, int(max(days, 0))


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
    trail_events: Optional[List[Dict]] = None,
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
                min_holding_days=int(max_holding_days * float(TP_SL_GRACE_PERIOD_FRACTION)),
                trailing_stop_pct=0.05,
                trailing_review_days=int(TP_SL_TRAILING_REVIEW_DAYS),
                trail_events=trail_events,
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
        "trail_events": trail_events if trail_events else [],
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

