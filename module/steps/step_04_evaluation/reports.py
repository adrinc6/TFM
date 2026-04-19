"""Report generation helpers."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def generate_text_report(
	summary: Dict,
	fold_results: List[Dict],
	agent_diag_history: Dict[str, List],
	backtest_results_dir: str,
) -> None:
	lines = []
	sep = "=" * 65
	sep_s = "-" * 65

	lines.append(sep)
	lines.append("  RESULTS REPORT — Walk-Forward Backtest")
	lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append(sep)

	lines.append("\n  GLOBAL METRICS (all folds concatenated)")
	lines.append(sep_s)
	lines.append(f"  Folds completados:         {summary.get('n_folds', 0)}")
	lines.append(f"  Alpha medio por fold:      {summary.get('mean_alpha', 0):+.2%}")
	lines.append(f"  Folds con alpha positivo:  {summary.get('pct_folds_positive_alpha', 0):.0%}")

	gs = summary.get("global_strategy_sharpe", 0)
	gb = summary.get("global_benchmark_sharpe", 0)
	lines.append(f"  Sharpe estrategia:         {gs:.3f}")
	lines.append(f"  Sharpe benchmark (S&P500): {gb:.3f}")
	lines.append(f"  Sortino estrategia:        {summary.get('global_strategy_sortino', 0):.3f}")
	lines.append(f"  Max Drawdown estrategia:   {summary.get('global_strategy_max_drawdown', 0):.2%}")
	lines.append(f"  Max Drawdown benchmark:    {summary.get('global_benchmark_max_drawdown', 0):.2%}")
	lines.append(f"  Calmar ratio:              {summary.get('global_strategy_calmar', 0):.3f}")
	lines.append(f"  Volatilidad anualizada:    {summary.get('global_strategy_volatility', 0):.2%}")

	if fold_results:
		lines.append("\n  DETALLE POR FOLD")
		lines.append(sep_s)
		header = (
			f"  {'Fold':>4}  {'Train':>4}Y  "
			f"{'Periodo Test':<24}  "
			f"{'Ret Strat':>9}  {'Ret Bench':>9}  "
			f"{'Alpha':>7}  {'Sharpe':>6}  {'AUC':>6}"
		)
		lines.append(header)
		lines.append("  " + "-" * 63)
		for fr in fold_results:
			test_period = f"{fr.get('test_start','')} -> {fr.get('test_end','')}"
			strat_ret = fr.get("strategy_cumulative_return", 0)
			bench_ret = fr.get("benchmark_cumulative_return", 0)
			alpha_v = fr.get("alpha", 0)
			sharpe_v = fr.get("strategy_sharpe", 0)
			auc_v = fr.get("roc_auc", fr.get("auc", float("nan")))
			train_y = fr.get("train_years", "?")
			fold_id = fr.get("fold", "?")
			auc_str = f"{auc_v:.3f}" if isinstance(auc_v, float) and not pd.isna(auc_v) else "  N/A"
			lines.append(
				f"  {fold_id:>4}  {train_y:>4}Y  "
				f"{test_period:<24}  "
				f"{strat_ret:>+9.2%}  {bench_ret:>+9.2%}  "
				f"{alpha_v:>+7.2%}  {sharpe_v:>6.3f}  {auc_str:>6}"
			)

	by_train = summary.get("by_train_years", {})
	if by_train:
		lines.append("\n  DESGLOSE POR LONGITUD DE TRAIN")
		lines.append(sep_s)
		lines.append(f"  {'Train':>5}  {'N folds':>7}  {'Ret medio':>9}  {'Alpha medio':>11}  {'a>0':>5}  {'Sharpe':>6}")
		lines.append("  " + "-" * 50)
		for ny, stats in sorted(by_train.items()):
			lines.append(
				f"  {ny:>4}Y  {stats['n_folds']:>7}  "
				f"{stats['mean_strategy_return']:>+9.2%}  "
				f"{stats['mean_alpha']:>+11.2%}  "
				f"{stats['pct_positive_alpha']:>5.0%}  "
				f"{stats['mean_strategy_sharpe']:>6.3f}"
			)

	lines.append("\n  AGENT AUC (last trained fold)")
	lines.append(sep_s)
	for ag_name, history in agent_diag_history.items():
		if not history:
			continue
		last = history[-1]
		cv = last.get("cv_metrics") or last.get("cv_lr") or {}
		auc = cv.get("mean_auc", None)
		std = cv.get("std_auc", None)
		if auc is not None:
			std_str = f" ± {std:.4f}" if std is not None else ""
			lines.append(f"  {ag_name:<15}  AUC = {auc:.4f}{std_str}")

	lines.append(f"\n{sep}")
	lines.append(f"  Results saved to: {backtest_results_dir}/")
	lines.append(sep)

	report_text = "\n".join(lines)
	report_path = Path(backtest_results_dir) / "report.txt"
	report_path.parent.mkdir(parents=True, exist_ok=True)
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report_text)
