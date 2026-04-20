# Performance Report - Baseline vs Improved System

## Evaluation Protocol
- Walk-forward backtesting in production pipeline.
- Temporal integrity with purged + embargoed OOF generation.
- Portfolio allocation via HRP using pre-entry trailing returns.

## Baseline System
- Binary target (outperform vs median).
- Legacy stack meta learner.
- Score-based weighting/equal-weight fallback.

## Improved System
- Continuous alpha target + quintile + triple-barrier labels.
- AlphaMetaLearner with ranking objective.
- Regime-aware score adjustment.
- HRP portfolio allocation.
- FinBERT sentiment features.

## Metrics to Compare
- Sharpe Ratio
- Sortino Ratio
- CAGR
- Max Drawdown
- Hit Rate
- Turnover
- Alpha vs benchmark

## Where Metrics Are Produced
- Fold-level and global: results/strategy/backtest_summary.json
- Fold details: results/strategy/folds_results.csv
- Return series: results/strategy/returns_series.csv
- Baseline comparisons: results/strategy/baselines/*

## Regime Breakdown
The improved pipeline emits regime_state and regime_adjusted_score in scored folds.
Use scored exports to segment performance by regime:
- Risk-On
- Neutral
- Risk-Off

## Before vs After Table (to fill after full run)
| Metric | Baseline | Improved | Delta |
|---|---:|---:|---:|
| Sharpe | N/A | N/A | N/A |
| Sortino | N/A | N/A | N/A |
| CAGR | N/A | N/A | N/A |
| Max Drawdown | N/A | N/A | N/A |
| Hit Rate | N/A | N/A | N/A |
| Turnover | N/A | N/A | N/A |
| Alpha | N/A | N/A | N/A |

## Notes
- This document is generated as a reproducible template.
- Populate values by running the full analyzer pipeline in the upgraded environment.
