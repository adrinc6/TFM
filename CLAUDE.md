# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GARP AI Portfolio System — a research pipeline (TFM/thesis project) that tests whether a GARP
(Growth At a Reasonable Price) / Value-Growth equity strategy can generate alpha vs. SPY, using a
point-in-time dataset, a walk-forward ML scoring model, deterministic + optional-LLM research
generation, and a monthly portfolio backtest with a static HTML viewer/report as output.

`README.md` and `doc.md` have been **updated to reflect the current state** after an integral audit
of methodology, robustness, and presentation. Both files are now current and synchronized with
`CLAUDE.md`. Key topics: forward-looking ML targets, integrated position sizing, statistical
rigor (IR/TE/t-stat), Spanish-language viewer, explicit methodological limitations.

## Running

```bash
python main.py
```

There is no CLI/argparse — all run configuration lives in `environment.py` (edit the constants
directly, it's not read from env vars except API keys):

- `RUN_MODE`: one of `download`, `dataset`, `features`, `ml`, `watchlist`, `research_ai`, `backtest`,
  `viewer`, `report`, or `full`. Each stage is independently re-runnable as long as its upstream
  parquet/CSV inputs already exist on disk (see pipeline stages below).
- `DEV_MODE`: when `True`, restricts the universe to `DEV_TICKERS + SPY` — use this for any manual
  testing/iteration instead of the full ~500-ticker universe.
- `FORCE_RAW_DOWNLOAD`: `False` reuses cached raw JSON under `data/raw/json/`; set `True` to bypass
  the cache and re-hit Finnhub/Yahoo.
- `.env` holds only `FINNHUB_API_KEY` and `OPENAI_API_KEY`; `environment.py` parses it manually
  (no `python-dotenv` dependency).

There's no dependency manifest in the repo (no `requirements.txt`/`pyproject.toml`) — dependencies
are installed ad hoc. From imports: `pandas`, `requests`, `lightgbm` (falls back to
`sklearn.ensemble.RandomForestRegressor` if unavailable/erroring), `shap` (optional, wrapped in
try/except), `matplotlib`. No test suite exists yet.

## Pipeline architecture

`main.py` runs stages in this fixed order, gated by `settings.run_mode`:

```
download → dataset → features → ml → watchlist → research_ai → backtest → viewer → report
```

| Stage | Entry point | Reads | Writes |
|---|---|---|---|
| download | `module.ingest.pipeline.download_raw_data` | Finnhub/Yahoo APIs + `data/raw/json/` cache | `data/raw/{profiles,finnhub_metrics,prices,news}.parquet` |
| dataset | `module.dataset.build_master_dataset` | `data/raw/{prices,finnhub_metrics,profiles}.parquet` | `data/master/master_point_in_time.parquet` |
| features | `module.features.pipeline.build_features` | `data/master/master_point_in_time.parquet` | `data/processed/features.parquet` |
| ml | `module.ml.train_and_score` | `data/processed/features.parquet`, `data/raw/prices.parquet` | `data/processed/scored_universe.parquet`, `data/processed/model_explainability.json`, `results/<run>/model_walk_forward_diagnostics.csv` |
| watchlist | `module.strategy.selection.build_watchlist` | `data/processed/scored_universe.parquet` | `data/processed/watchlist.parquet`, `results/<run>/watchlist.csv`, `.../audit/watchlist_history.csv` |
| research_ai | `module.research.ai.build_openai_research` | `data/processed/scored_universe.parquet`, `data/raw/news.parquet` | `results/<run>/research_ai.csv`, `.../audit/research_ai_history.csv`, `.../audit/research_ai/<ticker>.json` |
| backtest | `module.backtest.run_backtest` | `data/processed/scored_universe.parquet`, `data/raw/prices.parquet` | ~20 CSVs under `results/<run>/` and `.../audit/`, `results/<run>/portfolio_monthly_summary.json` |
| viewer | `module.viewer.build_viewer` | all CSVs under `results/<run>/`(`/audit/`), `model_explainability.json` | `results/<run>/viewer/*.html`, `.../viewer/charts/*.png`, `.../expl_results.md`, `.../result_manifest.json` |
| report | `module.report.build_final_report` | `results/<run>/*.csv` (+ audit fallback), `model_explainability.json` | `results/<run>/final_report.html` |

`Settings.run_dir` (`environment.py`) = `results/<dev|full>_<start_date>_<end_date>_<review_frequency>/`
— every stage after `download`/`dataset` reads/writes under this run-scoped directory, so switching
`PORTFOLIO_START_DATE`/`PORTFOLIO_END_DATE`/`DEV_MODE`/`PORTFOLIO_REVIEW_FREQUENCY` in
`environment.py` targets a different results folder rather than overwriting the previous run.

## Module map

- `module/ingest/` — `clients.py` has `FinnhubClient` (rate-limited, handles 429/401/403 with
  max **5 retries** per 429 before logging error and returning None) and `YahooClient` (calls Yahoo's
  `v8/finance/chart` endpoint directly, no `yfinance` dep, same retry limit).
  `pipeline.py`'s `download_raw_data` caches every raw response to
  `data/raw/json/<source>/<ticker>/<dataset>.json` (network-free re-runs unless `FORCE_RAW_DOWNLOAD=True`).
  **Robust**: per-ticker try/except means one ticker's failure doesn't abort the entire run;
  writes `data/raw/download_coverage.json` + `download_failures.csv` for visibility.
- `module/dataset.py` — `build_master_dataset` builds one row per `(ticker, snapshot_date)`. Core
  point-in-time mechanism: `bisect_right` over sorted per-ticker series so every value is the latest
  one available strictly at-or-before the snapshot date (`_latest_price`, `_historical_value`,
  `_historical_growth`). This is what prevents lookahead bias at the data layer.
- `module/features/` — `pipeline.py`'s `build_features` computes cross-sectional percentile scores
  per snapshot_date (quality/growth/valuation/momentum/moat/catalyst/risk) into a deterministic
  `garp_score`; delegates expectation-gap, sector/universe percentile, and trailing-window trend
  features to `transforms.py`.
- `module/ml.py` — `train_and_score` is walk-forward: for each historical snapshot date, trains only
  on a trailing `[date - max_walk_forward_training_years, date]` window. **Four DISJOINT specialist
  agents** (`AGENT_FEATURES` subsets share no feature), each fit against its own forward-looking
  target anchored to *observed* outcomes (not re-projections of input features): `target_quality`
  (forward change in **reported ROIC**), `target_improvement` (forward **observed** fundamental growth
  vs. today's market-implied expectation), `target_mispricing` (whether today's real valuation
  discount resolved into forward alpha, 6m horizon), `target_timing` (short-horizon **3m** forward
  excess return — entry/momentum). There is deliberately **no generalist "alpha" agent**: it predicted
  the same `target_future_alpha` the meta-agent scores against, with a superset of features, so it
  structurally dominated the blend and was removed. `target_future_alpha` (excess return 12m ahead)
  survives only as the **evaluation label** — what the meta-agent learns against and what OOS rank-IC
  is measured on, never an agent that predicts it. Each target is masked on rows whose *own-horizon*
  future is unobservable at training date (falls back to the deterministic GARP component). Per-snapshot
  **out-of-sample rank-IC / RMSE** vs. realized alpha (per agent and for the combined `final_score`)
  and every fallback are logged to `results/<run>/model_walk_forward_diagnostics.csv`. A **meta-agent**
  (`_meta_agent_scores` / `_fit_meta_weights`) then LEARNS, per snapshot, how to weight the four agents
  by each agent's **marginal ranking contribution** (partial rank-IC against the forward alpha the
  other three leave unexplained, on a 30% chronological hold-out inside the training window) into
  `final_score` (weights + partial ICs persisted to
  `data/processed/meta_weights_by_snapshot.parquet`; falls back to the fixed prior when history is
  thin). `opportunity_type` is a descriptive rule-based label. See invariants below.
- `module/research/` — `synthesis.py` is pure rule-based text generation from the numeric scores (no
  external calls). `thesis.py` computes thesis/health/conviction/exit scores and calls
  `module.strategy.selection.add_buy_today_decision` (imported lazily inside the function to avoid a
  circular import, since `strategy.selection` also calls back into `research.thesis`). `ai.py` is the
  only OpenAI integration point — gated behind `ENABLE_OPENAI_RESEARCH` + `OPENAI_API_KEY`
  (currently disabled by default), calls `https://api.openai.com/v1/responses` via raw `requests`
  (no `openai` SDK), and always falls back to the deterministic `synthesis.py` output on any failure
  or when disabled, so downstream artifacts never break.
- `module/strategy/` — `selection.py` ranks/filters into a watchlist and computes opportunity-cost
  vs. the best alternative ticker per snapshot. `portfolio.py` (`initial_portfolio`,
  `review_portfolio`) is pure in-memory concentrated-portfolio logic. `manager_score` now leads with
  the learned `final_score` (weight 0.45, the meta-agent output) plus manager overlays
  (timing/valuation/risk); it is still distinct from `final_score` but is driven by the learned
  signal. Rotation is governed by explicit `environment.py` thresholds (`MIN_ROTATION_ADVANTAGE`,
  `MIN_SCORE_ADVANTAGE_TO_REPLACE`, `MIN_CONVICTION_ADVANTAGE`, `MIN_OPPORTUNITY_COST_THRESHOLD` — no
  hardcoded duplicates), and **soft** exit triggers (manager-score hurdle, capital opportunity cost,
  slow deterioration) respect `MIN_HOLD_MONTHS_BEFORE_ROTATION` while **hard** triggers (broken thesis,
  exit-score, momentum+thesis breakdown) fire immediately. `sizing.py` computes
  equal/conviction/risk-adjusted/hybrid position weights, tilted hard toward the conviction component
  with a convexity exponent so the book concentrates on its best ideas; `hybrid_weight` also drives the
  actual return simulation in
  `module/backtest/performance.py` (not just display). No file I/O in this package — it's consumed
  by `module/backtest/engine.py`.
- `module/backtest/` — `engine.py`'s `run_backtest` is the monthly simulation loop calling
  `initial_portfolio`/`review_portfolio` per snapshot. `performance.py` does FIFO lot-based position
  tracking, applies `transaction_cost_bps`/`slippage_bps` as rebalance drag, **and uses
  `hybrid_weight` for weighted return calculation** (not equal-weight fallback); transaction cost is
  notional-weighted (weight traded × cost_rate), not count-based. `reviews.py` produces the
  full-universe BUY/SELL/HOLD/AVOID review (independent of current holdings) plus held-position
  ADD/REDUCE/WATCH/HOLD decisions — kept deliberately separate to support opportunity-cost analysis.
  `artifacts.py` shapes the executive-tier output tables and computes **information ratio, tracking
  error, t-stat with explicit small-sample caveat**. `robustness.py` re-reads the finished tables
  (no re-scoring) to add a **block bootstrap** CI on cumulative alpha / IR, a **cost-sensitivity
  sweep** whose headline is the cost multiplier at which alpha crosses zero, and a pure multi-cutoff
  aggregator (`compare_train_cutoffs`) ready for A/B over `TRAIN_CUTOFF_DATE`. Heavy per-ticker/
  per-snapshot tables go to `results/<run>/audit/` (see `AUDIT_OUTPUTS`); compact summaries stay at
  `results/<run>/` root.
- `module/viewer/` — **All UI is in Spanish**. Rewritten as **one professional single-page report**
  (`results/<run>/viewer/index.html`), not a 16-page dump. `pages.py`'s `build_viewer` loads the run
  CSVs and renders purpose-driven sections (executive KPIs, portfolio-vs-benchmark, current
  portfolio + trades, **learning evidence** = learned meta-agent weights + OOS rank-IC of the combined
  `final_score`, position attribution, a **Robustez** section = bootstrap CI + cost-sensitivity, and a
  separate debug/TFM block). `charts.py` renders a small set of charts (matplotlib
  `Agg`) with the validated dataviz palette. `shared.py` holds the theme-aware CSS design system and
  table/kpi/figure helpers with a NaN/Inf-safe formatter. Heavy audit CSVs stay on disk under
  `results/<run>/audit/` and are only linked, never embedded. Every element must pass the utility
  test in `shared.py` (answers a TFM question or aids debugging) before it is added.
- `module/report.py` — **All in Spanish**. Now the metrics module: keeps `_metrics`
  (CAGR/Sharpe/Sortino/max-drawdown/alpha, IR/TE/t-stat) and `drawdown_episodes`, reused by the
  viewer. `build_final_report` builds the single viewer report (the report **is** the viewer report),
  so the `report` stage points at `results/<run>/viewer/index.html` instead of a second HTML page.
- `module/utils.py` — shared `setup_logging`, `write_parquet`/`read_parquet`, `write_json` helpers
  used throughout.

## Key invariants to preserve when editing

- **No lookahead bias**: any change to `module/dataset.py` or `module/ml.py` must preserve the
  as-of-date semantics (`bisect_right` lookups, label-horizon masking). This is the central
  methodological guarantee of the whole project — breaking it silently invalidates the backtest.
  `results/<run>/model_walk_forward_diagnostics.csv` is the audit trail for every fallback decision;
  keep writing to it if you touch the walk-forward logic.
  - **Two independent scoring layers must stay separate**: `module/ml.py`'s `final_score` /
  `opportunity_type` (statistical) vs. `module/strategy/portfolio.py`'s `manager_score`
  (hand-weighted, used for actual entry/exit decisions). Don't collapse them into one score — the
  divergence between the two is what the opportunity-cost analysis in `backtest/reviews.py` depends
  on.
- **Executive vs. audit output tiering**: heavy per-ticker/per-snapshot tables go to
  `results/<run>/audit/` (`AUDIT_OUTPUTS` in `module/backtest/engine.py`); compact summaries stay at
  the run root. The viewer embeds only compact/executive tables and **links** audit CSVs, never
  embeds them — keep that split (a new page must pass the utility test in `module/viewer/shared.py`).
- **External calls stay optional and fail safe**: `module/research/ai.py` must keep working (via
  deterministic fallback) with `ENABLE_OPENAI_RESEARCH=False` or no API key — don't make any stage
  hard-depend on the OpenAI call succeeding.
- **The four `module/ml.py` agent targets must stay anchored to observed outcomes, not re-projections**:
  `target_quality` (forward reported ROIC change), `target_improvement` (forward observed fundamental
  growth vs. expectation), `target_mispricing` (real valuation discount resolving into forward alpha),
  and `target_timing` (short-horizon forward excess return) are each built from information observable
  their **own horizon** ahead (`AGENT_HORIZON_MONTHS_OVERRIDE`) and masked accordingly during
  walk-forward training. `target_future_alpha` is the separate 12m **evaluation label**, not an agent
  target — do not resurrect a generalist "alpha" agent that predicts it (it structurally dominated the
  blend). Do not reintroduce deterministic same-day-feature blends as targets (e.g. the old circular
  `realized_growth = 0.6*growth+0.25*quality+0.15*moat`) — that was the original design flaw fixed here.
- **Meta-agent weights are learned by marginal contribution, not hard-coded**: `_fit_meta_weights`
  weights each agent by its **partial rank-IC** — its Spearman IC against the realized forward alpha
  the other three agents leave unexplained, measured on a 30% chronological hold-out inside the
  training window (no lookahead) — so complements are rewarded over redundancy. This is the "learns
  from the simulation" loop. Don't revert `final_score` to a fixed blend or to raw (non-partial) IC;
  keep `AGENT_PRIOR_WEIGHTS` only as the thin-history fallback and `garp_score` as the fixed baseline.
- **Fundamental publication lag**: `module/dataset.py::_prepared_rows` shifts every fundamental's
  period-end date forward by `FUNDAMENTAL_PUBLICATION_LAG_WEEKS` so a fundamental is only observable
  after its real reporting date. Preserve this shift — removing it reintroduces a subtle lookahead.
- **`hybrid_weight` (from `module/strategy/sizing.py`) drives the actual backtest P&L**: `module/backtest/performance.py`'s
  `weighted_basket_return`/`period_transaction_cost` consume normalized per-period `hybrid_weight`
  values (falling back to equal-weight only when weights are unavailable). Keep sizing and the return
  simulation in sync — don't let them diverge again.
- **Viewer/report language is Spanish**: `module/viewer/` and `module/report.py` render user-facing
  text in Spanish (chart titles, page headers, prose). Keep new pages/charts consistent with this.

## Methodological limitations

- **Survivorship bias**: `environment.py`'s `TICKERS` is a static, present-day large-cap roster
  applied retroactively back to `DATA_START_DATE`. It excludes names delisted/acquired/dropped from
  major indices over that period and includes recent IPOs/spin-offs that did not exist historically.
  This is a deliberate scope decision (documented, not fixed) — the live portfolio only trades within
  `PORTFOLIO_START_DATE`/`PORTFOLIO_END_DATE`, which bounds but does not eliminate the issue, and the
  ML training window (`MAX_WALK_FORWARD_TRAINING_YEARS` of lookback) inherits the same bias. This
  caveat is surfaced in the viewer report's executive summary (`SMALL_SAMPLE_CAVEAT`).
- **Small-sample statistical power**: the backtest window is a few dozen monthly observations.
  `module/backtest/artifacts.py::excess_return_statistics` reports information ratio, tracking error,
  and a t-stat on excess return, but deliberately does not attempt bootstrap confidence intervals or
  deflated-Sharpe/multiple-testing corrections — those would imply more statistical precision than
  this sample size supports. Treat the t-stat as directional, not proof of a significant edge.

## Status & Verification

### Complete audit (all approved fixes implemented)
✅ ML methodology: targets redefined as genuine forward predictions + consistent fuga masking
✅ Ingesta robustness: per-ticker error handling + 429 retry limit + coverage reporting
✅ P&L correctness: hybrid_weight integrated + notional-weighted transaction costs
✅ Statistical rigor: IR/TE/t-stat with explicit small-sample caveat
✅ UI/UX: Spanish throughout, format unified, error isolation per page
✅ Curated analysis: walk-forward diagnostics + drawdown episodes surfaced
✅ Documentation: all limitations explicit (survivorship bias, sample size)

### To verify a fresh run
1. `DEV_MODE = True` in `environment.py` (5 tickers + SPY = fast iteration)
2. `RUN_MODE = "full"` (all stages) or target a specific stage if re-iterating
3. Check: `python main.py` completes without error, produces `results/<run>/final_report.html` + `results/<run>/viewer/dashboard.html`
4. No crashes on `NaN`/`Inf`, all numeric columns formatted, no "nan%" in prose, walk-forward diagnostics table present in model_explainability.html

### For production run with real data
1. Configure `FINNHUB_API_KEY` in `.env`
2. Set actual date range in `environment.py` (`DATA_START_DATE`, `PORTFOLIO_START_DATE`, `PORTFOLIO_END_DATE`)
3. Set `DEV_MODE = False` to use full `TICKERS` universe
4. Run `python main.py` with `RUN_MODE = "full"`
5. Results will be in `results/full_<start>_<end>_<freq>/` with full audit trail and 3-dimensional viewer
