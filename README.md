# Multi-Agent ML Stock Picker — TP/SL + Confidence Strategy

> **TFM (Trabajo de Fin de Máster)**  
> A modular, interpretable multi-agent system for S&P 500 stock selection
> driven by Take-Profit / Stop-Loss signals and agent confidence scoring.

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [How Decisions Are Made](#how-decisions-are-made)
3. [System Architecture](#system-architecture)
4. [New Strategy Modules](#new-strategy-modules)
5. [Configuration](#configuration)
6. [How to Run](#how-to-run)
7. [Output Format](#output-format)
8. [Running Tests](#running-tests)
9. [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## Strategy Overview

The system trains a multi-agent ensemble of ML models—each specialised in a
different market signal (fundamentals, valuation, momentum, sentiment,
sector rotation, bear-market detection)—and combines their outputs to:

1. **Estimate a Take-Profit (TP) level** per stock (as a percentage).
2. **Estimate a Stop-Loss (SL) level** per stock (as a percentage).
3. **Compute a Confidence score** representing the probability that TP is
   reached before SL within the holding window.
4. **Rank all stocks** by Expected Value (EV):

   ```
   EV = confidence × tp_pct  −  (1 − confidence) × sl_pct
   ```

5. **Select a portfolio** of **4–8 stocks** with the highest EV,
   subject to sector concentration constraints.
6. **Backtest** each prediction to record whether TP, SL, or neither
   (NONE) was hit first, and after how many days.

---

## How Decisions Are Made

### Step 1 — Agent Scoring

Each base agent (Fundamental, Valuation, Momentum, Bear, Sentiment,
SectorRotation) independently scores every stock in `[0, 1]` using its own
feature set and ML model (XGBoost, LightGBM, or a TFT-lite blend).

### Step 2 — TP/SL Signal Generation (`module/strategy/signal_generation.py`)

Agent scores are aggregated (optionally weighted by historical hit rates)
into a single `score`, then mapped to TP/SL percentages:

```
tp_pct = BASE_TP + TP_SENSITIVITY × (score − 0.5)
sl_pct = BASE_SL − SL_SENSITIVITY × (score − 0.5)
```

| Score | Interpretation | TP % | SL % |
|-------|---------------|------|------|
| 0.8   | Strongly bullish | ~10.5% | ~4.4% |
| 0.5   | Neutral        | 8%   | 5%   |
| 0.2   | Bearish        | ~5.5% | ~5.6% |

Both values are clipped to `[MIN_TP, MAX_TP]` and `[MIN_SL, MAX_SL]`.

### Step 3 — Confidence Scoring (`module/strategy/confidence_model.py`)

Confidence blends:
- **Raw model confidence** (weighted agent scores).
- **Historical calibration** (each agent's EWMA TP-hit rate from past folds).

```
confidence = 0.5 × raw_score + 0.5 × calibration
```

### Step 4 — Portfolio Selection (`module/strategy/portfolio_selection.py`)

Stocks are ranked by EV.  Selection rules:

- **Minimum 4 stocks** required to invest (no investment otherwise).
- **Maximum 8 stocks** in the portfolio.
- **Maximum 3 stocks per sector** (GICS sector cap, relaxed only to meet
  the minimum floor).

### Step 5 — TP/SL Backtesting (`module/strategy/backtesting_engine.py`)

For each stock, prices are scanned day by day from `entry_date`:

```
if price ≥ entry_price × (1 + tp_pct) → outcome = "TP"
if price ≤ entry_price × (1 − sl_pct) → outcome = "SL"
if holding window expires              → outcome = "NONE"
```

### Step 6 — Agent Weight Update (`module/strategy/agent_weighting.py`)

After each fold, each agent's TP-hit rate for its top-N picks is recorded.
An EWMA tracks accuracy over time; agents with higher hit rates receive
proportionally greater weight in the next fold.

---

## System Architecture

```
data_finnhub/           Raw market data (Finnhub)
module/
  agents/               Base ML agents (fundamental, momentum, …)
  common/               Shared utilities (features, anti-leakage, …)
  strategy/             ⬅ NEW: TP/SL + confidence strategy
    signal_generation.py
    confidence_model.py
    portfolio_selection.py
    backtesting_engine.py
    agent_weighting.py
  steps/
    step_01_data/       ETL pipeline
    step_02_dataset/    Feature engineering
    step_03_training/   Walk-forward agent training
    step_04_evaluation/ Backtesting, reporting, CSV export
      tp_sl_reporter.py ⬅ NEW: strategy CSV export
environment.py          All configuration (single source of truth)
analyzer.py             Main entry point
tests/                  Unit + integration tests
examples/               Sample output CSV
```

---

## New Strategy Modules

| Module | Purpose |
|--------|---------|
| `module/strategy/signal_generation.py` | Maps agent scores to TP/SL % levels |
| `module/strategy/confidence_model.py` | Computes TP-hit probability from scores + calibration |
| `module/strategy/portfolio_selection.py` | Ranks stocks by EV, enforces 4–8 stock constraints |
| `module/strategy/backtesting_engine.py` | Determines if TP/SL is hit first, records days |
| `module/strategy/agent_weighting.py` | EWMA-based dynamic agent weight tracking |
| `module/steps/step_04_evaluation/tp_sl_reporter.py` | Exports the full-universe strategy CSV |

---

## Configuration

All strategy parameters are defined in `environment.py` (section 10):

```python
# Signal generation
TP_SL_BASE_TP          = 0.08   # 8 % base take-profit
TP_SL_BASE_SL          = 0.05   # 5 % base stop-loss
TP_SL_TP_SENSITIVITY   = 0.10   # max TP shift when score → 1.0
TP_SL_SL_SENSITIVITY   = 0.04   # max SL shift when score → 1.0

# Portfolio constraints
TP_SL_MIN_STOCKS       = 4      # minimum stocks to invest
TP_SL_MAX_STOCKS       = 8      # maximum portfolio size
TP_SL_SECTOR_CAP       = 3      # max stocks per GICS sector
TP_SL_EV_THRESHOLD     = 0.0    # minimum EV to be eligible

# Backtesting
TP_SL_MAX_HOLDING_DAYS = 90     # max holding window (calendar days)

# Agent weighting
TP_SL_WEIGHT_DECAY     = 0.85   # EWMA decay (higher = slower forgetting)
TP_SL_WEIGHT_PRIOR     = 0.50   # initial hit-rate prior
TP_SL_WEIGHT_MIN       = 0.05   # minimum agent weight floor
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional for live data)

```bash
echo "FINNHUB_API_KEY=your_key_here" > .env
```

### 3. Run the full pipeline

```bash
python analyzer.py
```

### 4. Run the TP/SL strategy on a custom DataFrame

```python
from module.strategy.signal_generation import build_signals
from module.strategy.confidence_model import attach_confidence
from module.strategy.portfolio_selection import select_portfolio
from module.strategy.backtesting_engine import run_backtest
from module.steps.step_04_evaluation.tp_sl_reporter import export_strategy_csv

# agent_scores_df: DataFrame with 'ticker' + *_score columns
signals = build_signals(agent_scores_df)
signals = attach_confidence(signals, agent_hit_rates=hit_rates)
signals = select_portfolio(signals)
signals = run_backtest(signals, prices_dict, entry_date)

export_strategy_csv(signals, "results/strategy/output.csv", fold_id="2024Q1")
```

---

## Output Format

The strategy generates a CSV for **all stocks** (not only selected ones).
See `examples/sample_strategy_output.csv` for a concrete example.

Key columns:

| Column | Description |
|--------|-------------|
| `ticker` | Stock identifier |
| `fold_id` | Walk-forward fold label (e.g. `2024Q1`) |
| `score` | Aggregate agent score [0, 1] |
| `confidence` | Probability TP is reached before SL [0, 1] |
| `ev` | Expected value = confidence × tp_pct − (1−confidence) × sl_pct |
| `tp_pct` | Take-profit level as a fraction |
| `sl_pct` | Stop-loss level as a fraction |
| `tp_price` | Absolute TP price (when entry price is available) |
| `sl_price` | Absolute SL price |
| `selected` | True if the stock was chosen for the portfolio |
| `outcome` | `TP` / `SL` / `NONE` |
| `days_to_outcome` | Calendar days from entry to outcome |
| `fundamental_score` | Individual agent score |
| `weight_<agent>` | Dynamic weight assigned to each agent this fold |
| `hit_rate_<agent>` | EWMA historical TP-hit rate per agent |

---

## Running Tests

```bash
# All tests (strategy + existing)
python -m pytest tests/ -v

# Strategy tests only
python -m pytest tests/test_tp_sl_strategy.py -v

# Existing tests only
python -m pytest tests/test_antileakage_and_policy.py \
                 tests/test_financial_strategy_constraints.py \
                 tests/test_sector_specialized_agent.py -v
```

---

## Design Decisions & Trade-offs

### Why Expected Value as the ranking metric?

EV is the most natural single-number summary of a trade opportunity: it
captures both the upside magnitude and the probability of achieving it.
Alternatives (Sharpe, Kelly fraction, win-rate alone) require additional
assumptions or data that are less directly observable.

### Why 4–8 stocks?

- **Minimum 4**: ensures diversification and avoids single-stock risk.
- **Maximum 8**: keeps the portfolio manageable and avoids diluting signal
  with low-conviction picks.

### Why EWMA for agent weighting?

EWMA balances adaptivity (tracking recent accuracy) with stability
(not overreacting to a single fold).  The decay parameter `TP_SL_WEIGHT_DECAY`
can be tuned for faster or slower regime changes.

### Trade-off: interpretability vs. complexity

All mapping functions (score → TP/SL, score → confidence) are intentionally
linear and bounded.  This makes the system auditable and easy to debug,
at the cost of some modelling flexibility.  Non-linear mappings can be
plugged in by replacing the respective functions in `signal_generation.py`
and `confidence_model.py` without changing any other module.

### TP/SL symmetry assumption

The current baseline uses equal TP and SL sensitivities for simplicity.
In practice, different asset classes or market regimes may warrant
asymmetric sensitivities, which can be adjusted via `environment.py`.
