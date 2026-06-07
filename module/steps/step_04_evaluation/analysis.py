"""Consolidated evaluation analyses: ablation and portfolio-size studies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from module.steps.step_04_evaluation.strategy import simulate_tp_sl
log = logging.getLogger(__name__)

ABLATION_AGENTS = ["fundamental", "valuation", "momentum", "bear"]


def run_ablation_study(
    df_test_scored: pd.DataFrame,
    y_test: pd.Series,
    df_train_norm: pd.DataFrame,
    y_train: pd.Series,
    agents_results_dir: str,
    fold_id: int,
    random_seed: int = 42,
    fold_result: Dict | None = None,
) -> Dict:
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.error("[Ablation] scikit-learn no disponible.")
        return {}

    score_cols = [f"{ag}_score" for ag in ABLATION_AGENTS]
    available_score_cols = [c for c in score_cols if c in df_test_scored.columns]

    if len(available_score_cols) < 2:
        log.warning(f"[Ablation] Fold {fold_id}: pocas columnas de score disponibles â€” omitido.")
        return {}

    X_train = df_train_norm[[c for c in available_score_cols if c in df_train_norm.columns]].copy()
    X_test = df_test_scored[[c for c in available_score_cols if c in df_test_scored.columns]].copy()

    y_tr = y_train.reindex(X_train.index).dropna()
    X_train = X_train.loc[y_tr.index].fillna(0.5)
    y_te = y_test.reindex(X_test.index).dropna()
    X_test = X_test.loc[y_te.index].fillna(0.5)

    if len(y_tr) < 20 or len(y_te) < 5:
        log.warning(f"[Ablation] Fold {fold_id}: insufficient data for ablation.")
        return {}

    def _auc(X_tr, y_tr, X_te, y_te):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=500,
                random_state=random_seed, solver="lbfgs",
            )),
        ])
        pipe.fit(X_tr, y_tr)
        if y_te.nunique() < 2:
            return float("nan")
        p = pipe.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, p))

    baseline_auc = _auc(X_train, y_tr, X_test, y_te)

    ablation_results = {}
    for ag in ABLATION_AGENTS:
        col = f"{ag}_score"
        if col not in X_train.columns:
            continue
        X_tr_ab = X_train.drop(columns=[col])
        X_te_ab = X_test.drop(columns=[col])
        if X_tr_ab.shape[1] == 0:
            continue
        auc_ab = _auc(X_tr_ab, y_tr, X_te_ab, y_te)
        contribution = (baseline_auc - auc_ab) if not np.isnan(auc_ab) else float("nan")
        ablation_results[ag] = {
            "auc_without": round(auc_ab, 4),
            "auc_baseline": round(baseline_auc, 4),
            "marginal_contribution": round(contribution, 4),
        }
        log.info(
            f"[Ablation] Fold {fold_id} | sin {ag:<12} "
            f"AUC={auc_ab:.4f} (baseline={baseline_auc:.4f}, d={contribution:+.4f})"
        )

    component_ablation: Dict[str, float] = {}

    # Regime ablation: remove regime-adjusted layer and evaluate raw ranking signal.
    if "ranking_score" in df_test_scored.columns and y_te.nunique() > 1:
        try:
            component_ablation["without_regime_auc"] = float(roc_auc_score(y_te, df_test_scored.loc[y_te.index, "ranking_score"]))
            component_ablation["regime_layer_auc_delta"] = float(baseline_auc - component_ablation["without_regime_auc"])
        except Exception:
            pass

    # NLP ablation: remove FinBERT-derived columns from meta-style linear probe.
    nlp_cols = [c for c in X_train.columns if str(c).startswith("finbert_")]
    if nlp_cols and len(X_train.columns) - len(nlp_cols) >= 2:
        auc_wo_nlp = _auc(X_train.drop(columns=nlp_cols), y_tr, X_test.drop(columns=nlp_cols), y_te)
        component_ablation["without_nlp_auc"] = float(auc_wo_nlp)
        component_ablation["nlp_auc_delta"] = float(baseline_auc - auc_wo_nlp)

    # HRP ablation: compare fold return vs equal-weight return on same selected tickers.
    if fold_result is not None:
        try:
            ticker_returns = fold_result.get("ticker_returns", {}) or {}
            if ticker_returns:
                ew_ret = float(np.mean(list(ticker_returns.values())))
                hrp_ret = float(fold_result.get("strategy_cumulative_return", np.nan))
                component_ablation["without_hrp_equal_weight_return"] = ew_ret
                component_ablation["hrp_return_delta"] = float(hrp_ret - ew_ret)
        except Exception:
            pass

    result = {
        "fold": fold_id,
        "baseline_auc": round(baseline_auc, 4),
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "agents": ablation_results,
        "components": component_ablation,
    }

    out_path = Path(agents_results_dir) / f"ablation_fold{fold_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"[Ablation] Fold {fold_id} -> {out_path.name}")

    return result


def summarize_ablation(
    ablation_results: List[Dict],
    agents_results_dir: str,
) -> None:
    if not ablation_results:
        return

    per_agent: Dict[str, List[float]] = {ag: [] for ag in ABLATION_AGENTS}
    baseline_aucs: List[float] = []

    for res in ablation_results:
        if not res:
            continue
        baseline_aucs.append(res.get("baseline_auc", float("nan")))
        for ag, stats in res.get("agents", {}).items():
            contrib = stats.get("marginal_contribution", float("nan"))
            if not np.isnan(contrib):
                per_agent[ag].append(contrib)

    summary = {
        "n_folds": len(ablation_results),
        "mean_baseline_auc": float(np.nanmean(baseline_aucs)) if baseline_aucs else float("nan"),
        "agents": {},
    }

    rows = []
    for ag in ABLATION_AGENTS:
        contribs = per_agent[ag]
        if not contribs:
            continue
        mean_c = float(np.mean(contribs))
        std_c = float(np.std(contribs))
        n_pos = int(sum(1 for c in contribs if c > 0))
        summary["agents"][ag] = {
            "mean_contribution": round(mean_c, 4),
            "std_contribution": round(std_c, 4),
            "pct_folds_positive": round(n_pos / len(contribs), 3),
            "n_folds": len(contribs),
        }
        rows.append({
            "agent": ag,
            "mean_contribution": mean_c,
            "std_contribution": std_c,
            "pct_folds_positive": n_pos / len(contribs),
            "n_folds": len(contribs),
        })

    out_dir = Path(agents_results_dir)
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        df = pd.DataFrame(rows).sort_values("mean_contribution", ascending=False)
        df.to_csv(out_dir / "ablation_summary.csv", index=False, float_format="%.4f")
        log.info("[Ablation] Ablation summary:")
        log.info(f"  Mean baseline AUC: {summary['mean_baseline_auc']:.4f}")
        for _, row in df.iterrows():
            log.info(
                f"  {row['agent']:<15}  dAuC medio={row['mean_contribution']:+.4f} "
                f"Â± {row['std_contribution']:.4f}  "
                f"(positivo en {row['pct_folds_positive']:.0%} de folds)"
            )

    log.info(f"[Ablation] Resumen guardado en {out_dir}/ablation_summary.{{json,csv}}")

log = logging.getLogger(__name__)

MAX_PORTFOLIO_SIZE = 20


def _compute_ticker_return(
    prices_dict: Dict[str, pd.DataFrame],
    ticker: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    max_holding_days: Optional[int] = None,
) -> Optional[float]:
    """Compute ticker return over [entry_date, exit_date] with optional TP/SL exit."""
    prices = prices_dict.get(ticker)
    if prices is None or prices.empty:
        return None

    if isinstance(prices, pd.Series):
        close = prices
    elif isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            close = prices["Close"]
        elif len(prices.columns) > 3:
            close = prices.iloc[:, 3]
        else:
            close = prices.iloc[:, 0]
    else:
        return None

    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    entry_window = close.loc[close.index >= entry_date]
    if entry_window.empty:
        entry_window = close.loc[close.index <= entry_date]
        if entry_window.empty:
            return None
        p0 = float(entry_window.iloc[-1])
    else:
        p0 = float(entry_window.iloc[0])

    exit_ts = pd.Timestamp(exit_date)

    tp = float(tp_pct) if tp_pct is not None and np.isfinite(tp_pct) else np.nan
    sl = float(sl_pct) if sl_pct is not None and np.isfinite(sl_pct) else np.nan
    if np.isfinite(tp) and np.isfinite(sl) and tp > 0.0 and sl > 0.0:
        horizon_days = int(max_holding_days) if max_holding_days is not None else int(max((exit_ts - pd.Timestamp(entry_date)).days, 1))
        sim = simulate_tp_sl(
            ticker=str(ticker),
            prices=close,
            entry_date=pd.Timestamp(entry_date),
            tp_pct=float(tp),
            sl_pct=float(sl),
            max_holding_days=int(max(horizon_days, 1)),
        )
        out_dt = sim.get("outcome_date")
        if not pd.isna(out_dt):
            exit_ts = min(exit_ts, pd.Timestamp(out_dt))

    exit_window = close.loc[close.index <= exit_ts]
    if exit_window.empty:
        return None
    p1 = float(exit_window.iloc[-1])

    if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
        return None
    return (p1 - p0) / p0


def compute_portfolio_size_alpha_matrix(
    fold_contexts: List[Dict],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark_prices,
    max_size: int = MAX_PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """Build a matrix of alpha vs benchmark by fold and portfolio size.

    Args:
        fold_contexts: List of dicts with keys:
            - fold_id (str): e.g. "2022YQ1"
            - entry_date (Timestamp)
            - exit_date (Timestamp)
            - preds_df (DataFrame): must have 'ticker' and 'score' columns
        prices_dict: ticker -> price DataFrame
        benchmark_prices: SPY (or benchmark proxy) price series/dataframe
        max_size: Maximum portfolio size (default 20)

    Returns:
        DataFrame with fold_id as index and columns 1..max_size representing
        equally weighted portfolio alpha (portfolio return - benchmark return).
    """
    rows = []

    for ctx in fold_contexts:
        fold_id = str(ctx["fold_id"])
        entry_date = pd.Timestamp(ctx["entry_date"])
        exit_date = pd.Timestamp(ctx["exit_date"])
        preds_df = ctx.get("preds_df", pd.DataFrame())

        if preds_df.empty or "score" not in preds_df.columns or "ticker" not in preds_df.columns:
            log.warning("[PortfolioSize] Fold %s: no predictions available â€” skipped", fold_id)
            continue

        # Reuse scored predictions already produced by the fold (no model rerun).
        # Rank tickers by score (descending)
        keep_cols = [c for c in ["ticker", "score", "tp_pct", "sl_pct", "max_holding_days"] if c in preds_df.columns]
        ranked = (
            preds_df[keep_cols]
            .drop_duplicates(subset=["ticker"], keep="last")
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        benchmark_return = _compute_ticker_return(
            prices_dict={"__BENCH__": benchmark_prices},
            ticker="__BENCH__",
            entry_date=entry_date,
            exit_date=exit_date,
        )
        if benchmark_return is None:
            log.warning("[PortfolioSize] Fold %s: benchmark return unavailable â€” skipped", fold_id)
            continue

        # Compute return for each ranked ticker
        ticker_returns = {}
        tp_sl_by_ticker: Dict[str, Dict[str, float]] = {}
        if "ticker" in ranked.columns:
            for _, row in ranked.iterrows():
                tk = str(row["ticker"])
                tp_sl_by_ticker[tk] = {
                    "tp_pct": float(pd.to_numeric(row.get("tp_pct", np.nan), errors="coerce")),
                    "sl_pct": float(pd.to_numeric(row.get("sl_pct", np.nan), errors="coerce")),
                    "max_holding_days": int(float(pd.to_numeric(row.get("max_holding_days", np.nan), errors="coerce"))) if np.isfinite(pd.to_numeric(row.get("max_holding_days", np.nan), errors="coerce")) else None,
                }
        for _, row in ranked.iterrows():
            tk = str(row["ticker"])
            plan = tp_sl_by_ticker.get(tk, {})
            ret = _compute_ticker_return(
                prices_dict,
                tk,
                entry_date,
                exit_date,
                tp_pct=plan.get("tp_pct"),
                sl_pct=plan.get("sl_pct"),
                max_holding_days=plan.get("max_holding_days"),
            )
            if ret is not None:
                ticker_returns[tk] = ret

        # Build equally weighted portfolios of size 1..max_size and convert to alpha.
        ordered_tickers = [
            str(row["ticker"]) for _, row in ranked.iterrows()
            if str(row["ticker"]) in ticker_returns
        ]

        row_data = {"fold": fold_id}
        for n in range(1, max_size + 1):
            if n <= len(ordered_tickers):
                selected = ordered_tickers[:n]
                portfolio_return = np.mean([ticker_returns[tk] for tk in selected])
                row_data[n] = float(portfolio_return - benchmark_return)
            else:
                row_data[n] = np.nan

        rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("fold")
    df.columns = [int(c) if isinstance(c, (int, np.integer)) else c for c in df.columns]
    return df


def save_portfolio_size_csv(matrix: pd.DataFrame, output_path: Path) -> None:
    """Save the portfolio size matrix as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, float_format="%.6f")
    log.info("[PortfolioSize] CSV matrix saved to %s", output_path)


def plot_portfolio_size_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Portfolio Size vs Alpha (Benchmark)",
) -> None:
    """Generate a heatmap showing alpha by fold and size.

    Args:
        matrix: DataFrame with folds as rows, portfolio sizes as columns.
        output_path: Path to save the PNG image.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if matrix.empty:
        log.warning("[PortfolioSize] Empty matrix â€” skipping heatmap")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = matrix.values.astype(float) * 100.0  # convert to percentages
    fold_labels = [str(f) for f in matrix.index]
    size_labels = [str(c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(size_labels) * 0.8), max(4, len(fold_labels) * 0.6)))

    # Diverging colormap centered at zero
    vmin = np.nanmin(data) if np.any(np.isfinite(data)) else -10
    vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 10
    abs_max = max(abs(vmin), abs(vmax), 1e-6)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(size_labels)))
    ax.set_xticklabels(size_labels, fontsize=8)
    ax.set_yticks(range(len(fold_labels)))
    ax.set_yticklabels(fold_labels, fontsize=8)
    ax.set_xlabel("Number of Stocks in Portfolio")
    ax.set_ylabel("Fold")
    ax.set_title(title, fontweight="bold", fontsize=12)

    # Add text annotations
    for i in range(len(fold_labels)):
        for j in range(len(size_labels)):
            val = data[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > abs_max * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Alpha (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[PortfolioSize] Heatmap saved to %s", output_path)


def run_portfolio_size_analysis(
    fold_contexts: List[Dict],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark_prices,
    output_dir: Path,
    max_size: int = MAX_PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """Run the full portfolio size alpha analysis: matrix, CSV, heatmap and summary.

    Args:
        fold_contexts: List of fold context dicts from the walk-forward pipeline.
        prices_dict: ticker -> price DataFrame.
        benchmark_prices: SPY (or benchmark proxy) price series/dataframe.
        output_dir: Directory where CSV and heatmap are saved.
        max_size: Maximum portfolio size to evaluate (default 20).

    Returns:
        The portfolio size alpha matrix.
    """
    log.info("[PortfolioSize] Running portfolio size alpha analysis (1..%d stocks)", max_size)

    matrix = compute_portfolio_size_alpha_matrix(
        fold_contexts=fold_contexts,
        prices_dict=prices_dict,
        benchmark_prices=benchmark_prices,
        max_size=max_size,
    )
    if matrix.empty:
        log.warning("[PortfolioSize] No data to analyze")
        return matrix

    output_dir = Path(output_dir)
    save_portfolio_size_csv(matrix, output_dir / "portfolio_size_alpha_vs_benchmark.csv")
    plot_portfolio_size_heatmap(
        matrix,
        output_dir / "portfolio_size_alpha_heatmap.png",
        title="Portfolio Size vs Alpha (EW Portfolio - Benchmark)",
    )

    # Summary statistics
    mean_by_size = matrix.mean(axis=0)
    summary = pd.DataFrame({
        "portfolio_size": mean_by_size.index,
        "mean_alpha": mean_by_size.values,
        "std_alpha": matrix.std(axis=0).values,
        "min_alpha": matrix.min(axis=0).values,
        "max_alpha": matrix.max(axis=0).values,
        "median_alpha": matrix.median(axis=0).values,
    })
    summary.to_csv(output_dir / "portfolio_size_alpha_summary.csv", index=False, float_format="%.6f")
    log.info("[PortfolioSize] Summary saved to %s", output_dir / "portfolio_size_alpha_summary.csv")

    return matrix



def summarize_tp_sl_vs_buy_hold_counterfactual(strategy_dir: str | Path) -> Dict:
    """Load exported TP/SL vs Buy & Hold artifacts and summarize value-add.

    This is an evaluation-only analysis helper: it consumes the CSV generated by
    the backtester and does not retrain models, rescore tickers or alter the
    selected portfolio.  It is intentionally safe to call after a run from
    notebooks, analyzer_II grids or reporting jobs.
    """
    strategy_path = Path(strategy_dir)
    fold_csv = strategy_path / "tp_sl_vs_buy_hold_by_fold.csv"
    ticker_csv = strategy_path / "tp_sl_vs_buy_hold_by_ticker.csv"
    if not fold_csv.exists():
        return {"available": False, "reason": f"missing {fold_csv}"}

    folds = pd.read_csv(fold_csv)
    tickers = pd.read_csv(ticker_csv) if ticker_csv.exists() else pd.DataFrame()
    delta = pd.to_numeric(folds.get("tp_sl_minus_buy_hold", pd.Series(dtype=float)), errors="coerce")
    alpha_tp = pd.to_numeric(folds.get("alpha_tp_sl_vs_benchmark", pd.Series(dtype=float)), errors="coerce")
    alpha_bh = pd.to_numeric(folds.get("alpha_buy_hold_vs_benchmark", pd.Series(dtype=float)), errors="coerce")

    summary = {
        "available": True,
        "n_folds": int(len(folds)),
        "folds_tp_sl_wins": int((delta > 0).sum()),
        "folds_buy_hold_wins": int((delta < 0).sum()),
        "mean_alpha_tp_sl": float(alpha_tp.mean()) if len(alpha_tp) else float("nan"),
        "mean_alpha_buy_hold": float(alpha_bh.mean()) if len(alpha_bh) else float("nan"),
        "mean_tp_sl_minus_buy_hold": float(delta.mean()) if len(delta) else float("nan"),
        "median_tp_sl_minus_buy_hold": float(delta.median()) if len(delta) else float("nan"),
    }
    if not tickers.empty and "tp_sl_improved" in tickers.columns:
        improved = tickers["tp_sl_improved"].astype(str).str.lower().isin({"true", "1", "yes"})
        summary["ticker_level_tp_sl_win_rate"] = float(improved.mean()) if len(improved) else float("nan")
        if "tp_sl_exit_reason" in tickers.columns:
            summary["exit_reason_counts"] = tickers["tp_sl_exit_reason"].fillna("unknown").value_counts().to_dict()
    return summary
