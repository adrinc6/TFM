# =============================================================================
# module/backtester.py — Walk-Forward Backtesting + Métricas Financieras
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

log = logging.getLogger(__name__)


# ── Métricas financieras ──────────────────────────────────────────────────────

def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    cr = cumulative_return(returns)
    # Si la pérdida acumulada supera el 100%, la base (1+cr) es negativa.
    # Elevar un número negativo a una potencia fraccionaria produce un número
    # complejo en Python → devolver -1.0 (pérdida total) como caso límite.
    base = 1 + cr
    if base <= 0:
        return -1.0
    return float(base ** (periods_per_year / n) - 1)

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04,
                 periods_per_year: int = 252) -> float:
    if returns.std() == 0: return 0.0
    excess = returns - risk_free / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))

def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return float(dd.min())

def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0: return 0.0
    return float(annualized_return(returns, periods_per_year) / mdd)

def sortino_ratio(returns: pd.Series, risk_free: float = 0.04,
                  periods_per_year: int = 252) -> float:
    excess    = returns - risk_free / periods_per_year
    downside  = excess[excess < 0].std()
    if downside == 0: return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))

def hit_rate(predictions: pd.Series, labels: pd.Series) -> float:
    """% de predicciones Outperform que realmente superaron al benchmark."""
    mask = predictions == 1
    if mask.sum() == 0: return 0.0
    return float(labels[mask].mean())

def compute_all_metrics(
    returns: pd.Series, risk_free: float = 0.04, label: str = "strategy"
) -> Dict:
    return {
        f"{label}_cumulative_return":  cumulative_return(returns),
        f"{label}_sharpe":             sharpe_ratio(returns, risk_free),
        f"{label}_sortino":            sortino_ratio(returns, risk_free),
        f"{label}_max_drawdown":       max_drawdown(returns),
        f"{label}_calmar":             calmar_ratio(returns),
        f"{label}_volatility":         float(returns.std() * np.sqrt(252)),
        f"{label}_n_periods":          len(returns),
    }


# ── Walk-Forward Backtester ───────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward backtesting con paso trimestral y test de 1 quarter.

    Genera dos familias de folds:

    ROLLING (ventana fija de train_years, paso de 1 quarter):
        Cada fold: train = exactamente train_years, test = 1 quarter siguiente.
        Ejemplo con train=3Y, datos 2019-2026:
            train 2019-Q1 → 2022-Q1,  test 2022-Q1→Q2
            train 2019-Q2 → 2022-Q2,  test 2022-Q2→Q3
            train 2019-Q3 → 2022-Q3,  test 2022-Q3→Q4
            ... (un fold por quarter disponible)

    EXPANDING (inicio anclado, train crece en años enteros, test desliza por quarters):
        Solo se añade un nuevo grupo expanding cuando se cumple un año completo
        adicional de datos (manteniendo siempre duración en años enteros).
        Para cada year-expansion, el test desliza quarter a quarter dentro
        de ese año de test antes de que el train vuelva a crecer.
        Ejemplo con train_min=3Y:
            train 2019→2023 (4Y):  test 2023-Q1, 2023-Q2, 2023-Q3, 2023-Q4
            train 2019→2024 (5Y):  test 2024-Q1, 2024-Q2, 2024-Q3, 2024-Q4
            train 2019→2025 (6Y):  test 2025-Q1, 2025-Q2, 2025-Q3, 2025-Q4

    Los duplicados (el primer expanding = un rolling) se eliminan.
    El orden final es cronológico por (test_end, train_start).
    """

    def __init__(
        self,
        train_years:    int   = 3,
        test_quarters:  int   = 1,
        risk_free:      float = 0.04,
        results_dir:    str   = "results/backtest",
        top_n_stocks:   int   = 10,
        long_only:      bool  = True,
    ):
        self.train_years   = train_years
        self.test_quarters = test_quarters          # siempre 1 en el uso estándar
        self.risk_free     = risk_free
        self.results_dir   = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.top_n_stocks  = top_n_stocks
        self.long_only     = long_only
        self.fold_results: List[Dict]  = []
        self.all_strategy_returns      = pd.Series(dtype=float)
        self.all_benchmark_returns     = pd.Series(dtype=float)

    # ── Generación de folds ───────────────────────────────────────────────────

    def generate_folds(self, start_date: str, end_date: str) -> List[tuple]:
        """
        Para cada longitud de train en años enteros (train_years, train_years+1, ...),
        genera todos los folds rolling trimestrales posibles dentro del rango de datos.

        Devuelve lista de tuplas (train_start, train_end, test_end, train_years_int).
          - train_end  == test_start
          - test_end   == test_start + test_quarters * 3 meses
          - train_end - train_start == exactamente N años enteros
          - train_start >= start_date, test_end <= end_date

        Orden final: (test_end, train_years_int, train_start) — cronológico.
        """
        start  = pd.Timestamp(start_date)
        end    = pd.Timestamp(end_date)
        q_step = pd.DateOffset(months=3 * self.test_quarters)

        folds: List[tuple] = []
        seen:  set         = set()

        # Para cada longitud de train posible en años enteros
        n_years = self.train_years
        while True:
            y_offset  = pd.DateOffset(years=n_years)
            # Primer train_end posible con este n_years: start + n_years
            train_end = start + y_offset
            found_any = False
            while True:
                test_end    = train_end + q_step
                train_start = train_end - y_offset        # exactamente n_years atrás
                if test_end > end:
                    break
                if train_start >= start:
                    key = (train_start.date(), train_end.date(), test_end.date())
                    if key not in seen:
                        seen.add(key)
                        folds.append((train_start, train_end, test_end, n_years))
                        found_any = True
                train_end += q_step
            # Si no cabe ningún fold con esta longitud, no habrá con más años tampoco
            if not found_any:
                break
            n_years += 1

        # Orden cronológico: test_end → train_years → train_start
        folds.sort(key=lambda x: (x[2], x[3], x[0]))

        # Log resumen por train_years
        from collections import Counter
        counts = Counter(f[3] for f in folds)
        log.info(
            f"[Backtester] {len(folds)} folds generados | test={self.test_quarters}Q | "
            + " | ".join(f"{n}Y→{c}folds" for n, c in sorted(counts.items()))
        )
        for i, (ts, te, tse, ny) in enumerate(folds):
            log.debug(
                f"  [{i:02d}] {ny}Y train | "
                f"{ts.date()} → {te.date()} | test {te.date()} → {tse.date()}"
            )
        return folds

    # ── Simulación de cartera ─────────────────────────────────────────────────

    def simulate_portfolio(self, predictions_df, prices_dict, benchmark,
                           fold_id, test_start, test_end,
                           train_start=None, train_years_int: int = 0):
        top = (predictions_df
               .sort_values("score", ascending=False)
               .head(self.top_n_stocks)["ticker"].tolist())
        if not top:
            log.warning(f"[Backtester] Fold {fold_id}: sin selección de stocks")
            return {}

        daily_returns   = []
        ticker_returns  = {}   # {ticker: cumulative_return}
        bench_period    = benchmark.loc[test_start:test_end].dropna()

        for ticker in top:
            if ticker not in prices_dict:
                continue
            prices = prices_dict[ticker]
            cc     = "Close" if "Close" in prices.columns else prices.columns[3]
            period = prices.loc[test_start:test_end, cc]
            if len(period) < 2:
                continue
            ret = period.pct_change().dropna()
            daily_returns.append(ret)
            ticker_returns[ticker] = round(float((1 + ret).prod() - 1), 6)

        if not daily_returns:
            return {}

        strat_returns = pd.concat(daily_returns, axis=1).mean(axis=1).dropna()
        common_idx    = strat_returns.index.intersection(bench_period.index)
        strat_aligned = strat_returns.loc[common_idx]
        bench_aligned = bench_period.loc[common_idx]

        strat_metrics = compute_all_metrics(strat_aligned, self.risk_free, "strategy")
        bench_metrics = compute_all_metrics(bench_aligned, self.risk_free, "benchmark")
        alpha         = (strat_metrics["strategy_cumulative_return"]
                         - bench_metrics["benchmark_cumulative_return"])
        excess_sharpe = (strat_metrics["strategy_sharpe"]
                         - bench_metrics["benchmark_sharpe"])

        ticker_returns_sorted = dict(
            sorted(ticker_returns.items(), key=lambda x: x[1], reverse=True)
        )

        fold_result = {
            "fold":             fold_id,
            "train_years":      train_years_int,
            "train_start":      str(train_start.date()) if train_start is not None else None,
            "test_start":       str(test_start.date()),
            "test_end":         str(test_end.date()),
            "selected_tickers": top,
            "n_stocks":         len(top),
            **strat_metrics, **bench_metrics,
            "alpha":            alpha,
            "excess_sharpe":    excess_sharpe,
            "ticker_returns":   ticker_returns_sorted,
        }

        self.all_strategy_returns  = pd.concat([self.all_strategy_returns,  strat_aligned])
        self.all_benchmark_returns = pd.concat([self.all_benchmark_returns, bench_aligned])

        path = self.results_dir / f"fold_{fold_id:03d}_{train_years_int}Y_metrics.json"
        with open(path, "w") as f:
            json.dump(fold_result, f, indent=2, default=str)

        log.info(
            f"[Backtester] Fold {fold_id:03d} [{train_years_int}Y train] "
            f"test {test_start.date()}→{test_end.date()} | "
            f"Strat={strat_metrics['strategy_cumulative_return']:.2%} "
            f"Bench={bench_metrics['benchmark_cumulative_return']:.2%} "
            f"α={alpha:.2%} Sharpe={strat_metrics['strategy_sharpe']:.2f}"
        )
        return fold_result

    # ── Resumen global ────────────────────────────────────────────────────────

    def summarize(self) -> Dict:
        if not self.fold_results:
            log.warning("[Backtester] Sin folds completados.")
            return {}

        global_strat  = compute_all_metrics(self.all_strategy_returns,  self.risk_free, "global_strategy")
        global_bench  = compute_all_metrics(self.all_benchmark_returns, self.risk_free, "global_benchmark")
        folds_df      = pd.DataFrame(self.fold_results)
        mean_alpha    = float(folds_df["alpha"].mean())        if "alpha" in folds_df else 0.0
        pct_alpha_pos = float((folds_df["alpha"] > 0).mean()) if "alpha" in folds_df else 0.0

        # Desglose por train_years
        by_train_years: Dict = {}
        if "train_years" in folds_df.columns:
            for ny, grp in folds_df.groupby("train_years"):
                by_train_years[int(ny)] = {
                    "n_folds":              int(len(grp)),
                    "mean_alpha":           float(grp["alpha"].mean()),
                    "pct_positive_alpha":   float((grp["alpha"] > 0).mean()),
                    "mean_strategy_sharpe": float(grp["strategy_sharpe"].mean())
                                            if "strategy_sharpe" in grp else 0.0,
                    "mean_strategy_return": float(grp["strategy_cumulative_return"].mean())
                                            if "strategy_cumulative_return" in grp else 0.0,
                }

        summary = {
            "timestamp":               datetime.now().isoformat(),
            "n_folds":                 len(self.fold_results),
            "train_years_min":         self.train_years,
            "test_quarters":           self.test_quarters,
            "top_n_stocks":            self.top_n_stocks,
            "mean_alpha":              mean_alpha,
            "pct_folds_positive_alpha": pct_alpha_pos,
            "by_train_years":          by_train_years,
            **global_strat, **global_bench,
        }

        path = self.results_dir / "backtest_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        pd.DataFrame({
            "strategy":  self.all_strategy_returns,
            "benchmark": self.all_benchmark_returns,
        }).to_csv(self.results_dir / "returns_series.csv")

        log.info("=" * 65)
        log.info(f"  Folds totales:     {len(self.fold_results)}")
        for ny, stats in sorted(by_train_years.items()):
            log.info(
                f"  [{ny}Y train] n={stats['n_folds']:3d} | "
                f"ret={stats['mean_strategy_return']:+.2%} | "
                f"α={stats['mean_alpha']:+.2%} | "
                f"α_pos={stats['pct_positive_alpha']:.0%} | "
                f"Sharpe={stats['mean_strategy_sharpe']:.3f}"
            )
        log.info(f"  Alpha medio total: {mean_alpha:+.2%}")
        log.info(f"  Folds α positivo:  {pct_alpha_pos:.0%}")
        log.info(f"  Sharpe Estrategia: {global_strat.get('global_strategy_sharpe', 0):.3f}")
        log.info(f"  Sharpe Benchmark:  {global_bench.get('global_benchmark_sharpe', 0):.3f}")
        log.info(f"  Max DD Estrategia: {global_strat.get('global_strategy_max_drawdown', 0):.2%}")
        log.info("=" * 65)
        return summary

    # ── CSV + Plot de resultados de todos los folds ───────────────────────────

    def save_folds_summary(self, plots_dir: str = "results/plots"):
        """
        Genera un CSV y un dashboard de resultados con las métricas reales
        obtenidas en cada fold tras el test.

        Artefactos:
          1. folds_results.csv       — una fila por fold con todas las métricas.
          2. folds_results.png       — dashboard 4-panel:
               [A] Alpha por fold (barras, coloreadas por train_years)
               [B] Retorno anualizado estrategia vs benchmark por fold
               [C] Sharpe ratio por fold
               [D] Distribución de alpha por train_years (boxplot)
        """
        if not self.fold_results:
            log.warning("[Backtester] save_folds_summary: sin resultados todavía.")
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        plots_path = Path(plots_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        # ── 1. CSV de resultados ──────────────────────────────────────────────
        df = pd.DataFrame(self.fold_results)

        # Columnas limpias y ordenadas para el CSV
        col_order = [
            "fold", "train_years", "train_start", "test_start", "test_end",
            "strategy_cumulative_return", "benchmark_cumulative_return", "alpha",
            "strategy_sharpe", "benchmark_sharpe", "excess_sharpe",
            "strategy_sortino", "strategy_max_drawdown", "strategy_calmar",
            "strategy_volatility", "n_stocks",
        ]
        existing_cols = [c for c in col_order if c in df.columns]
        df_csv = df[existing_cols].copy()

        # Etiqueta legible para el período de test
        if "test_start" in df_csv.columns and "test_end" in df_csv.columns:
            df_csv.insert(
                df_csv.columns.get_loc("test_start") + 1,
                "test_period",
                df_csv["test_start"].astype(str) + " → " + df_csv["test_end"].astype(str),
            )

        csv_path = self.results_dir / "folds_results.csv"
        df_csv.to_csv(csv_path, index=False, float_format="%.4f")
        log.info(f"[Backtester] Folds results CSV → {csv_path}")

        # ── 2. Dashboard de resultados ────────────────────────────────────────
        all_train_years = sorted(df["train_years"].unique()) if "train_years" in df.columns else []
        palette   = plt.cm.tab10.colors
        color_map = {ny: palette[i % len(palette)] for i, ny in enumerate(all_train_years)}

        # Etiqueta corta por fold: "Q1'23 (3Y)"
        def fold_label(row):
            ts = pd.Timestamp(row["test_start"])
            q  = (ts.month - 1) // 3 + 1
            return f"Q{q}'{str(ts.year)[2:]} ({int(row.get('train_years', 0))}Y)"

        df["label"]      = df.apply(fold_label, axis=1)
        bar_colors       = [color_map.get(int(ny), "steelblue")
                            for ny in df.get("train_years", ["?"] * len(df))]
        alpha_vals       = df["alpha"].values if "alpha" in df.columns else []
        strat_ret        = df["strategy_cumulative_return"].values if "strategy_cumulative_return" in df.columns else []
        bench_ret        = df["benchmark_cumulative_return"].values if "benchmark_cumulative_return" in df.columns else []
        strat_sharpe     = df["strategy_sharpe"].values if "strategy_sharpe" in df.columns else []
        bench_sharpe     = df["benchmark_sharpe"].values if "benchmark_sharpe" in df.columns else []
        x                = np.arange(len(df))

        fig, axes = plt.subplots(2, 2, figsize=(18, 11))
        fig.suptitle(
            f"Resultados Walk-Forward — {len(df)} folds | "
            f"α medio={float(df['alpha'].mean()):+.2%} | "
            f"α positivo={float((df['alpha']>0).mean()):.0%}",
            fontsize=13, fontweight="bold", y=0.99,
        )
        ax_a, ax_b, ax_c, ax_d = axes.flat

        # ── Panel A: Alpha por fold ───────────────────────────────────────────
        bars = ax_a.bar(x, alpha_vals * 100, color=bar_colors, alpha=0.85, width=0.7)
        ax_a.axhline(0, color="black", lw=0.8)
        ax_a.axhline(float(np.mean(alpha_vals) * 100), color="crimson",
                     lw=1.2, ls="--", label=f"Media {float(np.mean(alpha_vals))*100:+.1f}%")
        # Colorear positivo/negativo más oscuro
        for bar, val in zip(bars, alpha_vals):
            bar.set_edgecolor("darkgreen" if val >= 0 else "darkred")
            bar.set_linewidth(0.7)
        ax_a.set_title("Alpha por Fold (Estrategia − Benchmark)", fontweight="bold")
        ax_a.set_ylabel("Alpha anualizado (%)")
        ax_a.set_xticks(x); ax_a.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7)
        ax_a.legend(fontsize=8); ax_a.grid(axis="y", alpha=0.3)

        # ── Panel B: Retorno anualizado estrategia vs benchmark ───────────────
        w = 0.35
        ax_b.bar(x - w/2, strat_ret * 100, w, label="Estrategia",
                 color=[color_map.get(int(ny), "steelblue") for ny in df.get("train_years", [0]*len(df))],
                 alpha=0.85)
        ax_b.bar(x + w/2, bench_ret * 100, w, label="Benchmark",
                 color="#FF5722", alpha=0.65)
        ax_b.axhline(0, color="black", lw=0.8)
        ax_b.set_title("Retorno Acumulado por Fold (%)", fontweight="bold")
        ax_b.set_ylabel("Retorno acumulado (%)")
        ax_b.set_xticks(x); ax_b.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7)
        ax_b.legend(fontsize=8); ax_b.grid(axis="y", alpha=0.3)

        # ── Panel C: Sharpe ratio ─────────────────────────────────────────────
        ax_c.plot(x, strat_sharpe, "o-", color="#2196F3", lw=2, ms=6, label="Estrategia")
        ax_c.plot(x, bench_sharpe, "s--", color="#FF5722", lw=1.5, ms=5, label="Benchmark")
        ax_c.axhline(0, color="black", lw=0.8)
        ax_c.axhline(1, color="green", lw=0.8, ls=":", alpha=0.6, label="Sharpe=1")
        ax_c.fill_between(x, strat_sharpe, bench_sharpe,
                          where=np.array(strat_sharpe) >= np.array(bench_sharpe),
                          alpha=0.12, color="green", interpolate=True)
        ax_c.fill_between(x, strat_sharpe, bench_sharpe,
                          where=np.array(strat_sharpe) < np.array(bench_sharpe),
                          alpha=0.12, color="red", interpolate=True)
        ax_c.set_title("Sharpe Ratio por Fold", fontweight="bold")
        ax_c.set_ylabel("Sharpe Ratio")
        ax_c.set_xticks(x); ax_c.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=7)
        ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)

        # ── Panel D: Boxplot de alpha por train_years ─────────────────────────
        if all_train_years and "train_years" in df.columns:
            box_data   = [df[df["train_years"] == ny]["alpha"].values * 100
                          for ny in all_train_years]
            bp = ax_d.boxplot(box_data, patch_artist=True, widths=0.5,
                               medianprops=dict(color="black", lw=2))
            for patch, ny in zip(bp["boxes"], all_train_years):
                patch.set_facecolor(color_map[ny])
                patch.set_alpha(0.7)
            ax_d.axhline(0, color="black", lw=0.8, ls="--")
            ax_d.set_xticklabels([f"{ny}Y train\n(n={len(df[df['train_years']==ny])})"
                                   for ny in all_train_years], fontsize=9)
            ax_d.set_title("Distribución de Alpha por Longitud de Train", fontweight="bold")
            ax_d.set_ylabel("Alpha anualizado (%)")
            ax_d.grid(axis="y", alpha=0.3)
        else:
            ax_d.axis("off")

        # Leyenda global de colores de train_years
        patches = [mpatches.Patch(color=color_map[ny], label=f"{ny}Y train")
                   for ny in all_train_years]
        fig.legend(handles=patches, loc="lower center", ncol=len(all_train_years),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_path = plots_path / "folds_results.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"[Backtester] Folds results plot → {plot_path}")

        return csv_path, plot_path
