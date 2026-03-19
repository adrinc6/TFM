# =============================================================================
# module/pipeline/evaluator.py
# Evaluación por fold: SHAP, backtesting, diagnósticos y gráficos.
# =============================================================================
"""
Responsabilidades:
  - explain_top_tickers: genera explicaciones SHAP para los top/bottom tickers.
  - run_walkforward_pipeline: orquesta el bucle completo de walk-forward,
    llamando a trainer y backtester en cada fold y generando el resumen global.
"""
import datetime
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

from module.backtester import WalkForwardBacktester
from module.visualizer import Visualizer
from module.feature_engineering import SectorNormalizer
from module.pipeline.dataset_builder import apply_sector_normalization
from module.pipeline.trainer import train_fold
from module.pipeline.ablation import run_ablation_study, summarize_ablation

log = logging.getLogger(__name__)


def explain_top_tickers(
    agents: Dict,
    df_test: pd.DataFrame,
    scores: pd.Series,
    fold_id: int,
    agents_results_dir: str,
    top_n: int = 10,
) -> None:
    """
    Genera explicaciones SHAP para los top_n tickers con mayor y menor score
    del fold, y las guarda en un JSON consolidado.

    Args:
        agents: Diccionario {nombre: agente_entrenado}.
        df_test: DataFrame de test con features.
        scores: Series con el final_score por observación.
        fold_id: Número de fold para nombrar el fichero.
        agents_results_dir: Directorio donde guardar el JSON.
        top_n: Número de tickers bull y bear a explicar.
    """
    if scores.empty:
        return

    tickers_col = df_test.index.get_level_values("ticker")
    ticker_scores = pd.Series(scores.values, index=tickers_col).groupby(level=0).last()

    top_bulls = ticker_scores.nlargest(top_n).index.tolist()
    top_bears = ticker_scores.nsmallest(top_n).index.tolist()
    selected = list(dict.fromkeys(top_bulls + top_bears))

    results_path = Path(agents_results_dir) / f"fold_{fold_id}_ticker_explanations.json"
    all_explanations = {"fold": fold_id, "tickers": {}}

    for ticker in selected:
        mask = tickers_col == ticker
        if not mask.any():
            continue
        row = df_test.loc[mask].iloc[-1]
        score = float(ticker_scores.get(ticker, 0.5))

        ticker_exp = {
            "score": round(score, 4),
            "label": "Outperform" if score >= 0.5 else "Underperform",
            "agents": {},
        }

        for ag_name, ag in agents.items():
            if ag._explainer is None:
                continue
            try:
                exp = ag._explainer.explain_prediction(row, ticker, score, top_n=6, fold=fold_id)
                ticker_exp["agents"][ag_name] = {
                    "text": exp.get("text", ""),
                    "top_drivers": exp.get("top_drivers", [])[:6],
                }
            except Exception as ex:
                log.debug(f"Explain {ag_name}/{ticker}: {ex}")

        all_explanations["tickers"][ticker] = ticker_exp
        log.info(
            f"  [{'Outperform' if score >= 0.5 else 'Underperform':11s}] "
            f"{ticker:<6} score={score:.3f}"
        )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False, default=str)

    log.info(
        f"[Explainer] Fold {fold_id}: explicaciones de {len(selected)} tickers "
        f"-> {results_path.name}"
    )


def run_walkforward_pipeline(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark: pd.Series,
    agents_results_dir: str,
    backtest_results_dir: str,
    plots_dir: str,
    start_date: str,
    end_date: str,
    walkforward_train_years: int,
    walkforward_test_quarters: int,
    risk_free_rate: float,
    random_seed: int = 42,
) -> Dict:
    """
    Ejecuta el pipeline completo de walk-forward:
      1. Genera folds temporales.
      2. En cada fold: separa train/test, normaliza, entrena agentes, evalúa.
      3. Simula cartera y calcula métricas por fold.
      4. Genera resumen global y gráficos finales.

    Args:
        df: Dataset maestro multi-índice (ticker, date).
        sector_map: Diccionario {ticker: sector}.
        prices_dict: Diccionario {ticker: DataFrame OHLCV} para el backtester.
        benchmark: Serie de retornos diarios del S&P 500.
        agents_results_dir: Directorio para diagnósticos de agentes.
        backtest_results_dir: Directorio para métricas de backtest.
        plots_dir: Directorio para gráficos.
        start_date / end_date: Rango de fechas del pipeline.
        walkforward_train_years: Mínimo años de entrenamiento.
        walkforward_test_quarters: Trimestres de test por fold.
        risk_free_rate: Tasa libre de riesgo anualizada.
        random_seed: Semilla de reproducibilidad.

    Returns:
        Diccionario con métricas globales del backtest.
    """
    backtester = WalkForwardBacktester(
        train_years=walkforward_train_years,
        test_quarters=walkforward_test_quarters,
        risk_free=risk_free_rate,
        results_dir=backtest_results_dir,
    )
    visualizer = Visualizer(plots_dir=plots_dir)
    normalizer = SectorNormalizer()

    # Pasar forward_return_days para verificar coherencia con el período de test
    from environment import FORWARD_RETURN_DAYS
    folds = backtester.generate_folds(start_date, end_date, forward_return_days=FORWARD_RETURN_DAYS)
    agent_diag_history: Dict[str, List] = {
        "fundamental": [], "valuation": [], "momentum": [], "bear": [], "meta_learner": []
    }
    ablation_results: List[Dict] = []

    dates = df.index.get_level_values("date")

    for fold_id, (train_start, train_end, test_end, _train_years) in enumerate(folds, 1):
        log.info(f"\n{'='*60}")
        log.info(
            f"  FOLD {fold_id} | Train: {train_start.date()} -> {train_end.date()} "
            f"| Test: {train_end.date()} -> {test_end.date()}"
        )
        log.info(f"{'='*60}")

        # ── Separar train / test
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        df_train = df.loc[train_mask]
        df_test = df.loc[test_mask]

        if len(df_train) < 100:
            log.warning(f"Fold {fold_id}: Train insuficiente ({len(df_train)} obs). Skip.")
            continue

        df_train = df_train[~df_train.index.duplicated(keep="last")]
        df_test = df_test[~df_test.index.duplicated(keep="last")]

        # ── Normalización sectorial (fit solo en train)
        df_train_norm = apply_sector_normalization(df_train, sector_map, normalizer, fit=True)
        df_test_norm = apply_sector_normalization(df_test, sector_map, normalizer, fit=False)
        df_train_norm = df_train_norm[~df_train_norm.index.duplicated(keep="last")]
        df_test_norm = df_test_norm[~df_test_norm.index.duplicated(keep="last")]

        # ── Labels binarios (1 = Outperform vs mediana del período de train)
        median_ret = df_train["forward_return"].median()
        forward_train = df_train["forward_return"].reindex(df_train_norm.index)
        forward_test = df_test["forward_return"].reindex(df_test_norm.index)
        y_train = (forward_train > median_ret).astype(int).dropna()
        y_test = (forward_test > median_ret).astype(int).dropna()
        df_train_norm = df_train_norm.loc[y_train.index]
        df_test_norm = df_test_norm.loc[y_test.index]

        try:
            # ── Entrenar y predecir
            agents, df_test_scored, df_train_with_oof = train_fold(
                df_train_norm=df_train_norm,
                df_test_norm=df_test_norm,
                y_train=y_train,
                y_test=y_test,
                fold_id=fold_id,
                agents_results_dir=agents_results_dir,
                random_seed=random_seed,
            )

            # ── Evaluar meta-learner
            meta = agents["meta_learner"]
            eval_metrics = meta.evaluate(df_test_scored, y_test, fold=fold_id)

            # ── Plots de distribución de scores
            visualizer.plot_score_distribution(df_test_scored, fold=fold_id)

            # ── Preparar predicciones para el backtester
            preds_df = df_test_scored[["final_score", "label"]].copy()
            preds_df["ticker"] = preds_df.index.get_level_values("ticker")
            preds_df["date"] = preds_df.index.get_level_values("date")
            preds_df = preds_df.reset_index(drop=True)

            # ── Simular cartera
            fold_result = backtester.simulate_portfolio(
                predictions_df=preds_df.rename(columns={"final_score": "score"}),
                prices_dict=prices_dict,
                benchmark=benchmark,
                fold_id=fold_id,
                test_start=train_end,
                test_end=test_end,
                train_start=train_start,
                train_years_int=_train_years,
            )
            fold_result.update(eval_metrics)
            backtester.fold_results.append(fold_result)

            # ── Acumular diagnósticos
            for ag_name, ag in agents.items():
                agent_diag_history[ag_name].append(ag._diagnostics.copy())

            # ── Feature importances por agente
            for ag_name in ["fundamental", "valuation", "momentum", "bear"]:
                ag = agents[ag_name]
                if hasattr(ag, "_feature_cols") and ag.is_trained:
                    try:
                        imp_path = (
                            Path(agents_results_dir) / ag_name
                            / f"feature_importances_fold{fold_id}.csv"
                        )
                        if imp_path.exists():
                            imp = pd.read_csv(imp_path, index_col=0)["importance"]
                            visualizer.plot_feature_importances(imp, ag_name, fold=fold_id)
                    except Exception as plot_exc:
                        log.debug(f"[Visualizer] plot_feature_importances {ag_name} fold {fold_id}: {plot_exc}")

            # ── Explicaciones SHAP top/bottom tickers
            explain_top_tickers(
                agents=agents,
                df_test=df_test_scored,
                scores=df_test_scored["final_score"],
                fold_id=fold_id,
                agents_results_dir=agents_results_dir,
            )

            # ── Ablation study: contribución marginal de cada agente
            abl = run_ablation_study(
                df_test_scored=df_test_scored,
                y_test=y_test,
                df_train_norm=df_train_with_oof,
                y_train=y_train,
                agents_results_dir=agents_results_dir,
                fold_id=fold_id,
                random_seed=random_seed,
            )
            if abl:
                ablation_results.append(abl)

        except Exception as e:
            log.error(f"Fold {fold_id} falló: {e}", exc_info=True)
            continue

    # ── Resumen global
    summary = backtester.summarize()
    backtester.save_folds_summary(plots_dir=plots_dir)

    # ── Guardar diagnósticos históricos
    diag_path = Path(agents_results_dir) / "all_folds_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(agent_diag_history, f, indent=2, default=str)

    # ── Gráficas finales
    last_agent_diag = {k: v[-1] if v else {} for k, v in agent_diag_history.items()}
    visualizer.plot_full_report(
        strategy_returns=backtester.all_strategy_returns,
        benchmark_returns=backtester.all_benchmark_returns,
        fold_results=backtester.fold_results,
        agent_diagnostics=last_agent_diag,
    )

    # ── Resumen del ablation study
    if ablation_results:
        summarize_ablation(ablation_results, agents_results_dir=agents_results_dir)

    # ── Informe de texto legible
    generate_text_report(
        summary=summary,
        fold_results=backtester.fold_results,
        agent_diag_history=agent_diag_history,
        backtest_results_dir=backtest_results_dir,
    )

    return summary


def generate_text_report(
    summary: Dict,
    fold_results: List[Dict],
    agent_diag_history: Dict[str, List],
    backtest_results_dir: str,
) -> None:
    """
    Genera un informe de texto plano legible con las métricas clave del backtest
    y lo guarda en <backtest_results_dir>/report.txt.

    El informe incluye:
      - Resumen global de métricas de cartera vs benchmark
      - Tabla de resultados por fold (período, alpha, Sharpe, retorno)
      - AUC de cada agente en el último fold entrenado

    Args:
        summary: Diccionario de métricas globales devuelto por backtester.summarize().
        fold_results: Lista de diccionarios con métricas de cada fold.
        agent_diag_history: Diccionario {agente: [diag_fold_1, diag_fold_2, ...]}.
        backtest_results_dir: Directorio donde guardar el informe.
    """
    lines = []
    sep = "=" * 65
    sep_s = "-" * 65

    lines.append(sep)
    lines.append("  INFORME DE RESULTADOS — Walk-Forward Backtest")
    lines.append(f"  Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # ── Resumen global ────────────────────────────────────────────────────────
    lines.append("\n  MÉTRICAS GLOBALES (todos los folds concatenados)")
    lines.append(sep_s)
    lines.append(f"  Folds completados:         {summary.get('n_folds', 0)}")
    lines.append(f"  Alpha medio por fold:      {summary.get('mean_alpha', 0):+.2%}")
    lines.append(f"  Folds con alpha positivo:  {summary.get('pct_folds_positive_alpha', 0):.0%}")

    gs = summary.get("global_strategy_sharpe", summary.get("global_strategy_sharpe", 0))
    gb = summary.get("global_benchmark_sharpe", 0)
    lines.append(f"  Sharpe estrategia:         {gs:.3f}")
    lines.append(f"  Sharpe benchmark (S&P500): {gb:.3f}")
    lines.append(f"  Sortino estrategia:        {summary.get('global_strategy_sortino', 0):.3f}")
    lines.append(f"  Max Drawdown estrategia:   {summary.get('global_strategy_max_drawdown', 0):.2%}")
    lines.append(f"  Max Drawdown benchmark:    {summary.get('global_benchmark_max_drawdown', 0):.2%}")
    lines.append(f"  Calmar ratio:              {summary.get('global_strategy_calmar', 0):.3f}")
    lines.append(f"  Volatilidad anualizada:    {summary.get('global_strategy_volatility', 0):.2%}")

    # ── Tabla por fold ────────────────────────────────────────────────────────
    if fold_results:
        lines.append(f"\n  DETALLE POR FOLD")
        lines.append(sep_s)
        header = (
            f"  {'Fold':>4}  {'Train':>4}Y  "
            f"{'Período Test':<24}  "
            f"{'Ret Strat':>9}  {'Ret Bench':>9}  "
            f"{'Alpha':>7}  {'Sharpe':>6}  {'AUC':>6}"
        )
        lines.append(header)
        lines.append("  " + "-" * 63)
        for fr in fold_results:
            test_period = f"{fr.get('test_start','')} → {fr.get('test_end','')}"
            strat_ret   = fr.get("strategy_cumulative_return", 0)
            bench_ret   = fr.get("benchmark_cumulative_return", 0)
            alpha_v     = fr.get("alpha", 0)
            sharpe_v    = fr.get("strategy_sharpe", 0)
            auc_v       = fr.get("roc_auc", fr.get("auc", float("nan")))
            train_y     = fr.get("train_years", "?")
            fold_id     = fr.get("fold", "?")
            auc_str     = f"{auc_v:.3f}" if isinstance(auc_v, float) and not pd.isna(auc_v) else "  N/A"
            lines.append(
                f"  {fold_id:>4}  {train_y:>4}Y  "
                f"{test_period:<24}  "
                f"{strat_ret:>+9.2%}  {bench_ret:>+9.2%}  "
                f"{alpha_v:>+7.2%}  {sharpe_v:>6.3f}  {auc_str:>6}"
            )

    # ── Desglose por longitud de train ────────────────────────────────────────
    by_train = summary.get("by_train_years", {})
    if by_train:
        lines.append(f"\n  DESGLOSE POR LONGITUD DE TRAIN")
        lines.append(sep_s)
        lines.append(f"  {'Train':>5}  {'N folds':>7}  {'Ret medio':>9}  {'Alpha medio':>11}  {'α>0':>5}  {'Sharpe':>6}")
        lines.append("  " + "-" * 50)
        for ny, stats in sorted(by_train.items()):
            lines.append(
                f"  {ny:>4}Y  {stats['n_folds']:>7}  "
                f"{stats['mean_strategy_return']:>+9.2%}  "
                f"{stats['mean_alpha']:>+11.2%}  "
                f"{stats['pct_positive_alpha']:>5.0%}  "
                f"{stats['mean_strategy_sharpe']:>6.3f}"
            )

    # ── AUC de agentes en el último fold ──────────────────────────────────────
    lines.append(f"\n  AUC DE AGENTES (último fold entrenado)")
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
    lines.append(f"  Resultados guardados en: {backtest_results_dir}/")
    lines.append(sep)

    report_text = "\n".join(lines)
    report_path = Path(backtest_results_dir) / "report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log.info(f"[Evaluator] Informe de texto → {report_path}")
    # También loguear el resumen global para visibilidad inmediata en consola
    for line in lines[:20]:
        log.info(line)
