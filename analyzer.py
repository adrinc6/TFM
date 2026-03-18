# =============================================================================
# analyzer.py — Pipeline principal del sistema Multi-Agente ML Stock Picker
# =============================================================================
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from module import backtester

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from environment import *

from module.data_router import DataRouter
from module.feature_engineering import (
    FundamentalFeatureBuilder,
    TechnicalFeatureBuilder,
    ValuationFeatureBuilder,
    InsiderFeatureBuilder,
    SectorNormalizer,
)
from module.agents.fundamental_agent import FundamentalAgent
from module.agents.valuation_agent   import ValuationAgent
from module.agents.momentum_agent    import MomentumAgent
from module.agents.bear_agent        import BearAgent
from module.agents.meta_learner      import MetaLearner
from module.backtester               import WalkForwardBacktester, compute_all_metrics
from module.visualizer               import Visualizer

from module.fetcher import (
    fetch_tickers,
    fetch_prices,
    fetch_fundamentals,
    fetch_insider_trades,
    fetch_analyst_estimates,
    fetch_macro_data,
    unir_csvs_datasets,
    FinancialDataConsolidator,
)

# ── Logging ───────────────────────────────────────────────────────────────────
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
import io as _io
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        # UTF-8 en consola: evita UnicodeEncodeError en Windows (cp1252)
        logging.StreamHandler(_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")),
        logging.FileHandler(f"{RESULTS_DIR}/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# 1. Construcción del dataset maestro
# =============================================================================

def build_master_dataset(
    tickers: List[str],
    router:  DataRouter,
    fundamental_builder:  FundamentalFeatureBuilder,
    technical_builder:    TechnicalFeatureBuilder,
    valuation_builder:    ValuationFeatureBuilder,
    insider_builder:      InsiderFeatureBuilder,
) -> pd.DataFrame:
    """
    Para cada ticker genera observaciones trimestrales con:
      - Features fundamentales (con filing_lag anti-leakage)
      - Features técnicos de precio (hasta la fecha de evaluación)
      - Múltiplos de valoración
      - Señales de insiders
      - Macro context
      - Forward return (label) a FORWARD_RETURN_DAYS dias de trading
      - Sector e Industry de companies.csv
    """
    log.info("Construyendo dataset maestro...")
    macro   = router.load_macro()
    records = []

    for i, ticker in enumerate(tickers, 1):
        try:
            # ── Cargar datos del ticker
            prices      = router.load_prices(ticker)
            consolidated = router.load_consolidated(ticker)  # ya con filing_lag
            insider      = router.load_insider(ticker)
            analyst      = router.load_analyst(ticker)
            info         = router.get_ticker_info(ticker)

            if prices is None or consolidated is None:
                log.debug(f"[{ticker}] Sin precios o consolidado — skip")
                continue
            if len(consolidated) < MIN_HISTORY_QUARTERS:
                log.debug(f"[{ticker}] Insuficientes trimestres — skip")
                continue

            # ── Enriquecer fundamentales con features adicionales
            fund_enriched = fundamental_builder.build(consolidated)

            # ── Iterar sobre fechas trimestrales (evaluation dates)
            eval_dates = consolidated.index.tolist()

            for as_of in eval_dates:
                as_of = pd.Timestamp(as_of)

                # Snapshot fundamental (sin look-ahead garantizado por filing_lag)
                fund_snap = router.get_fundamental_snapshot(fund_enriched, as_of)
                if fund_snap is None:
                    continue

                # Ventana de precio hasta as_of
                price_window = router.get_price_window(prices, as_of, lookback_days=400)
                if len(price_window) < 20:
                    continue

                # Features técnicos
                tech_feats = technical_builder.build(price_window, as_of)

                # Features de valoración (precio × fundamentales)
                val_feats = valuation_builder.build(
                    prices_df=prices,
                    fund_snapshot=fund_snap,
                    hist_fund=fund_enriched[fund_enriched.index <= as_of],
                    as_of=as_of,
                    analyst_df=analyst,
                )

                # Features de insiders (90 días)
                insider_window = (router.get_insider_window(insider, as_of, 90)
                                  if insider is not None else None)
                insider_feats  = insider_builder.build(insider_window)

                # Macro snapshot
                macro_snap = router.get_macro_snapshot(macro, as_of) if macro is not None else None

                # Label: forward return a 252 días (¡solo para entrenar!)
                fwd_return = router.compute_forward_return(prices, as_of, FORWARD_RETURN_DAYS)
                if fwd_return is None:
                    continue

                # ── Combinar en un dict
                record = {
                    "ticker":  ticker,
                    "date":    as_of,
                    "sector":  info["sector"],
                    "industry": info["industry"],
                    "forward_return": fwd_return,
                }
                record.update(fund_snap.to_dict())
                record.update(tech_feats.to_dict())
                record.update(val_feats.to_dict())
                record.update(insider_feats.to_dict())
                if macro_snap is not None:
                    record.update(macro_snap.to_dict())

                records.append(record)

        except Exception as e:
            log.warning(f"[{ticker}] Error construyendo features: {e}")
            continue

        if i % 50 == 0:
            log.info(f"  Procesados {i}/{len(tickers)} tickers — {len(records)} observaciones")

    if not records:
        raise RuntimeError("Dataset maestro vacío. Revisa los datos en data/")

    df = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    log.info(f"Dataset maestro: {len(df)} observaciones — {df.index.get_level_values('ticker').nunique()} tickers")
    return df


# =============================================================================
# 2. Normalización sectorial del dataset
# =============================================================================

def apply_sector_normalization(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    normalizer: SectorNormalizer,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Aplica SectorNormalizer al dataset maestro.
    fit=True solo en entrenamiento para evitar leakage de estadísticas.
    Usa TODAS las observaciones de train (no solo la ultima por ticker)
    para obtener estadisticas sectoriales mas robustas.
    """
    if fit:
        # Usar TODAS las filas del train para estadisticas sectoriales
        features_dict: Dict[str, pd.Series] = {}
        for i, ((ticker, date), row) in enumerate(df.iterrows()):
            features_dict[f"{ticker}_{i}"] = row
        # Construir sector_map expandido para todas las observaciones
        expanded_sector_map = {
            f"{ticker}_{i}": sector_map.get(ticker, "Unknown")
            for i, (ticker, date) in enumerate(df.index)
        }
        normalizer.fit(features_dict, expanded_sector_map)

    # Transformar vectorizado: agrupar por sector y aplicar Z-score por columna
    tickers_col = df.index.get_level_values("ticker")
    sectors = tickers_col.map(lambda t: sector_map.get(t, "Unknown"))

    df_norm = df.copy()
    for col in normalizer.COLS:
        if col not in df_norm.columns:
            df_norm[f"{col}_zsector"] = np.nan
            continue
        zcol = f"{col}_zsector"
        df_norm[zcol] = np.nan
        for sector in sectors.unique():
            stats = normalizer._stats.get(sector, {}).get(col)
            if stats is None:
                continue
            mean, std = stats
            mask = sectors == sector
            if std > 0:
                df_norm.loc[mask, zcol] = (df_norm.loc[mask, col] - mean) / std
            else:
                df_norm.loc[mask, zcol] = 0.0

    return df_norm


# =============================================================================
# 3. Walk-Forward Pipeline
# =============================================================================


# =============================================================================
# Helper: Explicaciones SHAP por ticker
# =============================================================================

def _explain_top_tickers(
    agents:   Dict,
    df_test:  pd.DataFrame,
    scores:   pd.Series,
    fold_id:  int,
    top_n:    int = 10,
):
    """
    Genera explicaciones SHAP para los top_n tickers con mayor score
    y los top_n con menor score de cada fold.
    Guarda un JSON consolidado con el 'por que' de cada prediccion.
    """
    if scores.empty:
        return

    tickers_col = df_test.index.get_level_values("ticker")
    dates_col   = df_test.index.get_level_values("date")

    # Construir Series con el ultimo score por ticker (puede haber varios por fecha)
    ticker_scores = pd.Series(scores.values, index=tickers_col).groupby(level=0).last()

    top_bulls  = ticker_scores.nlargest(top_n).index.tolist()
    top_bears  = ticker_scores.nsmallest(top_n).index.tolist()
    selected   = list(dict.fromkeys(top_bulls + top_bears))

    results_path = Path(AGENTS_RESULTS_DIR) / f"fold_{fold_id}_ticker_explanations.json"
    all_explanations = {"fold": fold_id, "tickers": {}}

    for ticker in selected:
        mask = tickers_col == ticker
        if not mask.any():
            continue
        # Ultima observacion del ticker en el test set
        row   = df_test.loc[mask].iloc[-1]
        score = float(ticker_scores.get(ticker, 0.5))

        ticker_exp = {
            "score":     round(score, 4),
            "label":     "Outperform" if score >= 0.5 else "Underperform",
            "agents":    {},
        }

        for ag_name, ag in agents.items():
            if ag._explainer is None:
                continue
            try:
                exp = ag._explainer.explain_prediction(row, ticker, score, top_n=6, fold=fold_id)
                ticker_exp["agents"][ag_name] = {
                    "text":        exp.get("text", ""),
                    "top_drivers": exp.get("top_drivers", [])[:6],
                }
            except Exception as ex:
                log.debug(f"Explain {ag_name}/{ticker}: {ex}")

        all_explanations["tickers"][ticker] = ticker_exp
        log.info(f"  [{'Outperform' if score >= 0.5 else 'Underperform':11s}] "
                 f"{ticker:<6} score={score:.3f}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"[Explainer] Fold {fold_id}: explicaciones de {len(selected)} tickers -> {results_path.name}")

def _generate_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    agents_config: Dict,
    n_splits: int = 3,
    random_seed: int = 42,
) -> Dict[str, pd.Series]:
    """
    Genera out-of-fold predictions de cada agente para entrenar el meta-learner
    sin data leakage. Cada agente se entrena en K-1 folds y predice en el fold
    restante, asegurando que el meta-learner nunca ve scores de entrenamiento.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof: Dict[str, pd.Series] = {}

    for ag_name, cfg in agents_config.items():
        score_col = f"{ag_name}_score"
        oof_vals = pd.Series(np.nan, index=X.index, name=score_col)

        for fold_tr, fold_val in kf.split(X):
            X_tr = X.iloc[fold_tr]
            X_val = X.iloc[fold_val]
            y_tr = y.iloc[fold_tr]
            y_val_target = y.iloc[fold_val]

            agent = cfg["cls"](**cfg["kwargs"])
            y_fit = (1 - y_tr) if cfg.get("invert_y") else y_tr

            if cfg.get("sector_col"):
                agent.fit(X_tr, y_fit, fold=0, sector_col=cfg["sector_col"])
                preds = agent.predict_score(X_val, cfg["sector_col"])
            else:
                agent.fit(X_tr, y_fit, fold=0)
                preds = agent.predict_score(X_val)

            oof_vals.iloc[fold_val] = preds.values

        # Rellenar NaN residuales con 0.5 (neutro)
        oof[score_col] = oof_vals.fillna(0.5)

    return oof


def run_walkforward_pipeline(
    df:         pd.DataFrame,
    sector_map: Dict[str, str],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark:  pd.Series,
    results_dir: str,
) -> Dict:
    """
    Ejecuta el pipeline completo de walk-forward:
      1. Genera folds temporales
      2. En cada fold entrena los 4 agentes + meta-learner
      3. Evalúa en test y simula cartera
      4. Devuelve métricas globales
    """
    backtester = WalkForwardBacktester(
        train_years=WALKFORWARD_TRAIN_YEARS,
        test_quarters=WALKFORWARD_TEST_QUARTERS,
        risk_free=RISK_FREE_RATE,
        results_dir=BACKTEST_RESULTS_DIR,

    )
    visualizer = Visualizer(plots_dir=PLOTS_DIR, no_plot=NO_PLOT)
    normalizer = SectorNormalizer()

    folds = backtester.generate_folds(START_DATE, END_DATE)
    agent_diag_history: Dict[str, List] = {
        "fundamental": [], "valuation": [], "momentum": [], "bear": [], "meta_learner": []
    }

    dates = df.index.get_level_values("date")

    for fold_id, (train_start, train_end, test_end, _train_years) in enumerate(folds, 1):
        log.info(f"\n{'='*60}")
        log.info(f"  FOLD {fold_id} | Train: {train_start.date()} -> {train_end.date()} "
                 f"| Test: {train_end.date()} -> {test_end.date()}")
        log.info(f"{'='*60}")

        # ── Separar train / test (sin leakage: usamos fecha de evaluación)
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask  = (dates >= train_end)   & (dates < test_end)

        df_train = df.loc[train_mask]
        df_test  = df.loc[test_mask]

        if len(df_train) < 100:
            log.warning(f"Fold {fold_id}: Train insuficiente ({len(df_train)} obs). Skip.")
            continue

        # ── Eliminar duplicados de índice en origen (mismo ticker+date → conservar último)
        df_train = df_train[~df_train.index.duplicated(keep="last")]
        df_test  = df_test[~df_test.index.duplicated(keep="last")]

        # ── Label binario (1 = Outperform vs mediana del período)
        median_ret = df_train["forward_return"].median()

        # ── Normalización sectorial (fit solo en train)
        df_train_norm = apply_sector_normalization(df_train, sector_map, normalizer, fit=True)
        df_test_norm  = apply_sector_normalization(df_test,  sector_map, normalizer, fit=False)

        # ── Eliminar duplicados también en normalizados (por si la normalización introduce alguno)
        df_train_norm = df_train_norm[~df_train_norm.index.duplicated(keep="last")]
        df_test_norm  = df_test_norm[~df_test_norm.index.duplicated(keep="last")]

        # ── Alinear labels al índice REAL de los DataFrames normalizados.
        forward_train = df_train["forward_return"].reindex(df_train_norm.index)
        forward_test  = df_test["forward_return"].reindex(df_test_norm.index)
        y_train = (forward_train > median_ret).astype(int).dropna()
        y_test  = (forward_test  > median_ret).astype(int).dropna()
        df_train_norm = df_train_norm.loc[y_train.index]
        df_test_norm  = df_test_norm.loc[y_test.index]

        # ── Instanciar agentes
        fundamental = FundamentalAgent(results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
        valuation   = ValuationAgent(  results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
        momentum    = MomentumAgent(   results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
        bear        = BearAgent(       results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
        meta        = MetaLearner(     results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)

        try:
            # ── Entrenar agentes (cada uno ve solo sus columnas)
            fundamental.fit(df_train_norm, y_train, fold=fold_id, sector_col="sector")
            valuation.fit(  df_train_norm, y_train, fold=fold_id, sector_col="sector")
            momentum.fit(   df_train_norm, y_train, fold=fold_id)
            bear.fit(       df_train_norm, 1 - y_train, fold=fold_id)  # y invertida: 1=riesgo

            # ── Generar scores OUT-OF-FOLD para el meta-learner (anti-leakage)
            # Usar CV interno para que el meta-learner no vea scores inflados
            oof_scores = _generate_oof_scores(
                df_train_norm, y_train,
                agents_config={
                    "fundamental": {"cls": FundamentalAgent, "kwargs": {"results_dir": AGENTS_RESULTS_DIR, "random_seed": RANDOM_SEED}, "sector_col": "sector"},
                    "valuation":   {"cls": ValuationAgent,   "kwargs": {"results_dir": AGENTS_RESULTS_DIR, "random_seed": RANDOM_SEED}, "sector_col": "sector"},
                    "momentum":    {"cls": MomentumAgent,    "kwargs": {"results_dir": AGENTS_RESULTS_DIR, "random_seed": RANDOM_SEED}, "sector_col": None},
                    "bear":        {"cls": BearAgent,        "kwargs": {"results_dir": AGENTS_RESULTS_DIR, "random_seed": RANDOM_SEED}, "sector_col": None, "invert_y": True},
                },
                n_splits=3,
                random_seed=RANDOM_SEED,
            )
            for col_name, scores_series in oof_scores.items():
                df_train_norm[col_name] = scores_series

            meta.fit(df_train_norm, y_train, fold=fold_id, sector_col="sector")

            # ── Predecir en test
            df_test_norm["fundamental_score"] = fundamental.predict_score(df_test_norm, "sector").values
            df_test_norm["valuation_score"]   = valuation.predict_score(  df_test_norm, "sector").values
            df_test_norm["momentum_score"]    = momentum.predict_score(   df_test_norm).values
            df_test_norm["bear_score"]        = bear.predict_score(       df_test_norm).values
            df_test_norm["final_score"]       = meta.predict_score(       df_test_norm, "sector").values
            df_test_norm["label"]             = y_test.values

            # ── Evaluar meta-learner
            eval_metrics = meta.evaluate(df_test_norm, y_test, fold=fold_id)

            # ── Plots de distribución de scores por fold
            visualizer.plot_score_distribution(df_test_norm, fold=fold_id)

            # ── Guardar predicciones para backtesting
            preds_df = df_test_norm[["final_score","label"]].copy()
            preds_df["ticker"] = preds_df.index.get_level_values("ticker")
            preds_df["date"]   = preds_df.index.get_level_values("date")
            preds_df = preds_df.reset_index(drop=True)

            # ── Simular cartera
            fold_result = backtester.simulate_portfolio(
                predictions_df=preds_df.rename(columns={"final_score":"score"}),
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

            # ── Acumular diagnósticos de agentes
            for ag_name, ag in [("fundamental", fundamental), ("valuation", valuation),
                                  ("momentum", momentum), ("bear", bear), ("meta_learner", meta)]:
                agent_diag_history[ag_name].append(ag._diagnostics.copy())

            # ── Feature importances por agente
            for ag_name, ag in [("fundamental", fundamental), ("valuation", valuation),
                                  ("momentum", momentum), ("bear", bear)]:
                if hasattr(ag, '_feature_cols') and ag.is_trained:
                    try:
                        imp_path = Path(AGENTS_RESULTS_DIR) / ag_name / f"feature_importances_fold{fold_id}.csv"
                        if imp_path.exists():
                            imp = pd.read_csv(imp_path, index_col=0)["importance"]
                            visualizer.plot_feature_importances(imp, ag_name, fold=fold_id)
                    except Exception:
                        pass

            # -- Explicaciones SHAP: top 10 mejores y peores predicciones del fold
            _explain_top_tickers(
                agents={"fundamental": fundamental, "valuation": valuation,
                        "momentum": momentum, "bear": bear, "meta_learner": meta},
                df_test=df_test_norm,
                scores=df_test_norm["final_score"],
                fold_id=fold_id,
            )

        except Exception as e:
            log.error(f"Fold {fold_id} falló: {e}", exc_info=True)
            continue

    # ── Resumen global
    summary = backtester.summarize()
    backtester.save_folds_summary(plots_dir="results/plots")
    
    # ── Guardar diagnósticos históricos de agentes
    diag_path = Path(AGENTS_RESULTS_DIR) / "all_folds_diagnostics.json"
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

    return summary


# =============================================================================
# 4. Descarga y preparación de datos (fetcher)
# =============================================================================

def download_data() -> List[str]:
    """
    Descarga todos los datos necesarios usando el fetcher.
    Devuelve la lista de tickers del S&P 500 que se usará en todo el pipeline.
    Si FORCE_DOWNLOAD=False y los datos ya existen, omite la descarga.
    """
    log.info("PASO 1 — DESCARGANDO DATOS")

    tickers = fetch_tickers()
    log.info(f"  Tickers S&P 500: {len(tickers)}")

    if FORCE_DOWNLOAD:
        fetch_prices(tickers, START_DATE, END_DATE)
        fetch_fundamentals(tickers)
        fetch_insider_trades(tickers)
        fetch_analyst_estimates(tickers)
        fetch_macro_data()
        unir_csvs_datasets()
    else:
        # Descarga solo los tickers sin datos o desactualizados (DAYS_UPDATE)
        fetch_prices(tickers, START_DATE, END_DATE)
        fetch_macro_data()
        log.info("  (FORCE_DOWNLOAD=False — solo actualiza datos obsoletos)")

    return tickers


def prepare_data(tickers: List[str]):
    """
    Consolida los datos crudos en el formato que consume el pipeline ML.
    Solo procesa los tickers pasados como argumento.
    """
    log.info("PASO 2 — PREPARANDO DATOS")
    consolidator = FinancialDataConsolidator()
    consolidator.load_source_data()
    consolidator.process_all_tickers(tickers)
    log.info("Consolidacion completada")


# =============================================================================
# 5. Fold live Q1 2026 — predicción + evaluación con precios de hoy
# =============================================================================

def run_live_q1_2026_fold(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    router: DataRouter,
    fundamental_builder: FundamentalFeatureBuilder,
    technical_builder: TechnicalFeatureBuilder,
    valuation_builder: ValuationFeatureBuilder,
    insider_builder: InsiderFeatureBuilder,
    tickers_ok: List[str],
    top_n: int = 10,
) -> None:
    """
    Fold live fuera de muestra para Q1 2026:
      1. Construye features a fecha END_DATE (2026-01-01) usando datos guardados.
      2. Entrena agentes sobre todo el histórico disponible en df.
      3. Genera predicciones (scores) para cada ticker.
      4. Descarga en memoria (sin guardar) los precios de hoy para calcular
         los retornos reales del Q1 2026 y estimar el alpha de la estrategia.
    """
    as_of     = pd.Timestamp(END_DATE)          # inicio Q1 2026: 2026-01-01
    today     = pd.Timestamp.today().normalize() # ≈ 2026-03-06
    today_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("\n" + "=" * 60)
    log.info("  FOLD LIVE — Q1 2026 (out-of-sample)")
    log.info(f"  Train hasta: {as_of.date()}  |  Eval: {as_of.date()} → {today.date()}")
    log.info("=" * 60)

    # ── 1. Construir features a fecha as_of usando datos ya en disco
    log.info("Construyendo features de test (as_of=2026-01-01) desde datos guardados...")
    macro   = router.load_macro()
    records = []
    for ticker in tickers_ok:
        try:
            prices      = router.load_prices(ticker)
            consolidated = router.load_consolidated(ticker)
            insider      = router.load_insider(ticker)
            analyst      = router.load_analyst(ticker)
            info         = router.get_ticker_info(ticker)

            if prices is None or consolidated is None:
                continue
            if len(consolidated) < MIN_HISTORY_QUARTERS:
                continue

            fund_enriched = fundamental_builder.build(consolidated)
            fund_snap     = router.get_fundamental_snapshot(fund_enriched, as_of)
            if fund_snap is None:
                continue

            price_window = router.get_price_window(prices, as_of, lookback_days=400)
            if len(price_window) < 20:
                continue

            tech_feats   = technical_builder.build(price_window, as_of)
            val_feats    = valuation_builder.build(
                prices_df=prices,
                fund_snapshot=fund_snap,
                hist_fund=fund_enriched[fund_enriched.index <= as_of],
                as_of=as_of,
                analyst_df=analyst,
            )
            insider_window = (router.get_insider_window(insider, as_of, 90)
                              if insider is not None else None)
            insider_feats  = insider_builder.build(insider_window)
            macro_snap     = router.get_macro_snapshot(macro, as_of) if macro is not None else None

            record = {
                "ticker":   ticker,
                "date":     as_of,
                "sector":   info["sector"],
                "industry": info["industry"],
            }
            record.update(fund_snap.to_dict())
            record.update(tech_feats.to_dict())
            record.update(val_feats.to_dict())
            record.update(insider_feats.to_dict())
            if macro_snap is not None:
                record.update(macro_snap.to_dict())
            records.append(record)

        except Exception as e:
            log.debug(f"[{ticker}] Error features Q1-live: {e}")

    if not records:
        log.error("No se generaron features para el fold live Q1 2026 — abortando.")
        return

    df_q1 = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    log.info(f"Features de test Q1 2026: {len(df_q1)} tickers")

    # ── 2. Entrenar agentes sobre todo el histórico
    log.info("Entrenando agentes finales sobre dataset histórico completo...")
    normalizer    = SectorNormalizer()
    df_train      = df[~df.index.duplicated(keep="last")].copy()
    df_train_norm = apply_sector_normalization(df_train, sector_map, normalizer, fit=True)
    df_train_norm = df_train_norm[~df_train_norm.index.duplicated(keep="last")]

    median_ret    = df_train["forward_return"].median()
    forward_train = df_train["forward_return"].reindex(df_train_norm.index)
    y_train       = (forward_train > median_ret).astype(int).dropna()
    df_train_norm = df_train_norm.loc[y_train.index]

    fundamental = FundamentalAgent(results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
    valuation   = ValuationAgent(  results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
    momentum    = MomentumAgent(   results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
    bear        = BearAgent(       results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)
    meta        = MetaLearner(     results_dir=AGENTS_RESULTS_DIR, random_seed=RANDOM_SEED)

    fundamental.fit(df_train_norm, y_train, fold=0, sector_col="sector")
    valuation.fit(  df_train_norm, y_train, fold=0, sector_col="sector")
    momentum.fit(   df_train_norm, y_train, fold=0)
    bear.fit(       df_train_norm, 1 - y_train, fold=0)

    df_train_norm["fundamental_score"] = fundamental.predict_score(df_train_norm, "sector").values
    df_train_norm["valuation_score"]   = valuation.predict_score(  df_train_norm, "sector").values
    df_train_norm["momentum_score"]    = momentum.predict_score(   df_train_norm).values
    df_train_norm["bear_score"]        = bear.predict_score(       df_train_norm).values
    meta.fit(df_train_norm, y_train, fold=0, sector_col="sector")

    # ── 3. Predecir en Q1 2026
    df_q1_norm = apply_sector_normalization(df_q1, sector_map, normalizer, fit=False)
    df_q1_norm = df_q1_norm[~df_q1_norm.index.duplicated(keep="last")]

    df_q1_norm["fundamental_score"] = fundamental.predict_score(df_q1_norm, "sector").values
    df_q1_norm["valuation_score"]   = valuation.predict_score(  df_q1_norm, "sector").values
    df_q1_norm["momentum_score"]    = momentum.predict_score(   df_q1_norm).values
    df_q1_norm["bear_score"]        = bear.predict_score(       df_q1_norm).values
    df_q1_norm["final_score"]       = meta.predict_score(       df_q1_norm, "sector").values

    ticker_scores = (
        df_q1_norm["final_score"]
        .reset_index(level="date", drop=True)
        .sort_values(ascending=False)
    )

    top_bulls  = list(ticker_scores.head(top_n).index)
    top_bears  = list(ticker_scores.tail(top_n).index)

    log.info(f"\n{'='*60}")
    log.info(f"  TOP {top_n} predichos OUTPERFORM — Q1 2026")
    log.info(f"{'='*60}")
    for tk in top_bulls:
        log.info(f"  {tk:<8} score={ticker_scores[tk]:.3f}  [{sector_map.get(tk,'?')}]")

    log.info(f"\n  BOTTOM {top_n} predichos UNDERPERFORM — Q1 2026")
    log.info(f"{'='*60}")
    for tk in top_bears:
        log.info(f"  {tk:<8} score={ticker_scores[tk]:.3f}  [{sector_map.get(tk,'?')}]")

    # ── 4. Descargar precios de hoy en memoria (sin guardar en disco)
    # Solo los tickers del top/bottom + SPY para benchmark
    tickers_to_fetch = list(dict.fromkeys(top_bulls + top_bears + ["SPY"]))
    log.info(f"\nDescargando precios actuales ({as_of.date()} → {today.date()}) en memoria "
             f"para {len(tickers_to_fetch)} tickers (sin guardar)...")

    live_prices: Dict[str, pd.Series] = {}
    for ticker in tickers_to_fetch:
        try:
            df_live = yf.download(
                ticker,
                start=as_of.strftime("%Y-%m-%d"),
                end=today_str,
                auto_adjust=True,
                progress=False,
            )
            if df_live.empty or len(df_live) < 2:
                continue
            if isinstance(df_live.columns, pd.MultiIndex):
                df_live.columns = df_live.columns.droplevel(1)
            live_prices[ticker] = df_live["Close"]
        except Exception as e:
            log.debug(f"[{ticker}] Error precio live: {e}")

    # ── 5. Calcular retornos reales Q1 2026 y alpha
    def qtd_return(close_series: pd.Series) -> Optional[float]:
        """Retorno desde el primer día disponible hasta el último."""
        s = close_series.dropna()
        if len(s) < 2:
            return None
        return float((s.iloc[-1] - s.iloc[0]) / s.iloc[0])

    bull_returns = {t: r for t in top_bulls
                    if (r := qtd_return(live_prices.get(t, pd.Series()))) is not None}
    bear_returns = {t: r for t in top_bears
                    if (r := qtd_return(live_prices.get(t, pd.Series()))) is not None}
    benchmark_return = qtd_return(live_prices.get("SPY", pd.Series()))

    portfolio_return = float(np.mean(list(bull_returns.values()))) if bull_returns else None
    alpha = (portfolio_return - benchmark_return
             if portfolio_return is not None and benchmark_return is not None else None)

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTADO REAL Q1 2026 ({as_of.date()} → {today.date()})")
    log.info(f"{'='*60}")
    if benchmark_return is not None:
        log.info(f"  Benchmark (SPY):        {benchmark_return:+.2%}")
    if portfolio_return is not None:
        log.info(f"  Portfolio top-{top_n} (long): {portfolio_return:+.2%}")
    if alpha is not None:
        log.info(f"  Alpha Q1 2026:          {alpha:+.2%}")

    log.info(f"\n  Retornos individuales top-{top_n}:")
    for tk in top_bulls:
        ret = bull_returns.get(tk)
        bm  = benchmark_return if benchmark_return else 0.0
        alp = (ret - bm) if ret is not None else None
        log.info(f"    {tk:<8} {(ret or 0):+.2%}  (alpha {(alp or 0):+.2%})  [{sector_map.get(tk,'?')}]")

    # ── 6. Guardar resultados (solo scores y retornos, no precios)
    tickers_idx = df_q1_norm.index.get_level_values("ticker")
    all_preds = []
    for tk in ticker_scores.index:
        mask = tickers_idx == tk
        if not mask.any():
            continue
        row = df_q1_norm.loc[mask].iloc[0]
        actual_ret = bull_returns.get(tk) or bear_returns.get(tk)
        all_preds.append({
            "ticker":            tk,
            "sector":            sector_map.get(tk, "?"),
            "final_score":       round(float(row["final_score"]), 4),
            "fundamental_score": round(float(row["fundamental_score"]), 4),
            "valuation_score":   round(float(row["valuation_score"]), 4),
            "momentum_score":    round(float(row["momentum_score"]), 4),
            "bear_score":        round(float(row["bear_score"]), 4),
            "label":             "Outperform" if row["final_score"] >= 0.5 else "Underperform",
            "actual_return_q1":  round(actual_ret, 4) if actual_ret is not None else None,
        })

    pred_df = pd.DataFrame(all_preds).sort_values("final_score", ascending=False)
    csv_path  = Path(RESULTS_DIR) / "predictions_q1_2026.csv"
    json_path = Path(RESULTS_DIR) / "predictions_q1_2026.json"
    pred_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_end":        as_of.strftime("%Y-%m-%d"),
            "eval_date":        today.strftime("%Y-%m-%d"),
            "quarter":          "Q1 2026",
            "n_tickers":        len(all_preds),
            "portfolio_return": round(portfolio_return, 4) if portfolio_return is not None else None,
            "benchmark_return": round(benchmark_return, 4) if benchmark_return is not None else None,
            "alpha":            round(alpha, 4) if alpha is not None else None,
            "top_picks":        [p for p in all_preds if p["ticker"] in top_bulls],
            "bottom_picks":     [p for p in all_preds if p["ticker"] in top_bears],
            "all":              all_preds,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\nResultados Q1 2026 guardados en:")
    log.info(f"  {csv_path}")
    log.info(f"  {json_path}")


# =============================================================================
# 6. Main
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("  INICIANDO PIPELINE ML MULTI-AGENTE STOCK PICKER")
    log.info("=" * 60)

    # # ── 1. Descargar datos
    tickers = download_data()

    # # ── 2. Consolidar datos
    # prepare_data(tickers)

    # ── 3. DataRouter (usa los mismos tickers del fetcher, no escanea carpetas)
    router     = DataRouter(data_dir=DATA_DIR, filing_lag_days=FILING_LAG_DAYS)
    sector_map = router.get_sector_map()

    # Filtrar a los tickers que realmente tienen datos disponibles tras la descarga
    prices_dir       = Path(DATA_DIR) / "prices"
    consolidated_dir = Path(DATA_DIR) / "consolidated"
    tickers_ok = sorted(
        t for t in tickers
        if (prices_dir / f"{t}.csv").exists()
        and (consolidated_dir / f"{t}.csv").exists()
    )
    log.info(f"Tickers con datos completos: {len(tickers_ok)} / {len(tickers)}")

    # ── 4. Builders de features
    fundamental_builder = FundamentalFeatureBuilder()
    technical_builder   = TechnicalFeatureBuilder()
    valuation_builder   = ValuationFeatureBuilder()
    insider_builder     = InsiderFeatureBuilder()

    # ── 5. Construir dataset maestro solo con los tickers del S&P 500
    df = build_master_dataset(
        tickers=tickers_ok,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
    )

    df.to_csv(f"{RESULTS_DIR}/master_dataset.csv")
    log.info(f"Dataset maestro: {len(df)} observaciones — {len(tickers_ok)} tickers")

    # ── 6. Diagnostico sectorial
    visualizer = Visualizer(plots_dir=PLOTS_DIR, no_plot=NO_PLOT)
    visualizer.plot_sector_performance(df.reset_index())

    summary = {}
    if not SKIP_BACKTEST:
        # ── 7. Precios y benchmark para walk-forward
        prices_dict: Dict[str, pd.DataFrame] = {}
        for ticker in tickers_ok:
            p = router.load_prices(ticker)
            if p is not None:
                prices_dict[ticker] = p

        benchmark = router.load_sp500_prices()
        if benchmark is None:
            log.warning("Sin S&P 500 — usando retorno cero como benchmark")
            benchmark = pd.Series(0.0, index=pd.date_range(START_DATE, END_DATE))
        benchmark_returns = benchmark.pct_change().dropna()

        # ── 8. Walk-Forward Pipeline (backtest historico)
        summary = run_walkforward_pipeline(
            df=df,
            sector_map=sector_map,
            prices_dict=prices_dict,
            benchmark=benchmark_returns,
            results_dir=RESULTS_DIR,
        )
    else:
        log.info("SKIP_BACKTEST=True — saltando walk-forward backtest historico")

    # ── 10. Fold live Q1 2026
    run_live_q1_2026_fold(
        df=df,
        sector_map=sector_map,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
        tickers_ok=tickers_ok,
    )

    # ── Resultado final
    log.info("\n" + "=" * 60)
    log.info("  PIPELINE COMPLETADO")
    log.info("=" * 60)
    log.info(f"  Tickers analizados:   {len(tickers_ok)}")
    if summary:
        log.info(f"  Alpha medio:          {summary.get('mean_alpha', 0):.2%}")
        log.info(f"  Folds con alpha > 0:  {summary.get('pct_folds_positive_alpha', 0):.0%}")
        log.info(f"  Sharpe Estrategia:    {summary.get('global_strategy_sharpe', 0):.3f}")
        log.info(f"  Sharpe Benchmark:     {summary.get('global_benchmark_sharpe', 0):.3f}")
        log.info(f"  Max DD Estrategia:    {summary.get('global_strategy_max_drawdown', 0):.2%}")
    log.info(f"  Resultados en:        {RESULTS_DIR}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
