# =============================================================================
# module/pipeline/trainer.py
# Entrenamiento de agentes + generación de scores OOF para el meta-learner.
# =============================================================================
"""
Responsabilidades:
  - _generate_oof_scores: out-of-fold predictions de cada agente para
    entrenar el meta-learner sin data leakage.
  - train_fold: entrena los 5 agentes + meta-learner en un fold dado,
    genera OOF, y devuelve los agentes entrenados con sus predicciones.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from module.agents.fundamental_agent import FundamentalAgent
from module.agents.valuation_agent import ValuationAgent
from module.agents.momentum_agent import MomentumAgent
from module.agents.bear_agent import BearAgent
from module.agents.sentiment_agent import SentimentAgent
from module.agents.meta_learner import MetaLearner

log = logging.getLogger(__name__)


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
    restante, de modo que el meta-learner nunca ve scores calculados sobre sus
    propios datos de entrenamiento.

    Args:
        X: Features de train normalizadas.
        y: Labels binarios de train (1=Outperform).
        agents_config: Diccionario de configuración por agente con claves:
            'cls' (clase del agente), 'kwargs' (parámetros del constructor),
            'sector_col' (nombre de columna sector o None),
            'invert_y' (True para BearAgent).
        n_splits: Número de folds KFold internos.
        random_seed: Semilla para reproducibilidad.

    Returns:
        Diccionario {nombre_score: Series} con los scores OOF alineados al índice de X.

    Nota sobre KFold con shuffle=True:
        Usamos KFold con mezcla porque el DataFrame de train contiene observaciones
        de múltiples tickers interleaved (no es una serie temporal única). El orden
        temporal queda garantizado por el walk-forward externo: todas las fechas de
        train preceden a las de test. Dentro de train, mezclar mejora la
        representación sectorial en cada fold interno y evita que un fold contenga
        solo tickers de un período concreto.
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


def train_fold(
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fold_id: int,
    agents_results_dir: str,
    random_seed: int = 42,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Entrena los 5 agentes y el meta-learner en un fold de walk-forward.

    Flujo:
        1. Entrena FundamentalAgent, ValuationAgent, MomentumAgent, BearAgent, SentimentAgent.
        2. Genera OOF scores (anti-leakage) para el meta-learner.
        3. Entrena MetaLearner sobre los OOF scores.
        4. Predice en test con todos los agentes.

    Args:
        df_train_norm: DataFrame de train con features normalizadas.
        df_test_norm: DataFrame de test con features normalizadas.
        y_train: Labels binarios de train.
        y_test: Labels binarios de test.
        fold_id: Número de fold (para trazabilidad de ficheros).
        agents_results_dir: Directorio raíz donde cada agente guarda diagnósticos.
        random_seed: Semilla de reproducibilidad.

    Returns:
        Tuple (agents_dict, df_test_with_scores, df_train_with_oof):
          - agents_dict: {'fundamental': ..., 'valuation': ..., ...}
          - df_test_with_scores: df_test_norm enriquecido con scores y label.
          - df_train_with_oof: df_train_norm enriquecido con OOF scores de agentes
            (útil para ablation study y análisis post-hoc).
    """
    # ── 1. Instanciar agentes
    fundamental = FundamentalAgent(results_dir=agents_results_dir, random_seed=random_seed)
    valuation = ValuationAgent(results_dir=agents_results_dir, random_seed=random_seed)
    momentum = MomentumAgent(results_dir=agents_results_dir, random_seed=random_seed)
    bear = BearAgent(results_dir=agents_results_dir, random_seed=random_seed)
    sentiment = SentimentAgent(results_dir=agents_results_dir, random_seed=random_seed)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    # ── 2. Entrenar agentes base
    fundamental.fit(df_train_norm, y_train, fold=fold_id, sector_col="sector")
    valuation.fit(df_train_norm, y_train, fold=fold_id, sector_col="sector")
    momentum.fit(df_train_norm, y_train, fold=fold_id)
    bear.fit(df_train_norm, 1 - y_train, fold=fold_id)  # y invertida: 1=riesgo
    sentiment.fit(df_train_norm, y_train, fold=fold_id)

    # ── 3. Generar OOF scores para el meta-learner (anti-leakage)
    oof_scores = _generate_oof_scores(
        df_train_norm, y_train,
        agents_config={
            "fundamental": {
                "cls": FundamentalAgent,
                "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
                "sector_col": "sector",
            },
            "valuation": {
                "cls": ValuationAgent,
                "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
                "sector_col": "sector",
            },
            "momentum": {
                "cls": MomentumAgent,
                "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
                "sector_col": None,
            },
            "bear": {
                "cls": BearAgent,
                "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
                "sector_col": None,
                "invert_y": True,
            },
            "sentiment": {
                "cls": SentimentAgent,
                "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
                "sector_col": None,
            },
        },
        n_splits=3,
        random_seed=random_seed,
    )
    df_train_with_oof = df_train_norm.copy()
    for col_name, scores_series in oof_scores.items():
        df_train_with_oof[col_name] = scores_series

    # ── 4. Entrenar meta-learner sobre OOF scores
    meta.fit(df_train_with_oof, y_train, fold=fold_id, sector_col="sector")

    # ── 5. Predecir en test
    df_test = df_test_norm.copy()
    df_test["fundamental_score"] = fundamental.predict_score(df_test, "sector").values
    df_test["valuation_score"] = valuation.predict_score(df_test, "sector").values
    df_test["momentum_score"] = momentum.predict_score(df_test).values
    df_test["bear_score"] = bear.predict_score(df_test).values
    df_test["sentiment_score"] = sentiment.predict_score(df_test).values
    df_test["final_score"] = meta.predict_score(df_test, "sector").values
    df_test["label"] = y_test.values

    agents_dict = {
        "fundamental": fundamental,
        "valuation": valuation,
        "momentum": momentum,
        "bear": bear,
        "sentiment": sentiment,
        "meta_learner": meta,
    }
    return agents_dict, df_test, df_train_with_oof


def train_full_history(
    df_norm: pd.DataFrame,
    y: pd.Series,
    agents_results_dir: str,
    random_seed: int = 42,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Entrena todos los agentes sobre el histórico completo (fold live).
    Usa los propios scores de train como entrada al meta-learner
    (no se puede hacer OOF sobre todo el histórico para predicción live).

    Args:
        df_norm: DataFrame completo normalizado.
        y: Labels binarios completos.
        agents_results_dir: Directorio de diagnósticos.
        random_seed: Semilla.

    Returns:
        Tuple (agents_dict, df_norm_with_scores) con scores añadidos al DataFrame.
    """
    fundamental = FundamentalAgent(results_dir=agents_results_dir, random_seed=random_seed)
    valuation = ValuationAgent(results_dir=agents_results_dir, random_seed=random_seed)
    momentum = MomentumAgent(results_dir=agents_results_dir, random_seed=random_seed)
    bear = BearAgent(results_dir=agents_results_dir, random_seed=random_seed)
    sentiment = SentimentAgent(results_dir=agents_results_dir, random_seed=random_seed)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    fundamental.fit(df_norm, y, fold=0, sector_col="sector")
    valuation.fit(df_norm, y, fold=0, sector_col="sector")
    momentum.fit(df_norm, y, fold=0)
    bear.fit(df_norm, 1 - y, fold=0)
    sentiment.fit(df_norm, y, fold=0)

    df_with_scores = df_norm.copy()
    df_with_scores["fundamental_score"] = fundamental.predict_score(df_with_scores, "sector").values
    df_with_scores["valuation_score"] = valuation.predict_score(df_with_scores, "sector").values
    df_with_scores["momentum_score"] = momentum.predict_score(df_with_scores).values
    df_with_scores["bear_score"] = bear.predict_score(df_with_scores).values
    df_with_scores["sentiment_score"] = sentiment.predict_score(df_with_scores).values
    meta.fit(df_with_scores, y, fold=0, sector_col="sector")

    agents_dict = {
        "fundamental": fundamental,
        "valuation": valuation,
        "momentum": momentum,
        "bear": bear,
        "sentiment": sentiment,
        "meta_learner": meta,
    }
    return agents_dict, df_with_scores
