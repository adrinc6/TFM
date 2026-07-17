"""Features point-in-time, etiquetas futuras separadas y baselines."""

from __future__ import annotations

import logging

import pandas as pd

from environment import Settings
from module.baselines import build_baseline_scores
from module.dataset import snapshot_dates
from module.utils import read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)

FACTOR_SOURCES = {
    "roe": ("factor_roe", True, False),
    "roic": ("factor_roic", True, False),
    "net_margin": ("factor_net_margin", True, False),
    "operating_margin": ("factor_operating_margin", True, False),
    "gross_margin": ("factor_gross_margin", True, False),
    "fcf_margin": ("factor_fcf_margin", True, False),
    "debt_equity": ("factor_debt_equity", False, False),
    "current_ratio": ("factor_current_ratio", True, False),
    "eps_growth_yoy": ("factor_eps_growth_yoy", True, False),
    "sales_per_share_growth_yoy": ("factor_sales_per_share_growth_yoy", True, False),
    "pe": ("factor_pe", False, True),
    "pb": ("factor_pb", False, True),
    "ps": ("factor_ps", False, True),
    "ev_ebitda": ("factor_ev_ebitda", False, True),
}


def build_features(settings: Settings) -> pd.DataFrame:
    """Construye factores observables, baselines y targets en artefactos distintos."""
    output_dir = settings.processed_output_dir
    panel = read_parquet(output_dir / "panel_point_in_time.parquet", "RUN_MODE='dataset'")
    benchmark = read_parquet(output_dir / "benchmark_point_in_time.parquet", "RUN_MODE='dataset'")
    asset_prices = read_parquet(output_dir / "asset_price_point_in_time.parquet", "RUN_MODE='dataset'")
    if benchmark.empty:
        raise RuntimeError(
            f"El benchmark {settings.benchmark_ticker} no tiene precios PIT. "
            "Ejecuta RUN_MODE='download' y después RUN_MODE='dataset' con el mismo alcance."
        )

    sector_by_ticker = _load_sector_map(settings) if settings.neutralize_by_sector else {}
    features = _build_feature_frame(panel, benchmark, settings, sector_by_ticker)
    targets = _build_targets(features, asset_prices, benchmark, settings)
    baselines = build_baseline_scores(features)

    write_parquet(features, output_dir / "features_point_in_time.parquet")
    write_parquet(targets, output_dir / "targets_forward_3m.parquet")
    write_parquet(baselines, output_dir / "baseline_scores.parquet")
    write_json(_coverage(features, baselines, settings), output_dir / "features_coverage.json")
    log.info(
        "Features: rows=%s fresh_rows=%s targets=%s output=%s",
        len(features),
        int(features["is_price_fresh"].sum()),
        int(targets["target_available"].sum()),
        output_dir,
    )
    return features


# Fundamentales cuya TENDENCIA (cambio respecto a la publicacion anterior) se anade como
# feature. "mejora" = el fundamental sube respecto a su valor anterior de la misma empresa.
MOMENTUM_SOURCES = ("roe", "roic", "net_margin", "operating_margin", "eps_growth_yoy")

# Factores B3 que se suman a los agentes cuando fundamental_momentum esta activo.
MOMENTUM_FACTORS_QUALITY = tuple(f"factor_mom_{name}" for name in MOMENTUM_SOURCES)
MOMENTUM_FACTORS_VALUE = ("factor_pe_fundamental_component", "factor_pe_price_component")


def _add_fundamental_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    """Anade features de tendencia de fundamentales y descomposicion del cambio de P/E.

    Point-in-time por construccion: para cada empresa se compara el fundamental del periodo
    actual con el de su PUBLICACION anterior (detectada por cambio de `fundamental_period`),
    no con el trimestre de calendario. Entre publicaciones el valor se mantiene constante, asi
    que el delta solo cambia cuando de verdad hay informacion nueva — nunca usa el futuro.

    Descomposicion del P/E: como P/E = precio / EPS, el cambio de la valoracion se separa en
    su componente-precio (Dlog precio) y su componente-fundamental (-Dlog EPS). "Barato porque
    la empresa mejora" (componente fundamental positivo) y "barato porque el precio se hunde"
    (componente precio negativo) pasan a ser dos senales distintas.
    """
    import numpy as np

    frame = frame.sort_values(["ticker", "snapshot_date"]).copy()

    # EPS implicito de las columnas ya PIT: EPS = precio / (P/E). Solo donde ambos existen.
    pe = pd.to_numeric(frame["pe"], errors="coerce")
    price = pd.to_numeric(frame["price"], errors="coerce")
    frame["_eps_implied"] = (price / pe).where(pe.gt(0) & price.gt(0))

    # Delta del fundamental respecto a la publicacion anterior de la MISMA empresa.
    # Nos quedamos con una fila por (ticker, fundamental_period) para no repetir el delta.
    is_new_publication = (
        frame["fundamental_period"] != frame.groupby("ticker")["fundamental_period"].shift()
    )

    def _delta_vs_previous_publication(values: pd.Series) -> pd.Series:
        """Cambio del valor entre la publicacion actual y la anterior, constante entre
        publicaciones.

        Entre publicaciones NO hay informacion nueva, asi que el delta se MANTIENE (no cae a
        cero: eso el modelo lo leeria como "el fundamental dejo de mejorar" cuando solo es que
        aun no ha publicado el siguiente trimestre). Se calcula solo en las filas de nueva
        publicacion (donde el shift agrupado por ticker da la publicacion inmediatamente
        anterior, sin contar las repeticiones) y despues se propaga con ffill.
        """
        published_value = values.where(is_new_publication)
        # shift SOLO entre publicaciones reales: compactamos a esas filas, hacemos shift, y
        # reindexamos de vuelta. Asi la "anterior" es la publicacion previa, no la fila previa.
        pub_only = published_value.dropna()
        prev_pub = pub_only.groupby(frame.loc[pub_only.index, "ticker"]).shift()
        previous_value = prev_pub.reindex(frame.index)
        delta = published_value - previous_value
        return delta.groupby(frame["ticker"]).ffill()

    for source in MOMENTUM_SOURCES:
        values = pd.to_numeric(frame[source], errors="coerce")
        frame[f"mom_{source}"] = _delta_vs_previous_publication(values)

    # Descomposicion del cambio de P/E: Dlog(P/E) = Dlog(precio) - Dlog(EPS), cada componente
    # entre la publicacion actual y la anterior, constante entre publicaciones.
    frame["pe_fundamental_component"] = _delta_vs_previous_publication(np.log(frame["_eps_implied"]))
    frame["pe_price_component"] = _delta_vs_previous_publication(np.log(price))

    frame.drop(columns=["_eps_implied"], inplace=True)
    return frame


# B5: factores de interaccion regimen x factor que se anaden a los agentes.
REGIME_INTERACTION_FACTORS = (
    "factor_regime_bull",                       # el propio regimen (igual para todos ese dia)
    "factor_momentum_x_bull",                   # momentum pesa distinto en bull
    "factor_quality_x_bear",                    # calidad/defensa pesa distinto en bear
)


def _add_regime_features(frame: pd.DataFrame) -> None:
    """Anade el regimen de mercado y dos interacciones. Sin lookahead: el regimen se lee del
    retorno a 12 meses del benchmark HASTA la fecha del snapshot (dato pasado, ya PIT).

    - regime_bull = 1 si el SP500 subio en los ultimos 12 meses, 0 si no.
    - momentum_x_bull: rank de momentum relativo * bull (momentum importa mas en alcista).
    - quality_x_bear: rank de calidad * (1-bull) (defensa importa mas en bajista).
    El modelo Ridge aprende por si mismo el peso de cada interaccion; no imponemos la direccion.
    """
    bench_12m = pd.to_numeric(frame["benchmark_return_12m"], errors="coerce")
    bull = (bench_12m > 0).astype(float).where(frame["is_price_fresh"])
    frame["factor_regime_bull"] = bull

    momentum_rank = frame.get("factor_relative_return_12m")
    quality_rank = frame.get("factor_roe")
    frame["factor_momentum_x_bull"] = (momentum_rank * bull) if momentum_rank is not None else 0.0
    frame["factor_quality_x_bear"] = (
        (quality_rank * (1 - bull)) if quality_rank is not None else 0.0
    )


def _load_sector_map(settings: Settings) -> dict[str, str]:
    """Mapa ticker -> sector desde profiles.parquet, SOLO para agrupar (nunca como senal).

    El sector es un snapshot actual de Finnhub. Su uso como variable de neutralizacion
    introduce un lookahead residual menor (la industria de una empresa rara vez cambia), que se
    documenta en la bitacora. No entra en el panel ni se usa como feature predictiva.
    """
    profiles_path = settings.raw_output_dir / "profiles.parquet"
    if not profiles_path.exists():
        log.warning("neutralize_by_sector activo pero no hay %s; se rankea global.", profiles_path)
        return {}
    profiles = pd.read_parquet(profiles_path)
    if "finnhubIndustry" not in profiles.columns:
        return {}
    return {
        row.ticker: (row.finnhubIndustry or "N/A")
        for row in profiles[["ticker", "finnhubIndustry"]].itertuples(index=False)
    }


def _build_feature_frame(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    settings: Settings,
    sector_by_ticker: dict[str, str] | None = None,
) -> pd.DataFrame:
    required_panel = {"ticker", "snapshot_date", "in_sp500", "price_age_days", "price_return_3m"}
    required_benchmark = {"snapshot_date", "price_age_days", "price_return_3m"}
    if missing := required_panel - set(panel.columns):
        raise ValueError(f"panel_point_in_time.parquet no contiene: {sorted(missing)}")
    if missing := required_benchmark - set(benchmark.columns):
        raise ValueError(f"benchmark_point_in_time.parquet no contiene: {sorted(missing)}")

    bench = benchmark.rename(
        columns={
            "price": "benchmark_price",
            "price_as_of_date": "benchmark_price_as_of_date",
            "price_age_days": "benchmark_price_age_days",
            "price_return_1m": "benchmark_return_1m",
            "price_return_3m": "benchmark_return_3m",
            "price_return_6m": "benchmark_return_6m",
            "price_return_12m": "benchmark_return_12m",
        }
    )
    frame = panel.merge(bench, on="snapshot_date", how="left", validate="many_to_one")
    frame["price_age_days"] = pd.to_numeric(frame["price_age_days"], errors="coerce")
    frame["benchmark_price_age_days"] = pd.to_numeric(
        frame["benchmark_price_age_days"], errors="coerce"
    )
    frame["is_price_fresh"] = (
        frame["in_sp500"].fillna(False).astype(bool)
        & frame["price_age_days"].le(settings.max_price_age_days)
        & frame["benchmark_price_age_days"].le(settings.max_price_age_days)
    )

    # B3: tendencia de fundamentales y descomposicion precio/fundamental (point-in-time).
    if settings.fundamental_momentum:
        frame = _add_fundamental_momentum(frame)

    # Grupo de neutralizacion: sector si B1 esta activo, si no un unico grupo global.
    sector_by_ticker = sector_by_ticker or {}
    frame["_group"] = (
        frame["ticker"].map(sector_by_ticker).fillna("N/A") if sector_by_ticker else "ALL"
    )
    min_group = settings.neutralize_min_group if settings.neutralize_by_sector else 0

    for horizon in (3, 6, 12):
        stock = pd.to_numeric(frame[f"price_return_{horizon}m"], errors="coerce")
        index = pd.to_numeric(frame[f"benchmark_return_{horizon}m"], errors="coerce")
        relative = stock - index
        frame[f"relative_return_{horizon}m"] = relative.where(frame["is_price_fresh"])
        frame[f"factor_relative_return_{horizon}m"] = _cross_section_rank(
            frame, frame[f"relative_return_{horizon}m"], ascending=True, min_group=min_group
        )

    for source, (factor, ascending, positive_only) in FACTOR_SOURCES.items():
        values = pd.to_numeric(frame[source], errors="coerce")
        if positive_only:
            values = values.where(values.gt(0))
        values = values.where(frame["is_price_fresh"])
        frame[factor] = _cross_section_rank(frame, values, ascending=ascending, min_group=min_group)

    # B3: rankear las features de tendencia y descomposicion (mayor = mejor en todas: fundamental
    # que sube, y componente-fundamental positivo = barato por mejora del negocio, no por precio).
    if settings.fundamental_momentum:
        b3_sources = [f"mom_{name}" for name in MOMENTUM_SOURCES] + [
            "pe_fundamental_component", "pe_price_component"
        ]
        for source in b3_sources:
            values = pd.to_numeric(frame[source], errors="coerce").where(frame["is_price_fresh"])
            frame[f"factor_{source}"] = _cross_section_rank(
                frame, values, ascending=True, min_group=min_group
            )

    # B5: regimen de mercado (bull/bear) + interacciones factor x regimen.
    if settings.market_regime_feature:
        _add_regime_features(frame)

    frame.drop(columns=["_group"], inplace=True)
    frame.sort_values(["snapshot_date", "ticker"], inplace=True, ignore_index=True)
    return frame


def _cross_section_rank(
    frame: pd.DataFrame, values: pd.Series, ascending: bool, min_group: int = 0
) -> pd.Series:
    """Rank percentil del corte transversal.

    Con `min_group <= 0`, rankea todo el corte de cada fecha junto (comportamiento base). Con
    `min_group > 0`, rankea dentro de `(fecha, grupo)` — neutralizacion por sector (B1) — pero
    cae a ranking por fecha global para los grupos con menos de `min_group` miembros validos,
    porque un grupo diminuto da un percentil degenerado (un grupo de 1 siempre da 0.5).
    """
    if min_group <= 0:
        ranked = values.groupby(frame["snapshot_date"]).rank(
            method="average", pct=True, ascending=ascending
        )
        return ranked.where(frame["is_price_fresh"])

    # Tamano de cada (fecha, grupo) contando solo valores utilizables (no NA y precio fresco).
    usable = (values.notna() & frame["is_price_fresh"]).astype(int)
    group_size = usable.groupby([frame["snapshot_date"], frame["_group"]]).transform("sum")
    big_enough = group_size >= min_group

    within_group = values.groupby([frame["snapshot_date"], frame["_group"]]).rank(
        method="average", pct=True, ascending=ascending
    )
    global_rank = values.groupby(frame["snapshot_date"]).rank(
        method="average", pct=True, ascending=ascending
    )
    ranked = within_group.where(big_enough, global_rank)
    return ranked.where(frame["is_price_fresh"])


def _label_dates(settings: Settings) -> dict[str, str]:
    """Fecha de etiqueta de cada snapshot: el que está N posiciones más adelante en la rejilla.

    La fecha NO se deriva sumando meses. La rejilla clampa los fines de mes con
    `min(snapshot_day, days_in_month)` y `DateOffset` lo hace con otra regla distinta, así que
    con `SNAPSHOT_DAY = 31` había snapshots cuya etiqueta caía en un día inexistente: el merge
    no encontraba pareja y el target se degradaba a NaN sin ningún error, perdiendo ~40 % del
    entrenamiento en silencio. Tomando la rejilla como única fuente de verdad, la fecha de
    etiqueta existe por construcción, sea cual sea `SNAPSHOT_DAY`.

    `SNAPSHOT_DAY` es uno de los parámetros que barre la Fase 6: si la cobertura de etiquetas
    dependiese de su valor, la rejilla compararía escenarios con distinta cantidad de datos.
    Ver `docs/plan_fases.md` (Hallazgo 2).
    """
    grid = [date.date().isoformat() for date in snapshot_dates(settings)]
    step = settings.target_horizon_months // settings.snapshot_step_months
    return {date: grid[index + step] for index, date in enumerate(grid[:-step or None])}


def _build_targets(
    features: pd.DataFrame,
    asset_prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    base = features[["ticker", "snapshot_date", "price", "price_as_of_date", "is_price_fresh", "benchmark_price"]].copy()
    base["snapshot_ts"] = pd.to_datetime(base["snapshot_date"])
    base["future_snapshot_date"] = base["snapshot_date"].map(_label_dates(settings))

    future_assets = asset_prices.rename(
        columns={
            "snapshot_date": "future_snapshot_date",
            "price": "future_price",
            "price_as_of_date": "future_price_as_of_date",
            "price_age_days": "future_price_age_days",
        }
    )
    target = base.merge(
        future_assets,
        on=["ticker", "future_snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    future_benchmark = benchmark.rename(
        columns={
            "snapshot_date": "future_snapshot_date",
            "price": "future_benchmark_price",
            "price_as_of_date": "future_benchmark_as_of_date",
            "price_age_days": "future_benchmark_age_days",
        }
    )[["future_snapshot_date", "future_benchmark_price", "future_benchmark_as_of_date", "future_benchmark_age_days"]]
    target = target.merge(future_benchmark, on="future_snapshot_date", how="left", validate="many_to_one")
    target["future_price_age_days"] = pd.to_numeric(target["future_price_age_days"], errors="coerce")
    target["future_benchmark_age_days"] = pd.to_numeric(
        target["future_benchmark_age_days"], errors="coerce"
    )
    valid = (
        target["is_price_fresh"].fillna(False)
        & target["future_price_age_days"].le(settings.max_price_age_days)
        & target["future_benchmark_age_days"].le(settings.max_price_age_days)
        & pd.to_numeric(target["price"], errors="coerce").gt(0)
        & pd.to_numeric(target["benchmark_price"], errors="coerce").gt(0)
    )
    stock_return = target["future_price"] / target["price"] - 1
    benchmark_return = target["future_benchmark_price"] / target["benchmark_price"] - 1
    target["forward_return_3m"] = stock_return.where(valid)
    target["forward_benchmark_return_3m"] = benchmark_return.where(valid)
    target["forward_excess_return_3m"] = (stock_return - benchmark_return).where(valid)
    label_dates = pd.concat(
        [
            pd.to_datetime(target["future_price_as_of_date"]),
            pd.to_datetime(target["future_benchmark_as_of_date"]),
        ],
        axis=1,
    ).max(axis=1)
    target["label_end_date"] = label_dates.dt.date.astype("string")
    target["target_available"] = valid & target["forward_excess_return_3m"].notna()
    return target[
        [
            "ticker",
            "snapshot_date",
            "future_snapshot_date",
            "label_end_date",
            "target_available",
            "forward_return_3m",
            "forward_benchmark_return_3m",
            "forward_excess_return_3m",
        ]
    ]


def _coverage(features: pd.DataFrame, baselines: pd.DataFrame, settings: Settings) -> dict:
    factor_columns = [factor for factor, _, _ in FACTOR_SOURCES.values()]
    factor_columns.extend(["factor_relative_return_3m", "factor_relative_return_6m", "factor_relative_return_12m"])
    return {
        "run_scope": settings.run_scope,
        "max_price_age_days": settings.max_price_age_days,
        "rows": len(features),
        "fresh_rows": int(features["is_price_fresh"].sum()),
        "factor_rows": {column: int(features[column].notna().sum()) for column in factor_columns},
        "baseline_rows": {
            "garp": int(baselines["garp_score"].notna().sum()),
            "momentum": int(baselines["momentum_score"].notna().sum()),
        },
    }
