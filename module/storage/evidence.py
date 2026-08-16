"""Escritura de la evidencia final del único Model Study."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from module.common.utils import write_json
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR


def _number(value: Any, digits: int = 4, percent: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value * 100:.2f} %" if percent else f"{value:.{digits}f}"


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Informe con cifras reales y su procedencia.

    Cada número declara de qué artefacto sale y qué papel cumple —selección, confirmación fuera de
    muestra o diagnóstico—, porque una cifra sin ese papel es la clase de dato que un tribunal no
    puede evaluar: no sabe si la eligió el propio procedimiento que la reporta.
    """
    summary = payload.get("summary", {}).get("summary", {})
    confirmation = payload.get("summary", {}).get("confirmation_2025_2026", {})
    robustness = payload.get("robustness", {})
    attribution = payload.get("attribution", {})
    regression = attribution.get("factor_regression", {}).get("selection", {})
    neutral = attribution.get("neutralized_rank_ic", {})
    deflated = attribution.get("deflated_sharpe", {})
    dispersion = robustness.get("seed_dispersion", {})
    alpha_range = dispersion.get("geometric_excess_return", {})
    configuration = payload.get("configuration", {})

    lines = [
        "# Informe del Model Study",
        "",
        f"- Study: `{payload['study_id']}`",
        f"- Ganador: `{payload.get('winner_run_id', 'no disponible')}`",
        f"- Hash de dataset: `{payload.get('summary', {}).get('dataset_hash', '—')}`",
        f"- Selección: exclusivamente Rank-IC pareado hasta {SELECTION_UNTIL_YEAR}.",
        (f"- {KNOWN_STRESS_YEARS[0]}–{KNOWN_STRESS_YEARS[-1]}: confirmación fuera de muestra, "
         "no utilizada en ninguna decisión."),
        "",
        "## 1. Aprendizaje (ventana de selección)",
        "",
        "| Métrica | Valor | Artefacto |",
        "|---|---|---|",
        f"| Rank-IC medio | {_number(summary.get('mean_rank_ic'))} | `evidence/summary.json` |",
        f"| IC-IR | {_number(summary.get('ic_ir'))} | `evidence/summary.json` |",
        f"| Cohortes positivas | {_number(summary.get('rank_ic_positive_fraction'), percent=True)} | `evidence/summary.json` |",
        f"| Cohortes | {summary.get('n_cohorts', '—')} | `evidence/summary.json` |",
        f"| p permutación | {_number(robustness.get('permutation', {}).get('p_value'), 5)} | `robustness.json` |",
        "",
        "## 2. Confirmación fuera de muestra (no participó en ninguna decisión)",
        "",
        "| Métrica | Valor | Artefacto |",
        "|---|---|---|",
        f"| Rank-IC medio | {_number(confirmation.get('mean_rank_ic'))} | `evidence/summary.json` |",
        f"| Cohortes cerradas | {confirmation.get('n_cohorts', '—')} | `evidence/summary.json` |",
        (f"| Observaciones independientes | "
         f"{attribution.get('ic_significance', {}).get('confirmation', {}).get('effective_independent_observations', '—')} "
         "| `attribution.json` |"),
        "",
        ("Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda la "
         "etiqueta. El número de cohortes **no** es el número de pruebas independientes: esta "
         "confirmación es evidencia direccional del signo, no un contraste con potencia."),
        "",
        "## 3. Traducción a alfa (ventana de selección)",
        "",
        "| Métrica | Valor | Artefacto |",
        "|---|---|---|",
        f"| CAGR cartera | {_number(summary.get('cagr_portfolio'), percent=True)} | `evidence/summary.json` |",
        f"| CAGR benchmark | {_number(summary.get('cagr_benchmark'), percent=True)} | `evidence/summary.json` |",
        f"| Alfa geométrico | {_number(summary.get('geometric_excess_return'), percent=True)} | `evidence/summary.json` |",
        f"| Information Ratio anualizado | {_number(summary.get('information_ratio'))} | `evidence/summary.json` |",
        f"| Turnover anualizado | {_number(summary.get('annualized_turnover'), percent=True)} | `evidence/summary.json` |",
        f"| Efectivo medio | {_number(summary.get('mean_cash_weight'), percent=True)} | `evidence/summary.json` |",
        f"| Coeficiente de transferencia | {_number(summary.get('transfer_coefficient'))} | `evidence/summary.json` |",
        "",
        "## 4. ¿Aprende algo propio?",
        "",
        "| Métrica | Valor | Artefacto |",
        "|---|---|---|",
        f"| Alfa de la regresión por periodo | {_number(regression.get('alpha_per_period'), percent=True)} | `attribution.json` |",
        f"| t de Newey-West del alfa | {_number(regression.get('alpha_t_stat'), 2)} | `attribution.json` |",
        f"| Rank-IC bruto | {_number(neutral.get('raw_mean_rank_ic'))} | `attribution.json` |",
        f"| Rank-IC neutralizado por estilo | {_number(neutral.get('neutralized_mean_rank_ic'))} | `attribution.json` |",
        f"| Probabilidad Deflated Sharpe | {_number(deflated.get('deflated_sharpe_probability'), 3)} | `attribution.json` |",
        f"| Configuraciones probadas | {deflated.get('n_trials', '—')} | `attribution.json` |",
        "",
        "## 5. Estabilidad entre semillas",
        "",
        (f"- Alfa geométrico: mínimo {_number(alpha_range.get('min'), percent=True)}, "
         f"mediana {_number(alpha_range.get('median'), percent=True)}, "
         f"máximo {_number(alpha_range.get('max'), percent=True)}."),
        (f"- Conclusión económica estable entre semillas: "
         f"**{'sí' if dispersion.get('economic_conclusion_stable') else 'no'}**."),
        "",
        "## 6. Configuración ganadora",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(configuration.items())],
        "",
        "## Interpretación",
        "",
        ("La robustez, los perfiles, las carteras y la atribución son evidencia informativa "
         "posterior: se calculan con el ganador ya congelado y no modifican la configuración "
         "predictiva. La política de efectivo y el tamaño de cartera son decisiones de cartera, no "
         "de modelo, y se comparan en `portfolio_comparison.parquet`."),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_winner(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(dict(payload), path)


# Nombres legibles de las métricas que compara la mejora de cartera. El informe lo lee una persona,
# no un programa, así que la columna no puede ser el identificador interno.
IMPROVEMENT_LABELS = {
    "information_ratio": "Information Ratio",
    "geometric_excess_return": "Exceso geométrico",
    "annualized_turnover": "Turnover anualizado",
    "max_drawdown": "Máxima caída",
    "beat_rate": "Años que baten",
    "transfer_coefficient": "Coeficiente de transferencia",
}


def _break_even_line(block: Mapping[str, Any]) -> str:
    """Frase del equilibrio de costes, o el motivo por el que no hay una."""
    if not block:
        return "—"
    if block.get("available"):
        return (
            f"{_number(block.get('bps_per_trade'), 1)} pb por operación "
            f"({_number(block.get('pct_per_trade'), 2)} %, "
            f"{_number(block.get('round_trip_bps'), 1)} pb ida y vuelta)"
        )
    if block.get("beyond_ladder"):
        return f"por encima de {_number(block.get('last_cost_bps'), 0)} pb: fuera de la escalera medida"
    if block.get("never_positive"):
        return "no hay: el exceso ya es negativo sin coste alguno"
    return "—"


def write_portfolio_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Informe del Portfolio Study, con la misma disciplina que el del Model Study.

    Cada cifra declara de qué artefacto sale. Los diagnósticos que este study calcula —costes,
    capacidad y narrativa de cartera— son **posteriores** a la elección y no participaron en ella;
    el informe lo dice explícitamente para que nadie los lea como criterio.
    """
    winner = payload.get("winner_summary", {})
    confirmation = payload.get("winner_confirmation", {})
    combination = payload.get("winner_combination", {})
    improvement = payload.get("improvement", {})
    costs = payload.get("cost_sensitivity", {})
    capacity = payload.get("capacity", {})
    narrative = payload.get("portfolio_narrative", {}).get("windows", {}).get("selection", {})
    frozen = costs.get("break_even", {}).get("frozen_path", {}).get("selection", {})
    resimulated = costs.get("break_even", {}).get("resimulated", {}).get("selection", {})
    capacity_window = capacity.get("windows", {}).get("selection", {})
    maximum_aum = capacity_window.get("maximum_aum_usd", {})
    duration = narrative.get("holding_duration", {})
    concentration = narrative.get("concentration", {})

    lines = [
        "# Informe del Portfolio Study",
        "",
        f"- Study: `{payload.get('study_id', '—')}`",
        f"- Model Study de origen: `{payload.get('source_study_id', '—')}`",
        f"- Combinaciones evaluadas: {payload.get('combinations', '—')}",
        (f"- Criterio: `{payload.get('selection_metric', '—')}` sobre la ventana de selección "
         f"hasta {SELECTION_UNTIL_YEAR}."),
        (f"- {KNOWN_STRESS_YEARS[0]}–{KNOWN_STRESS_YEARS[-1]}: confirmación fuera de muestra. La "
         "rejilla ni siquiera la simuló."),
        "",
        "## 1. Cartera ganadora",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(combination.items())],
        "",
        "| Métrica | Selección | Era reservada | Artefacto |",
        "|---|---|---|---|",
        *[
            f"| {label} | {_number(winner.get(key), percent=percent)} | "
            f"{_number(confirmation.get(key), percent=percent)} | `portfolio_winner.json` |"
            for label, key, percent in (
                ("Information Ratio", "information_ratio", False),
                ("Exceso geométrico", "geometric_excess_return", True),
                ("CAGR cartera", "cagr_portfolio", True),
                ("CAGR benchmark", "cagr_benchmark", True),
                ("Máxima caída", "max_drawdown", True),
                ("Años que baten", "beat_rate", True),
                ("Turnover anualizado", "annualized_turnover", True),
            )
        ],
        "",
        "## 2. Qué aporta frente a la cartera del modelo",
        "",
        "| Métrica | Cartera del modelo | Cartera optimizada | Diferencia |",
        "|---|---|---|---|",
        *[
            f"| {IMPROVEMENT_LABELS.get(key, key)} | {_number(block.get('baseline'))} | "
            f"{_number(block.get('winner'))} | {_number(block.get('delta'))} |"
            for key, block in sorted(improvement.items())
        ],
        "",
        "## 3. ¿Aguanta el supuesto de coste?",
        "",
        f"- Coste adoptado: {_number(costs.get('adopted_cost_bps'), 1)} pb por operación.",
        f"- Equilibrio de ruta congelada (conservador): {_break_even_line(frozen)}.",
        f"- Equilibrio resimulado: {_break_even_line(resimulated)}.",
        "",
        ("El equilibrio se define **contra el índice**: es el coste al que el exceso geométrico se "
         "anula. La familia congelada mantiene las decisiones ya tomadas, así que subestima el "
         "margen; la resimulada deja que la cartera opere menos al encarecerse. Artefacto: "
         "`cost_sensitivity.json`, con sus salvedades dentro."),
        "",
        "## 4. ¿Hasta qué patrimonio es ejecutable?",
        "",
        *[
            f"- Con órdenes por debajo del {threshold} del volumen diario habitual: hasta "
            f"{_number(value, 0)} USD."
            for threshold, value in sorted(maximum_aum.items())
        ],
        (f"- Cobertura de volumen de las órdenes: "
         f"{_number(capacity_window.get('volume_coverage'), percent=True)}."),
        "",
        ("Se mide participación sobre el volumen habitual, no impacto de mercado. Artefacto: "
         "`capacity.json`."),
        "",
        "## 5. Qué hizo la cartera",
        "",
        f"- Acciones distintas que llegó a tener: {concentration.get('distinct_tickers_ever_held', '—')}.",
        f"- Posiciones simultáneas (media): {_number(concentration.get('mean_positions'), 1)}.",
        f"- Permanencia mediana de una posición: {_number(duration.get('median_months'), 1)} meses.",
        f"- Episodios cerrados: {duration.get('episodes', '—')}.",
        "",
        ("Los nombres más presentes, las mayores y menores contribuciones, las mejores y peores "
         "operaciones cerradas y las ventas que luego subieron están en "
         "`portfolio_narrative.json` y `portfolio_narrative_holdings.parquet`. El mapa de "
         "posiciones se lee de `evidence_best_full/positions.parquet`."),
        "",
        "## Interpretación",
        "",
        ("Los tres diagnósticos de este informe —costes, capacidad y narrativa— se calculan con la "
         "cartera **ya elegida** y no intervinieron en la elección. En particular, el coste nunca "
         "selecciona: optimizarlo sería escoger el mundo en el que la estrategia luce mejor."),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
