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
