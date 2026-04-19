"""Organized results hub generation for easier navigation and comparison."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _extract_quarter(file_name: str) -> str:
    m = re.search(r"quarter_([A-Za-z0-9\-]+)", file_name)
    return m.group(1) if m else ""


def _extract_fold(file_name: str) -> str:
    m = re.search(r"fold_([A-Za-z0-9\-]+)", file_name)
    return m.group(1) if m else ""


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _agent_from_path(path: Path, agents_root: Path) -> str:
    try:
        rel = path.relative_to(agents_root)
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if parts else ""


def _write_by_fold_indexes(catalog_df: pd.DataFrame, out_dir: Path) -> None:
    by_fold_dir = out_dir / "by_fold"
    by_fold_dir.mkdir(parents=True, exist_ok=True)

    quarter_vals = [q for q in sorted(catalog_df["quarter"].dropna().unique().tolist()) if str(q).strip()]
    for quarter in quarter_vals:
        q_dir = by_fold_dir / str(quarter)
        q_dir.mkdir(parents=True, exist_ok=True)
        fold_df = catalog_df[catalog_df["quarter"] == quarter].copy()
        fold_df.sort_values(["source_group", "relative_path"], inplace=True)
        fold_df.to_csv(q_dir / "files.csv", index=False)


def _write_by_agent_indexes(catalog_df: pd.DataFrame, out_dir: Path) -> None:
    by_agent_dir = out_dir / "by_agent"
    by_agent_dir.mkdir(parents=True, exist_ok=True)

    agents = [a for a in sorted(catalog_df["agent"].dropna().unique().tolist()) if str(a).strip()]
    for agent in agents:
        a_dir = by_agent_dir / str(agent)
        a_dir.mkdir(parents=True, exist_ok=True)
        sub = catalog_df[catalog_df["agent"] == agent].copy().sort_values("relative_path")
        sub.to_csv(a_dir / "files.csv", index=False)


def _build_feature_usage_overview(agents_root: Path, overview_dir: Path) -> None:
    files = sorted(agents_root.glob("quarter_*_feature_usage_report.csv"))
    if not files:
        return

    all_rows: List[pd.DataFrame] = []
    used_long_rows: List[Dict] = []
    for path in files:
        df = _safe_read_csv(path)
        if df.empty:
            continue
        quarter = _extract_quarter(path.name)
        df = df.copy()
        df["quarter"] = quarter
        all_rows.append(df)

        for _, row in df.iterrows():
            used_str = str(row.get("used", "") or "")
            used_feats = [u for u in used_str.split("|") if u]
            for feat in used_feats:
                used_long_rows.append(
                    {
                        "quarter": quarter,
                        "agent": row.get("agent", ""),
                        "feature": feat,
                    }
                )

    if not all_rows:
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(overview_dir / "feature_usage_all_folds.csv", index=False)

    summary_cols = [
        "requested_n",
        "available_n",
        "used_n",
        "missing_n",
        "not_calculated_or_no_data_n",
        "low_data_coverage_n",
        "skipped_due_to_data_n",
        "used_not_requested_n",
    ]
    available_summary_cols = [c for c in summary_cols if c in full_df.columns]
    grouped = (
        full_df.groupby("agent", as_index=False)[available_summary_cols].mean(numeric_only=True)
        if available_summary_cols
        else pd.DataFrame()
    )
    if not grouped.empty:
        grouped.to_csv(overview_dir / "feature_usage_agent_summary.csv", index=False)

    if used_long_rows:
        used_long_df = pd.DataFrame(used_long_rows)
        used_long_df.to_csv(overview_dir / "feature_usage_selected_features_long.csv", index=False)
        top_feats = (
            used_long_df.groupby(["agent", "feature"], as_index=False)
            .size()
            .rename(columns={"size": "n_folds_selected"})
            .sort_values(["agent", "n_folds_selected", "feature"], ascending=[True, False, True])
        )
        top_feats.to_csv(overview_dir / "feature_usage_top_features_by_agent.csv", index=False)


def _build_sector_feature_overview(agents_root: Path, overview_dir: Path) -> None:
    files = sorted(agents_root.glob("*/sectors/*/feature_importances*.csv"))
    if not files:
        return

    rows: List[Dict] = []
    for path in files:
        df = _safe_read_csv(path)
        if df.empty:
            continue

        # Expected structure from save_feature_importances: index=feature, col=importance
        if "importance" in df.columns and df.shape[1] >= 2:
            if df.columns[0] != "feature":
                df = df.rename(columns={df.columns[0]: "feature"})
        elif df.shape[1] == 2:
            df = df.rename(columns={df.columns[0]: "feature", df.columns[1]: "importance"})
        else:
            continue

        if "feature" not in df.columns or "importance" not in df.columns:
            continue

        agent = path.parts[-4] if len(path.parts) >= 4 else ""
        sector = path.parts[-2] if len(path.parts) >= 2 else ""
        fold = _extract_fold(path.name)

        top_df = (
            df[["feature", "importance"]]
            .copy()
            .sort_values("importance", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        for i, r in top_df.iterrows():
            rows.append(
                {
                    "agent": agent,
                    "sector": sector,
                    "fold": fold,
                    "rank": int(i + 1),
                    "feature": r["feature"],
                    "importance": float(r["importance"]),
                    "source_file": path.as_posix(),
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(overview_dir / "sector_top_features.csv", index=False)


def _build_agent_training_overview(agent_diag_history: Dict[str, List], overview_dir: Path) -> None:
    rows: List[Dict] = []
    for agent, history in (agent_diag_history or {}).items():
        if not history:
            continue
        for h in history:
            row = {
                "agent": agent,
                "fold": h.get("fold", ""),
                "n_features": h.get("n_features", np.nan),
                "n_sector_obs": h.get("n_sector_obs", np.nan),
                "n_sectors_seen": h.get("n_sectors_seen", np.nan),
                "n_sectors_trained": h.get("n_sectors_trained", np.nan),
                "n_sectors_skipped": h.get("n_sectors_skipped", np.nan),
            }
            cv = h.get("cv_metrics") or h.get("cv_lr") or {}
            row["cv_mean_auc"] = cv.get("mean_auc", np.nan)
            row["cv_std_auc"] = cv.get("std_auc", np.nan)
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(overview_dir / "agent_training_overview.csv", index=False)


def build_results_hub(
    *,
    results_root: str,
    agents_results_dir: str,
    backtest_results_dir: str,
    plots_dir: str,
    fold_results: List[Dict],
    agent_diag_history: Dict[str, List],
) -> Optional[Path]:
    """Create an organized, human-friendly index of all generated outputs."""
    root = Path(results_root)
    hub = root / "organized"
    overview = hub / "overview"
    overview.mkdir(parents=True, exist_ok=True)

    agents_root = Path(agents_results_dir)
    backtest_root = Path(backtest_results_dir)
    plots_root = Path(plots_dir)

    file_rows: List[Dict] = []
    grouped_roots = [
        ("agents", agents_root),
        ("backtest", backtest_root),
        ("plots", plots_root),
    ]
    for group, group_root in grouped_roots:
        for file_path in _iter_files(group_root):
            rel = file_path.relative_to(root) if file_path.is_relative_to(root) else file_path
            file_rows.append(
                {
                    "source_group": group,
                    "relative_path": rel.as_posix(),
                    "file_name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_kb": round(file_path.stat().st_size / 1024.0, 3),
                    "quarter": _extract_quarter(file_path.name),
                    "fold": _extract_fold(file_path.name),
                    "agent": _agent_from_path(file_path, agents_root),
                }
            )

    if not file_rows:
        return None

    catalog_df = pd.DataFrame(file_rows).sort_values(["source_group", "relative_path"]).reset_index(drop=True)
    catalog_df.to_csv(overview / "artifact_catalog.csv", index=False)

    # Quick high-level fold metrics for comparison.
    if fold_results:
        fold_df = pd.DataFrame(fold_results)
        cols = [
            c
            for c in [
                "fold",
                "analysis_quarter",
                "train_years",
                "strategy_cumulative_return",
                "benchmark_cumulative_return",
                "alpha",
                "strategy_sharpe",
                "roc_auc",
                "precision",
                "recall",
                "f1",
            ]
            if c in fold_df.columns
        ]
        if cols:
            fold_df[cols].to_csv(overview / "fold_metrics_overview.csv", index=False)

    _build_feature_usage_overview(agents_root, overview)
    _build_sector_feature_overview(agents_root, overview)
    _build_agent_training_overview(agent_diag_history, overview)

    _write_by_fold_indexes(catalog_df, hub)
    _write_by_agent_indexes(catalog_df, hub)

    readme = hub / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Results Hub",
                "",
                "Vista simplificada para navegar y comparar resultados sin revisar cientos de archivos sueltos.",
                "",
                "## Carpeta overview",
                "- artifact_catalog.csv: inventario completo de archivos generados.",
                "- fold_metrics_overview.csv: comparativa de rendimiento por fold.",
                "- feature_usage_all_folds.csv: detalle de features solicitadas/disponibles/usadas.",
                "- feature_usage_agent_summary.csv: resumen agregado por agente.",
                "- feature_usage_selected_features_long.csv: features seleccionadas por agente y fold.",
                "- feature_usage_top_features_by_agent.csv: top de features mas repetidas por agente.",
                "- sector_top_features.csv: top features por agente-sector para modelos especializados.",
                "- agent_training_overview.csv: estado de entrenamiento y AUC por agente/fold.",
                "",
                "## Carpeta by_fold",
                "- Un subdirectorio por quarter con files.csv para ver todos los artefactos de ese fold.",
                "",
                "## Carpeta by_agent",
                "- Un subdirectorio por agente con files.csv para ver rapidamente sus artefactos.",
            ]
        ),
        encoding="utf-8",
    )

    return hub
