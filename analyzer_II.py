"""Run analyzer scenarios in parallel and compare common fold alpha.

The script launches normal analyzer runs with different environment overrides,
waits until all finish while printing live progress, then scans the generated
result folders, finds folds shared by all successful runs, and plots the alpha
comparison in a single figure.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _experiments() -> list[dict[str, Any]]:
    """Return a 9-scenario calibration grid for the final TP/SL selection.

    The sweep keeps the current proven execution/control settings fixed and
    only varies the three selection-calibration knobs introduced for ticker
    ranking:
    - TP_SL_MIN_ACCEPTABLE_TP
    - TP_SL_SELECTION_CERTAINTY_WEIGHT
    - TP_SL_SELECTION_TP_QUALITY_WEIGHT

    This isolates the effect of "TP reachable + certainty" versus "TP too low
    or too stretched" so the comparison remains interpretable and focused on
    the final ticker selection step.
    """
    # Base reflects the current proven control: 12Y/8Y, no-sentiment, 7-3,
    # TP-edge active with blend=0.10, and the current rule-driven selection
    # overlay kept fixed while we sweep only the final selection calibration.
    _BASE = {
        "WALKFORWARD_TRAIN_LOOKBACK_YEARS": 12,
        "WALKFORWARD_TRAIN_MIN_YEARS": 8,
        "TP_SL_MAX_STOCKS": 7,
        "TP_SL_MIN_STOCKS": 3,
        "PORTFOLIO_MIN_SCORE": 0.57,
        "PORTFOLIO_MAX_STOCK_WEIGHT": 0.20,
        "SECTOR_SCORE_PRIOR_WEIGHT": 0.15,
        "SCORE_DISPERSION_MIN_STD": 0.05,
        "ENABLE_SENTIMENT_AGENT": False,
        "ENABLE_AGENT_RULE_ENGINE": True,
        "TP_EDGE_ENABLE": True,
        "META_ENABLE_CONSENSUS_FEATURES": True,
        "TP_SL_RULE_SIGNAL_RBS_WEIGHT": 0.25,
        "TP_SL_MIN_ACCEPTABLE_TP": 0.07,
        "TP_SL_SELECTION_CERTAINTY_WEIGHT": 0.35,
        "TP_SL_SELECTION_TP_QUALITY_WEIGHT": 0.25,
        # blend=0.10 is now the env.py default; stated explicitly so the reference
        # is self-documenting and immune to future default changes.
        "TP_EDGE_CONFIDENCE_BLEND": 0.10,
        "ENABLE_BUY_HOLD_COUNTERFACTUAL": True,
        "EXPORT_TP_SL_VS_BUY_HOLD": True,
        "ENABLE_TP_SL_RESEARCH_VARIANTS": True,
        "TP_SL_VARIANT_MODE": "base",  # switch to "hybrid_learned" to promote hybrid as main TP/SL
    }

    def _merge(**kwargs: Any) -> dict[str, Any]:
        """Return _BASE with the given keys overridden."""
        return {**_BASE, **kwargs}

    def _scenario(name: str, description: str, **overrides: Any) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "overrides": _merge(**overrides),
        }

    floor_levels = [
        ("floor006", 0.06, "TP floor 6%: slightly permissive, but still above the tiny-TP zone."),
        ("floor007", 0.07, "TP floor 7%: current reference floor, balanced against feasibility."),
        ("floor009", 0.09, "TP floor 9%: stricter floor to force more meaningful upside."),
    ]

    profile_levels = [
        ("cert045_q018", 0.45, 0.18, "certainty-heavy: prefers high-confidence names with moderate TP quality."),
        ("cert035_q025", 0.35, 0.25, "balanced: close to the current default calibration."),
        ("cert028_q034", 0.28, 0.34, "TP-quality-heavy: allows a bit less certainty in exchange for better target quality."),
    ]

    scenarios: list[dict[str, Any]] = []
    scenario_idx = 1
    for floor_tag, min_tp, floor_desc in floor_levels:
        for profile_tag, certainty_weight, tp_quality_weight, profile_desc in profile_levels:
            scenarios.append(
                _scenario(
                    f"exp{scenario_idx:02d}_{floor_tag}_{profile_tag}",
                    f"{floor_desc} {profile_desc}",
                    TP_SL_MIN_ACCEPTABLE_TP=min_tp,
                    TP_SL_SELECTION_CERTAINTY_WEIGHT=certainty_weight,
                    TP_SL_SELECTION_TP_QUALITY_WEIGHT=tp_quality_weight,
                )
            )
            scenario_idx += 1

    return scenarios


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _run_config_matches(run_config_path: Path, batch_id: str, experiment_name: str) -> bool:
    payload = _json_load(run_config_path)
    if not payload:
        return False

    run_context = payload.get("run_context") or {}
    return (
        run_context.get("batch_id") == batch_id
        and run_context.get("experiment_name") == experiment_name
    )


def _find_result_dir(repo_root: Path, batch_id: str, experiment_name: str) -> Path | None:
    results_root = repo_root / "results"
    if not results_root.exists():
        return None

    candidates: list[Path] = []
    for top_level in results_root.iterdir():
        if not top_level.is_dir():
            continue
        run_config = top_level / "general" / "run_config.json"
        if run_config.exists() and _run_config_matches(run_config, batch_id, experiment_name):
            candidates.append(top_level)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_run_summary(result_dir: Path | None) -> dict[str, Any] | None:
    if result_dir is None:
        return None

    for path in [
        result_dir / "strategy" / "final_summary.json",
        result_dir / "strategy" / "backtest_summary.json",
    ]:
        if path.exists():
            return _json_load(path)
    return None


def _extract_summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}

    if "legacy_summary" in summary or "usd_strategy" in summary:
        legacy = summary.get("legacy_summary") or {}
        usd_strategy = summary.get("usd_strategy") or {}
        return {
            "overall_mean_alpha": legacy.get("mean_alpha"),
            "overall_pct_positive_alpha": legacy.get("pct_folds_positive_alpha"),
            "overall_strategy_sharpe": legacy.get("global_strategy_sharpe"),
            "overall_benchmark_sharpe": legacy.get("global_benchmark_sharpe"),
            "overall_strategy_return": usd_strategy.get("total_return_pct"),
            "overall_strategy_drawdown": usd_strategy.get("max_drawdown"),
            "overall_strategy_fees": usd_strategy.get("total_fees_usd"),
            "n_folds": legacy.get("n_folds"),
            "tp_sl_folds_wins_vs_buy_hold": (legacy.get("tp_sl_vs_buy_hold") or {}).get("folds_tp_sl_wins"),
            "mean_alpha_buy_hold": (legacy.get("tp_sl_vs_buy_hold") or {}).get("mean_alpha_buy_hold"),
            "mean_tp_sl_minus_buy_hold": (legacy.get("tp_sl_vs_buy_hold") or {}).get("mean_tp_sl_minus_buy_hold"),
            "hybrid_folds_wins_vs_base": (legacy.get("tp_sl_hybrid_vs_base") or {}).get("folds_hybrid_wins"),
            "mean_hybrid_minus_base": (legacy.get("tp_sl_hybrid_vs_base") or {}).get("mean_hybrid_minus_base"),
        }

    return {
        "overall_mean_alpha": summary.get("mean_alpha"),
        "overall_pct_positive_alpha": summary.get("pct_folds_positive_alpha"),
        "overall_strategy_sharpe": summary.get("strategy_sharpe") or summary.get("global_strategy_sharpe"),
        "overall_benchmark_sharpe": summary.get("benchmark_sharpe") or summary.get("global_benchmark_sharpe"),
        "overall_strategy_return": summary.get("strategy_total_return_pct") or summary.get("global_strategy_cumulative_return"),
        "overall_strategy_drawdown": summary.get("strategy_max_drawdown") or summary.get("global_strategy_max_drawdown"),
        "overall_strategy_fees": summary.get("strategy_total_fees_usd"),
        "n_folds": summary.get("n_folds"),
        "tp_sl_folds_wins_vs_buy_hold": (summary.get("tp_sl_vs_buy_hold") or {}).get("folds_tp_sl_wins"),
        "mean_alpha_buy_hold": (summary.get("tp_sl_vs_buy_hold") or {}).get("mean_alpha_buy_hold"),
        "mean_tp_sl_minus_buy_hold": (summary.get("tp_sl_vs_buy_hold") or {}).get("mean_tp_sl_minus_buy_hold"),
        "hybrid_folds_wins_vs_base": (summary.get("tp_sl_hybrid_vs_base") or {}).get("folds_hybrid_wins"),
        "mean_hybrid_minus_base": (summary.get("tp_sl_hybrid_vs_base") or {}).get("mean_hybrid_minus_base"),
    }


def _parse_fold_label(label: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d{4})Q([1-4])$", str(label).strip())
    if match:
        return int(match.group(1)), int(match.group(2)), str(label)

    match = re.match(r"^(\d{4})$", str(label).strip())
    if match:
        return int(match.group(1)), 0, str(label)

    return 9999, 99, str(label)


def _fmt_pct(value: Any) -> str:
    try:
        if value is None or value == "":
            return ""
        return f"{float(value):+.2%}"
    except Exception:
        return str(value)


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if value is None or value == "":
            return ""
        return f"{float(value):+.{digits}f}"
    except Exception:
        return str(value)


def _launch_process(
    repo_root: Path,
    batch_id: str,
    out_dir: Path,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    name = str(experiment["name"])
    description = str(experiment.get("description", ""))
    overrides = dict(experiment["overrides"])
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    env = os.environ.copy()
    env["ENV_OVERRIDES_JSON"] = json.dumps(overrides, ensure_ascii=True)
    env["EXPERIMENT_NAME"] = name
    env["EXPERIMENT_BATCH_ID"] = batch_id

    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "analyzer.py"],
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        log_handle.close()
        raise

    return {
        "name": name,
        "description": description,
        "overrides": overrides,
        "process": process,
        "log_handle": log_handle,
        "log_path": log_path,
        "started_at": datetime.now().isoformat(),
        "started_monotonic": time.monotonic(),
    }


def _load_fold_alphas(result_dir: Path) -> dict[str, float] | None:
    csv_path = result_dir / "strategy" / "folds_results.csv"
    if not csv_path.exists():
        return None

    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty or "fold" not in df.columns or "alpha" not in df.columns:
        return None

    fold_map: dict[str, float] = {}
    for _, row in df.iterrows():
        fold = str(row.get("fold", "")).strip()
        if not fold:
            continue
        try:
            fold_map[fold] = float(row.get("alpha"))
        except Exception:
            continue
    return fold_map


def _write_comparison_artifacts(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    import matplotlib

    successful = [row for row in rows if int(row.get("returncode", 1)) == 0 and row.get("result_dir")]
    fold_maps: dict[str, dict[str, float]] = {}
    for row in successful:
        result_dir = Path(str(row["result_dir"]))
        fold_map = _load_fold_alphas(result_dir)
        if fold_map:
            fold_maps[row["experiment"]] = fold_map

    common_folds: list[str] = []
    if fold_maps:
        fold_sets = [set(m.keys()) for m in fold_maps.values()]
        common_folds = sorted(set.intersection(*fold_sets), key=_parse_fold_label) if fold_sets else []

    matrix_rows: list[dict[str, Any]] = []
    if common_folds:
        matrix = pd.DataFrame(index=common_folds)
        for exp_name, fold_map in fold_maps.items():
            matrix[exp_name] = [fold_map.get(fold) for fold in common_folds]

        matrix.index.name = "fold"
        matrix.to_csv(out_dir / "common_fold_alpha_matrix.csv", float_format="%.6f")

        ranking = pd.DataFrame(
            {
                "experiment": matrix.columns,
                "mean_alpha_common_folds": [float(matrix[col].mean()) for col in matrix.columns],
                "pct_positive_common_folds": [float((matrix[col] > 0).mean()) for col in matrix.columns],
                "n_common_folds": [int(matrix[col].count()) for col in matrix.columns],
            }
        ).sort_values("mean_alpha_common_folds", ascending=False)
        ranking.to_csv(out_dir / "common_fold_alpha_ranking.csv", index=False, float_format="%.6f")

        for fold in common_folds:
            row = {"fold": fold}
            for exp_name in matrix.columns:
                row[exp_name] = matrix.loc[fold, exp_name]
            matrix_rows.append(row)
        pd.DataFrame(matrix_rows).to_csv(out_dir / "common_fold_alpha_by_fold.csv", index=False, float_format="%.6f")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(15, 7))
        x = list(range(len(common_folds)))
        for exp_name in matrix.columns:
            ax.plot(x, matrix[exp_name].values * 100, marker="o", linewidth=2.0, label=exp_name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(common_folds, rotation=45, ha="right")
        ax.set_ylabel("Alpha (%)")
        ax.set_title("Alpha por fold común entre escenarios")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "common_fold_alpha_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        report_lines = [
            "# Parallel Analyzer Comparison",
            "",
            f"Batch id: `{out_dir.name}`",
            f"Successful runs: {len(successful)}",
            f"Common folds: {', '.join(common_folds)}" if common_folds else "Common folds: none",
            "",
            "## Ranking by common-fold mean alpha",
            "",
            "| Rank | Experiment | Mean alpha | % Positive | N common folds | Result dir |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for rank, (_, row) in enumerate(ranking.iterrows(), start=1):
            exp_name = str(row["experiment"])
            result_dir = next((r.get("result_dir", "") for r in successful if r.get("experiment") == exp_name), "")
            report_lines.append(
                f"| {rank} | {exp_name} | {_fmt_pct(row['mean_alpha_common_folds'])} | {_fmt_pct(row['pct_positive_common_folds'])} | {int(row['n_common_folds'])} | {result_dir} |"
            )

        report_lines.extend(["", "## Per-fold alpha matrix", "", "See `common_fold_alpha_matrix.csv` and `common_fold_alpha_comparison.png`." ])
        (out_dir / "comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    else:
        (out_dir / "comparison_report.md").write_text(
            "# Parallel Analyzer Comparison\n\nNo common folds were found across successful runs.\n",
            encoding="utf-8",
        )

    scenario_summary_csv = out_dir / "scenario_summary.csv"
    scenario_summary = pd.DataFrame(rows).copy()
    if not scenario_summary.empty:
        def _parse_overrides(raw: Any) -> dict[str, Any]:
            if isinstance(raw, dict):
                return raw
            if not isinstance(raw, str) or not raw.strip():
                return {}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

        overrides_series = scenario_summary["overrides"].apply(_parse_overrides) if "overrides" in scenario_summary.columns else pd.Series([{}] * len(scenario_summary), index=scenario_summary.index)
        overrides_frame = pd.DataFrame(list(overrides_series)) if len(overrides_series) else pd.DataFrame(index=scenario_summary.index)

        def _override_column(key: str) -> pd.Series:
            if key in overrides_frame.columns:
                return pd.to_numeric(overrides_frame[key], errors="coerce")
            return pd.Series([float("nan")] * len(scenario_summary), index=scenario_summary.index)

        scenario_summary["tp_sl_min_acceptable_tp"] = _override_column("TP_SL_MIN_ACCEPTABLE_TP")
        scenario_summary["tp_sl_selection_certainty_weight"] = _override_column("TP_SL_SELECTION_CERTAINTY_WEIGHT")
        scenario_summary["tp_sl_selection_tp_quality_weight"] = _override_column("TP_SL_SELECTION_TP_QUALITY_WEIGHT")
        scenario_summary["tp_sl_rule_signal_rbs_weight"] = _override_column("TP_SL_RULE_SIGNAL_RBS_WEIGHT")
        scenario_summary["tp_edge_confidence_blend"] = _override_column("TP_EDGE_CONFIDENCE_BLEND")

        rank_specs = [
            ("rank_common_fold_mean_alpha", "common_fold_mean_alpha", False),
            ("rank_overall_mean_alpha", "overall_mean_alpha", False),
            ("rank_overall_strategy_sharpe", "overall_strategy_sharpe", False),
            ("rank_overall_strategy_return", "overall_strategy_return", False),
            ("rank_overall_strategy_drawdown", "overall_strategy_drawdown", False),
            ("rank_overall_strategy_fees", "overall_strategy_fees", True),
        ]
        for rank_col, metric_col, ascending in rank_specs:
            if metric_col in scenario_summary.columns:
                scenario_summary[rank_col] = pd.to_numeric(scenario_summary[metric_col], errors="coerce").rank(ascending=ascending, method="min")

        sort_metric = None
        for candidate in ["common_fold_mean_alpha", "overall_mean_alpha"]:
            if candidate in scenario_summary.columns:
                candidate_values = pd.to_numeric(scenario_summary[candidate], errors="coerce")
                if candidate_values.notna().any():
                    sort_metric = candidate
                    scenario_summary[candidate] = candidate_values
                    break

        if sort_metric is not None:
            scenario_summary = scenario_summary.sort_values(sort_metric, ascending=False, na_position="last")

        preferred_columns = [
            "rank_common_fold_mean_alpha",
            "experiment",
            "description",
            "tp_sl_min_acceptable_tp",
            "tp_sl_selection_certainty_weight",
            "tp_sl_selection_tp_quality_weight",
            "tp_sl_rule_signal_rbs_weight",
            "tp_edge_confidence_blend",
            "returncode",
            "duration_seconds",
            "n_folds",
            "overall_mean_alpha",
            "common_fold_mean_alpha",
            "overall_pct_positive_alpha",
            "common_fold_pct_positive",
            "overall_strategy_sharpe",
            "overall_benchmark_sharpe",
            "overall_strategy_return",
            "overall_strategy_drawdown",
            "overall_strategy_fees",
            "tp_sl_folds_wins_vs_buy_hold",
            "mean_alpha_buy_hold",
            "mean_tp_sl_minus_buy_hold",
            "hybrid_folds_wins_vs_base",
            "mean_hybrid_minus_base",
            "common_fold_count",
            "rank_overall_mean_alpha",
            "rank_overall_strategy_sharpe",
            "rank_overall_strategy_return",
            "rank_overall_strategy_drawdown",
            "rank_overall_strategy_fees",
            "result_dir",
            "summary_file",
            "overrides",
        ]
        ordered_columns = [column for column in preferred_columns if column in scenario_summary.columns]
        ordered_columns.extend(column for column in scenario_summary.columns if column not in ordered_columns)
        scenario_summary = scenario_summary[ordered_columns]

    scenario_summary.to_csv(scenario_summary_csv, index=False, float_format="%.6f")

    summary_csv = out_dir / "summary.csv"
    fieldnames = [
        "experiment",
        "description",
        "returncode",
        "started_at",
        "finished_at",
        "duration_seconds",
        "result_dir",
        "summary_file",
        "overall_mean_alpha",
        "overall_pct_positive_alpha",
        "overall_strategy_sharpe",
        "overall_benchmark_sharpe",
        "overall_strategy_return",
        "overall_strategy_drawdown",
        "overall_strategy_fees",
        "tp_sl_folds_wins_vs_buy_hold",
        "mean_alpha_buy_hold",
        "mean_tp_sl_minus_buy_hold",
        "hybrid_folds_wins_vs_base",
        "mean_hybrid_minus_base",
        "n_folds",
        "common_fold_mean_alpha",
        "common_fold_pct_positive",
        "common_fold_count",
        "overrides",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "results" / "parallel_experiments" / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = _experiments()
    _json_dump(out_dir / "manifest.json", experiments)

    print(f"Starting batch {batch_id} with {len(experiments)} scenarios.", flush=True)

    running: dict[str, dict[str, Any]] = {}
    completed_rows: list[dict[str, Any]] = []

    for index, experiment in enumerate(experiments, start=1):
        state = _launch_process(repo_root, batch_id, out_dir, experiment)
        running[state["name"]] = state
        print(
            f"[START {index}/{len(experiments)}] {state['name']} pid={state['process'].pid} | {state['description']}",
            flush=True,
        )

    heartbeat_interval = 300.0
    poll_interval = 5.0
    last_heartbeat = time.monotonic()

    try:
        while running:
            finished_names: list[str] = []
            for name, state in list(running.items()):
                process = state["process"]
                return_code = process.poll()
                if return_code is None:
                    continue

                state["log_handle"].close()
                finished_at = datetime.now().isoformat()
                duration_seconds = round(time.monotonic() - float(state["started_monotonic"]), 2)
                result_dir = _find_result_dir(repo_root, batch_id, name)
                summary = _load_run_summary(result_dir)
                metrics = _extract_summary_metrics(summary)

                common_fold_mean_alpha = ""
                common_fold_pct_positive = ""
                common_fold_count = ""
                if result_dir is not None:
                    fold_map = _load_fold_alphas(result_dir)
                    if fold_map:
                        common_fold_count = len(fold_map)

                row = {
                    "experiment": name,
                    "description": state["description"],
                    "returncode": int(return_code),
                    "started_at": state["started_at"],
                    "finished_at": finished_at,
                    "duration_seconds": duration_seconds,
                    "result_dir": str(result_dir).replace("\\", "/") if result_dir else "",
                    "summary_file": "",
                    "overall_mean_alpha": metrics.get("overall_mean_alpha", ""),
                    "overall_pct_positive_alpha": metrics.get("overall_pct_positive_alpha", ""),
                    "overall_strategy_sharpe": metrics.get("overall_strategy_sharpe", ""),
                    "overall_benchmark_sharpe": metrics.get("overall_benchmark_sharpe", ""),
                    "overall_strategy_return": metrics.get("overall_strategy_return", ""),
                    "overall_strategy_drawdown": metrics.get("overall_strategy_drawdown", ""),
                    "overall_strategy_fees": metrics.get("overall_strategy_fees", ""),
                    "n_folds": metrics.get("n_folds", ""),
                    "common_fold_mean_alpha": common_fold_mean_alpha,
                    "common_fold_pct_positive": common_fold_pct_positive,
                    "common_fold_count": common_fold_count,
                    "overrides": json.dumps(state["overrides"], ensure_ascii=True),
                }

                if result_dir is not None:
                    for candidate in [
                        result_dir / "strategy" / "final_summary.json",
                        result_dir / "strategy" / "backtest_summary.json",
                    ]:
                        if candidate.exists():
                            row["summary_file"] = str(candidate).replace("\\", "/")
                            break

                completed_rows.append(row)
                finished_names.append(name)
                print(
                    f"[DONE] {name} rc={int(return_code)} | overall_alpha={_fmt_pct(row.get('overall_mean_alpha'))} | sharpe={_fmt_float(row.get('overall_strategy_sharpe'))} | dir={row.get('result_dir', '') or 'n/a'}",
                    flush=True,
                )

            for name in finished_names:
                running.pop(name, None)

            if running:
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    print(
                        f"[STATUS] running={len(running)} completed={len(completed_rows)}/{len(experiments)} | active: {', '.join(sorted(running.keys()))}",
                        flush=True,
                    )
                    last_heartbeat = now
                time.sleep(poll_interval)
    finally:
        for state in running.values():
            try:
                state["log_handle"].close()
            except Exception:
                pass

    # Build the common-fold comparison now that all runs are complete.
    successful_rows = [row for row in completed_rows if int(row.get("returncode", 1)) == 0 and row.get("result_dir")]
    fold_maps: dict[str, dict[str, float]] = {}
    for row in successful_rows:
        result_dir = Path(str(row["result_dir"]))
        fold_map = _load_fold_alphas(result_dir)
        if fold_map:
            fold_maps[row["experiment"]] = fold_map

    common_folds: list[str] = []
    if fold_maps:
        fold_sets = [set(mapping.keys()) for mapping in fold_maps.values()]
        common_folds = sorted(set.intersection(*fold_sets), key=_parse_fold_label) if fold_sets else []

    if common_folds:
        import pandas as pd

        matrix = pd.DataFrame(index=common_folds)
        for experiment_name, fold_map in fold_maps.items():
            matrix[experiment_name] = [fold_map.get(fold) for fold in common_folds]

        matrix.index.name = "fold"
        matrix.to_csv(out_dir / "common_fold_alpha_matrix.csv", float_format="%.6f")

        ranking = pd.DataFrame(
            {
                "experiment": matrix.columns,
                "common_fold_mean_alpha": [float(matrix[col].mean()) for col in matrix.columns],
                "common_fold_pct_positive": [float((matrix[col] > 0).mean()) for col in matrix.columns],
                "common_fold_count": [int(matrix[col].count()) for col in matrix.columns],
            }
        ).sort_values("common_fold_mean_alpha", ascending=False)
        ranking.to_csv(out_dir / "common_fold_alpha_ranking.csv", index=False, float_format="%.6f")

        by_fold_rows = []
        for fold in common_folds:
            row = {"fold": fold}
            for experiment_name in matrix.columns:
                row[experiment_name] = matrix.loc[fold, experiment_name]
            by_fold_rows.append(row)
        pd.DataFrame(by_fold_rows).to_csv(out_dir / "common_fold_alpha_by_fold.csv", index=False, float_format="%.6f")

        matplotlib = __import__("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(15, 7))
        x = list(range(len(common_folds)))
        for experiment_name in matrix.columns:
            ax.plot(x, matrix[experiment_name].values * 100, marker="o", linewidth=2.0, label=experiment_name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(common_folds, rotation=45, ha="right")
        ax.set_ylabel("Alpha (%)")
        ax.set_title("Alpha por fold comun entre escenarios")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "common_fold_alpha_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        common_ranking = ranking.set_index("experiment")
        for row in completed_rows:
            exp_name = row["experiment"]
            if exp_name in common_ranking.index:
                row["common_fold_mean_alpha"] = float(common_ranking.loc[exp_name, "common_fold_mean_alpha"])
                row["common_fold_pct_positive"] = float(common_ranking.loc[exp_name, "common_fold_pct_positive"])
                row["common_fold_count"] = int(common_ranking.loc[exp_name, "common_fold_count"])
            else:
                row["common_fold_mean_alpha"] = ""
                row["common_fold_pct_positive"] = ""
                row["common_fold_count"] = ""
    else:
        for row in completed_rows:
            row.setdefault("common_fold_mean_alpha", "")
            row.setdefault("common_fold_pct_positive", "")
            row.setdefault("common_fold_count", "")

    # Rewrite the summary CSV after common-fold metrics are known.
    _write_comparison_artifacts(out_dir, completed_rows)

    failures = [row for row in completed_rows if int(row.get("returncode", 1)) != 0]
    ranked = sorted(
        [row for row in completed_rows if row.get("common_fold_mean_alpha") not in {"", None}],
        key=lambda row: float(row["common_fold_mean_alpha"]),
        reverse=True,
    )

    print(f"Parallel experiments completed: {len(completed_rows)} runs", flush=True)
    print(f"Summary CSV: {out_dir / 'summary.csv'}", flush=True)
    print(f"Scenario summary CSV: {out_dir / 'scenario_summary.csv'}", flush=True)
    print(f"Common-fold matrix: {out_dir / 'common_fold_alpha_matrix.csv'}", flush=True)
    print(f"Graph: {out_dir / 'common_fold_alpha_comparison.png'}", flush=True)
    print(f"Report: {out_dir / 'comparison_report.md'}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)

    if ranked:
        best = ranked[0]
        print(
            f"Best common-fold alpha: {best.get('experiment', '')} | alpha={_fmt_pct(best.get('common_fold_mean_alpha'))} | sharpe={_fmt_float(best.get('overall_strategy_sharpe'))}",
            flush=True,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
