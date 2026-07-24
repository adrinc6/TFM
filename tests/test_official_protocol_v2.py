from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import environment
import module.runs.official_protocol as official_module
from environment import Settings
from module.evaluation.portfolio import PortfolioState, VintageBook, active_fraction, policy_target
from module.evaluation.signal_diagnostics import rank_tail_diagnostics, signal_health_path
from module.modeling.meta import _constrain_stacked_weights, _meta_history_window
from module.modeling.targets import normalize_target_columns
from module.runs.official_protocol import (
    _fast_random_cagrs, _portfolio_record, official_preflight, run_official_protocol,
)
from module.runs.results_store import ResultsStore
from module.scenarios.variables import (
    OFFICIAL_SIGNAL_CHALLENGERS, official_evaluation_budget, validate_official_budget,
)


def test_official_protocol_has_exactly_48_evaluations() -> None:
    budget = official_evaluation_budget()

    assert budget["total"] == 48
    assert sum(budget["groups"].values()) == 48
    assert len(OFFICIAL_SIGNAL_CHALLENGERS) == 12
    assert budget["max_evaluations"] == 50
    assert budget["estimated_expensive_fits"] == 10


def test_budget_of_51_aborts_before_any_run_is_created() -> None:
    with pytest.raises(ValueError, match="máximo"):
        validate_official_budget({"synthetic": 51})
    with pytest.raises(ValueError, match="walk-forwards"):
        validate_official_budget({"synthetic": 48}, estimated_expensive_fits=11)


def test_official_preflight_is_read_only_and_reports_operational_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in (
        "finnhub_metrics.parquet", "prices.parquet", "profiles.parquet",
        "report_dates.parquet",
    ):
        (raw / name).write_bytes(b"available")
    monkeypatch.setattr(environment, "DEV_RAW_DIR", raw)
    store = ResultsStore(tmp_path / "results")

    preflight = official_preflight(Settings(run_scope="dev"), store)

    assert preflight["evaluation_budget"]["total"] == 48
    assert preflight["fit_budget"]["estimated_new"] == 10
    assert preflight["storage"]["estimated_incremental_bytes"] < 5 * 1024**3
    assert not any(store.studies_root.iterdir())
    assert not any(store.runs_root.iterdir())


def test_official_protocol_records_failed_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ResultsStore(tmp_path / "results")
    created: dict[str, str] = {}

    def fail_after_creation(*_args, **_kwargs):
        study_id, _ = store.create_study({"name": "fallo-controlado", "kind": "optimization"})
        created["study_id"] = study_id
        raise RuntimeError("fallo sintético")

    monkeypatch.setattr(official_module, "_run_official_protocol", fail_after_creation)
    with pytest.raises(RuntimeError, match="fallo sintético"):
        run_official_protocol(Settings(run_scope="dev"), store, name="fallo-controlado")

    manifest = json.loads(
        (store.studies_root / created["study_id"] / "study_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert manifest["current_phase"] == "failed"


def test_portfolio_selector_metrics_exclude_known_stress_years(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    artifacts = store.runs_root / "candidate" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "backtest_summary.json").write_text(
        json.dumps({
            "cagr_portfolio": 0.10, "cagr_difference": 0.01, "beat_rate": 0.50,
            "information_ratio": -99.0, "mean_annual_alpha": -99.0,
            "annualized_turnover": 99.0, "mean_active_fraction": 0.10,
        }),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "snapshot_date": "2024-01-31", "excess_return": 0.01,
            "gross_return": 0.02, "benchmark_return": 0.01,
            "turnover_pct": 0.10, "active_fraction": 0.60,
        },
        {
            "snapshot_date": "2024-02-29", "excess_return": 0.03,
            "gross_return": 0.04, "benchmark_return": 0.01,
            "turnover_pct": 0.20, "active_fraction": 0.80,
        },
        {
            "snapshot_date": "2025-01-31", "excess_return": -0.90,
            "gross_return": -0.80, "benchmark_return": 0.10,
            "turnover_pct": 9.00, "active_fraction": 0.00,
        },
    ]).to_parquet(artifacts / "equity.parquet", index=False)

    record = _portfolio_record(store, "candidate", "candidate")

    assert record["selection_until_year"] == 2024
    assert record["mean_annual_alpha"] == pytest.approx(0.24)
    assert record["annualized_turnover"] == pytest.approx(1.8)
    assert record["mean_active_fraction"] == pytest.approx(0.70)


def test_rolling_meta_uses_exact_quarter_window() -> None:
    history = pd.DataFrame({
        "snapshot_date": pd.date_range("2019-03-31", periods=12, freq="QE"),
        "value": range(12),
    })
    settings = replace(
        Settings(), meta_type="stacked_oos", meta_history_mode="rolling",
        meta_history_quarters=8,
    )

    selected = _meta_history_window(history, settings)

    assert len(selected["snapshot_date"].unique()) == 8
    assert selected["value"].tolist() == list(range(4, 12))


def test_meta_cap_shrinkage_is_normalized_and_bounded() -> None:
    settings = replace(Settings(), meta_weight_cap=0.50, meta_equal_shrinkage=0.50)
    weights = _constrain_stacked_weights(
        {"quality": 0.90, "value": 0.05, "growth": 0.05}, settings,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.50 + 1e-12
    assert all(weight >= 0 for weight in weights.values())


def test_rank_tail_metrics_match_monotonic_synthetic_signal() -> None:
    tickers = [f"T{index:02d}" for index in range(20)]
    scores = pd.DataFrame({
        "ticker": tickers,
        "snapshot_date": "2020-03-31",
        "meta_rank": [(index + 1) / 20 for index in range(20)],
        "meta_score": [(index + 1) / 20 for index in range(20)],
    })
    targets = pd.DataFrame({
        "ticker": tickers,
        "snapshot_date": "2020-03-31",
        "label_end_date": "2021-03-31",
        "target_available": True,
        "forward_excess_return": [(index + 1) / 100 for index in range(20)],
    })

    row = rank_tail_diagnostics(scores, targets).iloc[0]

    assert row["rank_ic"] == pytest.approx(1.0)
    assert row["top_decile_minus_universe"] > 0
    assert row["top_minus_bottom"] > 0
    assert row["top_n"] == 10


def test_signal_health_never_uses_unclosed_labels() -> None:
    history = pd.DataFrame({
        "prediction_date": pd.date_range("2017-03-31", periods=9, freq="QE").astype(str),
        "label_end_date": pd.date_range("2018-03-31", periods=9, freq="QE").astype(str),
        "rank_ic": [0.10] * 9,
        "top_decile_minus_universe": [0.10] * 9,
    })

    before_close = signal_health_path(history, ["2019-12-30"]).iloc[0]
    after_close = signal_health_path(history, ["2019-12-31"]).iloc[0]

    assert before_close["closed_cohorts"] == 7
    assert before_close["continuous_active_fraction"] == pytest.approx(0.5)
    assert after_close["closed_cohorts"] == 8
    assert after_close["continuous_active_fraction"] > 0.5


def test_four_vintages_of_three_positions_target_twelve_and_do_not_rotate_monthly() -> None:
    settings = replace(
        Settings(), portfolio_policy="staggered_vintages", vintage_count=4,
        holding_months=12, target_size=12, sizing_mode="equal",
    )
    state = PortfolioState.empty()
    book = VintageBook()
    last_target = None
    last_rows = []
    for quarter, date in enumerate(("2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31")):
        scores = pd.DataFrame({
            "ticker": [f"Q{quarter}T{index}" for index in range(6)],
            "snapshot_date": date,
            "meta_rank": [1 - index / 10 for index in range(6)],
            "meta_score": [1 - index / 10 for index in range(6)],
            "is_quarterly": True,
        })
        last_target, _, last_rows = policy_target(state, scores, settings, book)

    assert len(last_target or {}) == 12
    assert sum((last_target or {}).values()) == pytest.approx(1.0)
    assert {row["scheduled_exit_date"] for row in last_rows} == {
        "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    }

    monthly = scores.assign(snapshot_date="2021-01-31", is_quarterly=False)
    unchanged, _, _ = policy_target(state, monthly, settings, book)
    assert unchanged is None


def test_overlay_modes_and_historical_target_reader() -> None:
    scores = pd.DataFrame({
        "continuous_active_fraction": [0.35],
        "binary_active_fraction": [1.0],
    })
    assert active_fraction(scores, replace(Settings(), active_overlay_mode="continuous")) == 0.35
    assert active_fraction(scores, replace(Settings(), active_overlay_mode="binary")) == 1.0
    assert active_fraction(
        scores, replace(Settings(), active_overlay_mode="fixed", fixed_active_fraction=0.50),
    ) == 0.50

    legacy = pd.DataFrame({"forward_excess_return_3m": [0.1]})
    neutral = normalize_target_columns(legacy)
    assert neutral["forward_excess_return"].tolist() == [0.1]

    modern = pd.DataFrame({"forward_excess_return": [0.2]})
    compatible = normalize_target_columns(modern)
    assert compatible["forward_excess_return"].tolist() == [0.2]
    assert "forward_excess_return_3m" not in compatible


def test_portfolio_state_does_not_apply_target_without_order() -> None:
    state = PortfolioState(holdings={"AAA": 0.6, "BBB": 0.4})
    updated = state.apply([], {}, target_weights={"AAA": 0.5, "BBB": 0.5})

    assert updated.holdings == {"AAA": 0.6, "BBB": 0.4}


def test_random_portfolio_engine_uses_pit_snapshots_and_returns_two_nulls() -> None:
    dates = ("2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31")
    score_rows = []
    price_rows = []
    for date_index, date in enumerate(dates):
        # FUTURE no existe antes del último snapshot: el motor solo puede seleccionarlo entonces.
        tickers = [f"T{index}" for index in range(10)]
        if date_index == len(dates) - 1:
            tickers.append("FUTURE")
        for index, ticker in enumerate(tickers):
            score_rows.append({
                "ticker": ticker, "snapshot_date": date,
                "meta_rank": (index + 1) / len(tickers),
                "risk_rank": (len(tickers) - index) / len(tickers),
                "is_quarterly": True,
                "continuous_active_fraction": 1.0,
                "binary_active_fraction": 1.0,
            })
            price_rows.append({
                "ticker": ticker, "snapshot_date": date,
                "price": 100.0 + date_index + index,
            })
    benchmark = pd.DataFrame({
        "snapshot_date": list(dates),
        "price": [100.0, 101.0, 102.0, 103.0],
    })
    settings = replace(
        Settings(), portfolio_policy="staggered_vintages", vintage_count=4,
        holding_months=12, target_size=8, sizing_mode="equal",
        active_overlay_mode="continuous",
    )

    unconditional, matched = _fast_random_cagrs(
        pd.DataFrame(score_rows), pd.DataFrame(price_rows), benchmark, settings,
        simulations=12, seed=42,
    )

    assert unconditional.shape == matched.shape == (12,)
    assert pd.Series(unconditional).notna().all()
    assert pd.Series(matched).notna().all()


def test_source_and_documentation_are_utf8_without_mojibake() -> None:
    roots = [Path("module"), Path("app"), Path("docs"), Path("README.md")]
    paths = [
        path for root in roots
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".py", ".js", ".json", ".md", ".html"}
    ]
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in ("Ã", "Â", "�")):
            offenders.append(str(path))

    assert offenders == []
