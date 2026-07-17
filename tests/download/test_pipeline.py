import json

import pandas as pd
import pytest

import environment
import main
from environment import Settings
from module.ingest import pipeline


class FakeYahooClient:
    def ohlcv(self, ticker, start, end):
        if ticker == "CPQ":
            return {"data": [{"date": "2004-10-14", "close": 1.0}]}
        return {"data": [{"date": "2000-01-03", "close": 10.0}]}


class FakeFinnhubClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def company_profile2(self, ticker):
        return {"name": ticker}

    def basic_financials(self, ticker):
        return {
            "series": {
                "quarterly": {
                    "roeTTM": [{"period": "1999-12-31", "v": 0.2}],
                }
            }
        }

    def company_news(self, ticker, start, end):
        return []


class FakeEdgarClient:
    def __init__(self, *args, **kwargs):
        pass

    def ticker_to_cik(self):
        return {"AAPL": "0000320193", "XOM": "0002115436"}

    def lookup_cik(self, ticker):
        return "0000034088" if ticker == "XOM" else None

    def report_dates(self, ticker, cik):
        if ticker == "XOM" and cik == "0002115436":
            return []
        return [
            {"form": "10-K", "period": "1999-12-31", "filed_date": "2000-02-15"},
            {"form": "10-K", "period": "1999-12-31", "filed_date": "2000-02-01"},
        ]


def test_pipeline_isolates_dev_outputs_and_excludes_recycled_ticker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(environment, "DEV_TICKERS", ["AAPL", "CPQ", "XOM"])
    monkeypatch.setattr(environment, "DEV_RAW_DIR", tmp_path / "raw" / "dev")
    monkeypatch.setattr(pipeline, "RAW_JSON_DIR", tmp_path / "raw" / "json")
    monkeypatch.setattr(pipeline, "FinnhubClient", FakeFinnhubClient)
    monkeypatch.setattr(pipeline, "YahooClient", FakeYahooClient)
    monkeypatch.setattr(pipeline, "EdgarClient", FakeEdgarClient)

    pipeline.download_raw_data(Settings(run_scope="dev"))

    output_dir = tmp_path / "raw" / "dev"
    reports = pd.read_parquet(output_dir / "report_dates.parquet")
    failures = pd.read_csv(output_dir / "download_failures.csv")
    coverage = json.loads((output_dir / "universe_coverage.json").read_text(encoding="utf-8"))

    assert reports.loc[reports["ticker"] == "AAPL"].to_dict("records") == [
        {
            "ticker": "AAPL",
            "cik": "0000320193",
            "form": "10-K",
            "period": "1999-12-31",
            "filed_date": "2000-02-01",
        }
    ]
    assert reports.loc[reports["ticker"] == "XOM", "cik"].tolist() == ["0000034088"]
    assert "CPQ" in set(failures.loc[failures["reason"].str.startswith("recycled_ticker"), "ticker"])
    assert "SPY" not in set(failures["ticker"])
    raw_prices = pd.read_parquet(output_dir / "prices.parquet")
    assert "SPY" in set(raw_prices["ticker"])
    assert coverage["run_scope"] == "dev"
    assert not coverage["representative"]


def test_full_coverage_contains_annual_schema() -> None:
    coverage = pipeline._universe_coverage(Settings(run_scope="full"), {}, set())

    assert coverage["representative"]
    assert coverage["years"]
    assert {"year", "sp500_members", "panel_eligible_tickers", "coverage_pct", "exclusions"} <= set(
        coverage["years"][0]
    )


def test_stage_selector_runs_implemented_stages() -> None:
    assert main.stages_for_run("download") == ("download",)
    assert main.stages_for_run("full") == (
        "download", "dataset", "features", "agents", "backtest", "report",
    )
    assert main.stages_for_run("features") == ("features",)
    assert main.stages_for_run("backtest") == ("backtest",)
    assert main.stages_for_run("report") == ("report",)
    assert main.stages_for_run("experiments") == ("experiments",)
