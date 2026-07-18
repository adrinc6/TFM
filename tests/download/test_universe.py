from module.data.universe import (
    annual_membership_dates,
    is_recycled_ticker,
    members_at,
    normalize_ticker,
)


def test_membership_avoids_future_constituents() -> None:
    members_2000 = members_at("2000-01-03")

    assert "NVDA" not in members_2000
    assert "ENRNQ" in members_2000


def test_normalize_ticker_uses_yahoo_and_finnhub_format() -> None:
    assert normalize_ticker(" BRK.B ") == "BRK-B"


def test_recycled_tickers_are_excluded() -> None:
    assert is_recycled_ticker("CPQ", "2004-10-14")
    assert is_recycled_ticker("MOB", "2022-08-25")
    assert not is_recycled_ticker("AAPL", "1990-01-02")


def test_annual_membership_dates_use_last_snapshot_of_year() -> None:
    dates = annual_membership_dates()

    assert dates[0].year == 1996
    assert dates[-1].year == 2026
    assert len({date.year for date in dates}) == len(dates)
