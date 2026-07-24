from module.data.universe import is_recycled_ticker, members_at


def test_membership_avoids_future_constituents() -> None:
    members_2000 = members_at("2000-01-03")

    assert "NVDA" not in members_2000
    assert "ENRNQ" in members_2000

def test_recycled_tickers_are_excluded() -> None:
    assert is_recycled_ticker("CPQ", "2004-10-14")
    assert is_recycled_ticker("MOB", "2022-08-25")
    assert not is_recycled_ticker("AAPL", "1990-01-02")
