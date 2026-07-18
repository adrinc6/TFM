from module.data.ingest.edgar import EdgarClient, _cik_from_atom, _extract_periodic_reports


def test_extract_periodic_reports_ignores_non_periodic_and_incomplete_rows() -> None:
    block = {
        "form": ["8-K", "10-Q", "10-K", "10-Q"],
        "reportDate": ["2000-01-01", "2000-03-31", "2000-12-31", ""],
        "filingDate": ["2000-01-02", "2000-05-01", "2001-02-01", "2000-08-01"],
        "accessionNumber": ["ignored", "quarter", "annual", "incomplete"],
    }

    assert _extract_periodic_reports(block) == [
        {
            "form": "10-Q",
            "period": "2000-03-31",
            "filed_date": "2000-05-01",
            "accession": "quarter",
        },
        {
            "form": "10-K",
            "period": "2000-12-31",
            "filed_date": "2001-02-01",
            "accession": "annual",
        },
    ]


def test_report_dates_reads_recent_and_historical_submission_blocks(tmp_path) -> None:
    client = EdgarClient("tests@example.invalid", tmp_path)

    def fake_get(url, cache_path):
        if url.endswith("CIK0000000001.json"):
            return {
                "filings": {
                    "recent": {
                        "form": ["10-Q"],
                        "reportDate": ["2016-03-31"],
                        "filingDate": ["2016-05-01"],
                        "accessionNumber": ["recent"],
                    },
                    "files": [{"name": "CIK0000000001-submissions-001.json"}],
                }
            }
        return {
            "form": ["10-K"],
            "reportDate": ["1999-12-31"],
            "filingDate": ["2000-02-01"],
            "accessionNumber": ["historical"],
        }

    client._get = fake_get

    reports = client.report_dates("AAPL", "0000000001")

    assert {(row["period"], row["filed_date"]) for row in reports} == {
        ("2016-03-31", "2016-05-01"),
        ("1999-12-31", "2000-02-01"),
    }


def test_cik_from_atom_handles_sec_namespaces() -> None:
    atom = """<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>
    <company-info><cik>0000034088</cik></company-info></feed>"""

    assert _cik_from_atom(atom) == "0000034088"
