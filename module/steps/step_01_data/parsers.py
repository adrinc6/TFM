"""Parsers for Finnhub raw payloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FinnhubSECParser:
    """
    Extrae metricas financieras estandarizadas de los RAW SEC filings de Finnhub.
    """

    BS_ALIASES = {
        "total_assets": [
            "Assets",
            "us-gaap_Assets",
        ],
        "current_assets": [
            "AssetsCurrent",
            "us-gaap_AssetsCurrent",
        ],
        "current_liabilities": [
            "LiabilitiesCurrent",
            "us-gaap_LiabilitiesCurrent",
        ],
        "total_liabilities": [
            "Liabilities",
            "us-gaap_Liabilities",
            "LiabilitiesAndStockholdersEquity",
        ],
        "total_equity": [
            "StockholdersEquity",
            "us-gaap_StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "long_term_debt": [
            "LongTermDebt",
            "us-gaap_LongTermDebt",
            "LongTermDebtNoncurrent",
            "us-gaap_LongTermDebtNoncurrent",
            "LongTermDebtAndCapitalLeaseObligations",
        ],
        "short_term_debt": [
            "ShortTermBorrowings",
            "us-gaap_ShortTermBorrowings",
            "DebtCurrent",
            "us-gaap_DebtCurrent",
            "CurrentPortionOfLongTermDebt",
        ],
        "cash": [
            "CashAndCashEquivalentsAtCarryingValue",
            "us-gaap_CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
            "us-gaap_CashCashEquivalentsAndShortTermInvestments",
            "CashAndCashEquivalents",
        ],
        "shares_outstanding": [
            "CommonStockSharesOutstanding",
            "us-gaap_CommonStockSharesOutstanding",
            "SharesOutstanding",
        ],
    }

    IC_ALIASES = {
        "revenue": [
            "Revenues",
            "us-gaap_Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "us-gaap_SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
        "gross_profit": [
            "GrossProfit",
            "us-gaap_GrossProfit",
        ],
        "operating_income": [
            "OperatingIncomeLoss",
            "us-gaap_OperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ],
        "net_income": [
            "NetIncomeLoss",
            "us-gaap_NetIncomeLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "interest_expense": [
            "InterestExpense",
            "us-gaap_InterestExpense",
            "InterestAndDebtExpense",
            "us-gaap_InterestAndDebtExpense",
        ],
        "income_tax": [
            "IncomeTaxExpenseBenefit",
            "us-gaap_IncomeTaxExpenseBenefit",
        ],
        "eps_diluted": [
            "EarningsPerShareDiluted",
            "us-gaap_EarningsPerShareDiluted",
            "EarningsPerShareBasicAndDiluted",
        ],
        "shares_diluted": [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingDiluted",
        ],
    }

    CF_ALIASES = {
        "operating_cash_flow": [
            "NetCashProvidedByUsedInOperatingActivities",
            "us-gaap_NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "capex": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpenditures",
            "AcquisitionOfPropertyPlantAndEquipment",
            "PurchasesOfPropertyAndEquipment",
        ],
        "depreciation": [
            "DepreciationDepletionAndAmortization",
            "us-gaap_DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "us-gaap_DepreciationAndAmortization",
        ],
    }

    def _extract_value(self, section: list, aliases: List[str]) -> Optional[float]:
        if not section:
            return None
        concept_map = {}
        for item in section:
            if isinstance(item, dict):
                concept = item.get("concept", "")
                value = item.get("value")
                if concept and value is not None:
                    concept_map[concept] = value
        for alias in aliases:
            val = concept_map.get(alias)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    def parse_filing(self, filing: dict) -> dict:
        result = {}
        bs = filing.get("report", {}).get("bs", [])
        ic = filing.get("report", {}).get("ic", [])
        cf = filing.get("report", {}).get("cf", [])

        for col, aliases in self.BS_ALIASES.items():
            result[col] = self._extract_value(bs, aliases)

        for col, aliases in self.IC_ALIASES.items():
            result[col] = self._extract_value(ic, aliases)

        for col, aliases in self.CF_ALIASES.items():
            result[col] = self._extract_value(cf, aliases)

        ocf = result.get("operating_cash_flow")
        capex = result.get("capex")
        if ocf is not None and capex is not None:
            result["fcf"] = ocf - abs(capex)
        else:
            result["fcf"] = None

        lt_debt = result.get("long_term_debt") or 0.0
        st_debt = result.get("short_term_debt") or 0.0
        result["total_debt"] = lt_debt + st_debt

        form = filing.get("form", "")
        quarter = filing.get("quarter")
        result["report_type"] = "10-K" if (form == "10-K" or quarter == 0) else "10-Q"
        result["fiscal_quarter"] = int(quarter) if quarter is not None else (0 if form == "10-K" else None)

        return result

    def parse_financials_json(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"[SECParser] Error leyendo {path}: {e}")
            return pd.DataFrame()

        filings = data.get("data", [])
        if not filings:
            return pd.DataFrame()

        records = []
        for filing in filings:
            date_str = filing.get("endDate") or filing.get("startDate")
            if not date_str:
                continue
            try:
                date = pd.Timestamp(date_str)
            except Exception:
                continue

            parsed = self.parse_filing(filing)
            parsed["report_date"] = date
            records.append(parsed)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("report_date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df


class BasicFinancialsParser:
    """
    Extrae series temporales y metricas point-in-time de basic_financials.json.
    """

    SERIES_METRICS = {
        "bf_current_ratio": ("currentRatioQuarterly", "currentRatio"),
        "bf_revenue_growth": ("revenueGrowthQuarterly", "revenueGrowth"),
        "bf_eps_growth": ("epsGrowthQuarterlyYoy", "epsGrowth"),
        "bf_roe": ("roeRfy", "roeTTM"),
        "bf_roa": ("roaRfy", "roaTTM"),
        "bf_gross_margin": ("grossMarginTTM", "grossMarginTTM"),
        "bf_net_margin": ("netProfitMarginTTM", "netProfitMarginTTM"),
        "bf_debt_equity": ("totalDebt/totalEquityQuarterly", "totalDebt/totalEquityAnnual"),
        "bf_ev_ebitda": ("evToEbitda", "evToEbitda"),
        "bf_pe": ("peTTM", "peTTM"),
        "bf_pb": ("pbQuarterly", "pbAnnual"),
        "bf_ps": ("psTTM", "psTTM"),
        "bf_fcf_yield": ("fcfYieldTTM", "fcfYieldTTM"),
        "bf_revenue_per_share": ("revenuePerShareTTM", "revenuePerShareTTM"),
        "bf_book_value_ps": ("bookValuePerShareQuarterly", "bookValuePerShareAnnual"),
        "bf_long_term_debt_equity": ("longTermDebt/equityQuarterly", "longTermDebt/equityAnnual"),
    }

    POINT_IN_TIME = {
        "bf_beta": "beta",
        "bf_52w_high": "52WeekHigh",
        "bf_52w_low": "52WeekLow",
        "bf_10d_avg_vol": "10DayAverageTradingVolume",
        "bf_52w_avg_vol": "52WeekAverageTradingVolume",
        "bf_market_cap": "marketCapitalization",
        "bf_pe_ttm": "peTTM",
        "bf_pb_annual": "pbAnnual",
        "bf_ps_ttm": "psTTM",
        "bf_roic_ttm": "roicTTM",
        "bf_interest_coverage": "interestCoverageQuarterly",
    }

    def parse(self, path: Path) -> tuple[pd.DataFrame, dict]:
        if not path.exists():
            return pd.DataFrame(), {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"[BasicFinancials] Error leyendo {path}: {e}")
            return pd.DataFrame(), {}

        metric = data.get("metric", {})
        series_annual = data.get("series", {}).get("annual", {})
        series_quarterly = data.get("series", {}).get("quarterly", {})

        records_by_date: Dict[str, dict] = {}

        def _add_series(api_key: str, feat_name: str, series_dict: dict) -> None:
            for item in series_dict.get(api_key, []):
                period = item.get("period")
                value = item.get("v")
                if period and value is not None:
                    records_by_date.setdefault(period, {})
                    if feat_name not in records_by_date[period]:
                        try:
                            records_by_date[period][feat_name] = float(value)
                        except (ValueError, TypeError):
                            pass

        for feat_name, (q_key, a_key) in self.SERIES_METRICS.items():
            _add_series(q_key, feat_name, series_quarterly)
            _add_series(a_key, feat_name, series_annual)

        series_df = pd.DataFrame()
        if records_by_date:
            series_df = pd.DataFrame.from_dict(records_by_date, orient="index")
            series_df.index = pd.to_datetime(series_df.index)
            series_df = series_df.sort_index()
            series_df = series_df[~series_df.index.duplicated(keep="last")]

        pit = {}
        for feat_name, api_key in self.POINT_IN_TIME.items():
            val = metric.get(api_key)
            if val is not None:
                try:
                    pit[feat_name] = float(val)
                except (ValueError, TypeError):
                    pass

        return series_df, pit


class EPSSurprisesParser:
    """Extrae series temporales de sorpresas de EPS desde eps_surprises.json."""

    def parse(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return pd.DataFrame()

        items = data.get("data", [])
        if not items:
            return pd.DataFrame()

        records = []
        for item in items:
            period = item.get("period")
            if not period:
                continue
            try:
                dt = pd.Timestamp(period)
            except Exception:
                continue
            actual = item.get("actual")
            estimate = item.get("estimate")
            surprise = item.get("surprisePercent")
            records.append({
                "date": dt,
                "eps_actual": float(actual) if actual is not None else np.nan,
                "eps_estimate": float(estimate) if estimate is not None else np.nan,
                "eps_surprise_pct": float(surprise) if surprise is not None else np.nan,
                "eps_beat": int(actual > estimate)
                if (actual is not None and estimate is not None) else np.nan,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]


class RecommendationParser:
    """Extrae senales de consenso de analistas desde recommendation_trends.json."""

    def parse(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return pd.DataFrame()

        items = data.get("data", [])
        if not items:
            return pd.DataFrame()

        records = []
        for item in items:
            period = item.get("period")
            if not period:
                continue
            try:
                dt = pd.Timestamp(period)
            except Exception:
                continue

            strong_buy = float(item.get("strongBuy", 0) or 0)
            buy = float(item.get("buy", 0) or 0)
            hold = float(item.get("hold", 0) or 0)
            sell = float(item.get("sell", 0) or 0)
            strong_sell = float(item.get("strongSell", 0) or 0)
            total = strong_buy + buy + hold + sell + strong_sell

            if total == 0:
                continue

            bullish = strong_buy + buy
            bearish = sell + strong_sell

            consensus = (
                2 * strong_buy + 1 * buy + 0 * hold - 1 * sell - 2 * strong_sell
            ) / total

            records.append({
                "date": dt,
                "analyst_total": total,
                "analyst_buy_ratio": bullish / total,
                "analyst_bearish_score": bearish / total,
                "analyst_strong_buy_pct": strong_buy / total,
                "analyst_dispersion": (strong_buy + strong_sell) / total,
                "analyst_consensus": consensus,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]


class InsiderSentimentParser:
    """Extrae MSPR mensual desde insider_sentiment.json."""

    def parse(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return pd.DataFrame()

        items = data.get("data", [])
        if not items:
            return pd.DataFrame()

        records = []
        for item in items:
            year = item.get("year")
            month = item.get("month")
            mspr = item.get("mspr")
            change = item.get("change")
            if year is None or month is None:
                continue
            try:
                dt = pd.Timestamp(f"{int(year)}-{int(month):02d}-01")
            except Exception:
                continue
            records.append({
                "date": dt,
                "mspr": float(mspr) if mspr is not None else np.nan,
                "insider_net_buy": float(change) if change is not None else np.nan,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]


class InsiderTransactionsParser:
    """Extrae transacciones insider desde insider_transactions.json."""

    BUY_CODES = {"P"}
    SELL_CODES = {"S", "S+"}

    def parse(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return pd.DataFrame()

        items = data.get("data", [])
        if not items:
            return pd.DataFrame()

        records = []
        for item in items:
            date_str = item.get("transactionDate") or item.get("filingDate")
            if not date_str:
                continue
            try:
                dt = pd.Timestamp(date_str)
            except Exception:
                continue

            code = str(item.get("transactionCode", "")).strip()
            change = item.get("change", 0) or 0
            shares = abs(float(change)) if change else 0.0

            is_buy = code in self.BUY_CODES or change > 0
            is_sell = code in self.SELL_CODES or change < 0

            records.append({
                "date": dt,
                "name": item.get("name", ""),
                "transaction_code": code,
                "shares": shares,
                "is_buy": int(is_buy),
                "is_sell": int(is_sell),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df
