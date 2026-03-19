# =============================================================================
# module/finnhub_processor.py — Parser y consolidador de datos Finnhub
# =============================================================================
"""
Transforma los JSONs crudos descargados por fetcher_finnhub.py en DataFrames
normalizados listos para el pipeline ML.

Responsabilidades:
  1. FinnhubSECParser    : extrae líneas de balance/income/cashflow de los RAW
                           SEC filings (financials_reported_*.json) usando alias
                           XBRL robustos.
  2. BasicFinancialsParser: extrae series temporales de ratios de basic_financials.json
  3. FinnhubConsolidator : consolida todas las fuentes por ticker, calcula ratios
                           adicionales y guarda CSVs en data_finnhub/consolidated/
  4. build_companies_df  : construye el DataFrame de sector/industry desde los
                           profiles descargados (reemplaza companies.csv).
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# =============================================================================
# 1. PARSER SEC FILINGS (financials_reported_annual / quarterly)
# =============================================================================

class FinnhubSECParser:
    """
    Extrae métricas financieras estandarizadas de los RAW SEC filings de Finnhub.

    Los filings 10-K/10-Q contienen secciones 'bs', 'ic', 'cf' con nombres
    XBRL no estandarizados. Este parser usa múltiples alias para cada concepto
    y toma el primero no-nulo encontrado.
    """

    # ── Alias XBRL por concepto ───────────────────────────────────────────────
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
            "LiabilitiesAndStockholdersEquity",  # a veces fusionado
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
        """Busca la primera coincidencia de alias en una sección de filing."""
        if not section:
            return None
        # Construir dict de concepto → valor para búsqueda rápida
        concept_map = {}
        for item in section:
            if isinstance(item, dict):
                concept = item.get("concept", "")
                value   = item.get("value")
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
        """
        Extrae métricas de un filing individual (anual o trimestral).
        Retorna dict con columnas estandarizadas.
        """
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

        # Derivados directos
        ocf   = result.get("operating_cash_flow")
        capex = result.get("capex")
        if ocf is not None and capex is not None:
            # CapEx suele reportarse como positivo en Finnhub (salida de caja)
            result["fcf"] = ocf - abs(capex)
        else:
            result["fcf"] = None

        lt_debt = result.get("long_term_debt") or 0.0
        st_debt = result.get("short_term_debt") or 0.0
        result["total_debt"] = lt_debt + st_debt

        return result

    def parse_financials_json(self, path: Path) -> pd.DataFrame:
        """
        Lee un fichero financials_reported_*.json y devuelve DataFrame
        indexado por fecha de reporte (report_date).
        """
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


# =============================================================================
# 2. PARSER DE BASIC_FINANCIALS (ratios + series temporales)
# =============================================================================

class BasicFinancialsParser:
    """
    Extrae series temporales anuales y métricas point-in-time de basic_financials.json.

    Las series anuales están en: data["series"]["annual"][metric_name] = [{period, v}, ...]
    Las métricas point-in-time están en: data["metric"][metric_name]
    """

    # Series anuales que queremos extraer
    ANNUAL_SERIES = {
        "bf_current_ratio":     "currentRatio",
        "bf_revenue_growth":    "revenueGrowth",
        "bf_eps_growth":        "epsGrowth",
        "bf_roe":               "roeTTM",
        "bf_roa":               "roaTTM",
        "bf_gross_margin":      "grossMarginTTM",
        "bf_net_margin":        "netProfitMarginTTM",
        "bf_debt_equity":       "totalDebt/totalEquityAnnual",
        "bf_ev_ebitda":         "evToEbitda",
        "bf_pe":                "peTTM",
        "bf_pb":                "pbAnnual",
        "bf_ps":                "psTTM",
        "bf_fcf_yield":         "fcfYieldTTM",
        "bf_revenue_per_share": "revenuePerShareTTM",
        "bf_book_value_ps":     "bookValuePerShareQuarterly",
        "bf_long_term_debt_equity": "longTermDebt/equityAnnual",
    }

    # Métricas point-in-time (snapshot actual)
    POINT_IN_TIME = {
        "bf_beta":          "beta",
        "bf_52w_high":      "52WeekHigh",
        "bf_52w_low":       "52WeekLow",
        "bf_10d_avg_vol":   "10DayAverageTradingVolume",
        "bf_52w_avg_vol":   "52WeekAverageTradingVolume",
        "bf_market_cap":    "marketCapitalization",
        "bf_pe_ttm":        "peTTM",
        "bf_pb_annual":     "pbAnnual",
        "bf_ps_ttm":        "psTTM",
        "bf_roic_ttm":      "roicTTM",
        "bf_interest_coverage": "currentRatioQuarterly",
    }

    def parse(self, path: Path) -> tuple:
        """
        Retorna:
          - series_df: DataFrame indexado por fecha con series anuales
          - point_in_time: dict con métricas snapshot actuales
        """
        if not path.exists():
            return pd.DataFrame(), {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"[BasicFinancials] Error leyendo {path}: {e}")
            return pd.DataFrame(), {}

        metric = data.get("metric", {})
        series = data.get("series", {}).get("annual", {})

        # Extraer series anuales
        records_by_date: Dict[str, dict] = {}
        for feat_name, api_key in self.ANNUAL_SERIES.items():
            serie = series.get(api_key, [])
            for item in serie:
                period = item.get("period")
                value  = item.get("v")
                if period and value is not None:
                    if period not in records_by_date:
                        records_by_date[period] = {}
                    try:
                        records_by_date[period][feat_name] = float(value)
                    except (ValueError, TypeError):
                        pass

        series_df = pd.DataFrame()
        if records_by_date:
            series_df = pd.DataFrame.from_dict(records_by_date, orient="index")
            series_df.index = pd.to_datetime(series_df.index)
            series_df = series_df.sort_index()
            series_df = series_df[~series_df.index.duplicated(keep="last")]

        # Extraer point-in-time
        pit = {}
        for feat_name, api_key in self.POINT_IN_TIME.items():
            val = metric.get(api_key)
            if val is not None:
                try:
                    pit[feat_name] = float(val)
                except (ValueError, TypeError):
                    pass

        return series_df, pit


# =============================================================================
# 3. FINNHUB CONSOLIDATOR
# =============================================================================

class FinnhubConsolidator:
    """
    Consolida todas las fuentes Finnhub por ticker en un CSV estandarizado.

    Fuentes (por orden de prioridad):
      1. financials_reported_quarterly.json (datos SEC más detallados)
      2. financials_reported_annual.json   (fallback / completar huecos)
      3. basic_financials.json            (series temporales de ratios)

    Output: data_finnhub/consolidated/{TICKER}.csv
    """

    def __init__(self, finnhub_data_dir: str = "data_finnhub"):
        self.data_dir    = Path(finnhub_data_dir)
        self.output_dir  = self.data_dir / "consolidated"
        self.sec_parser  = FinnhubSECParser()
        self.bf_parser   = BasicFinancialsParser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def consolidate_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Consolida todas las fuentes para un ticker.
        Retorna DataFrame indexado por report_date.
        """
        ticker_dir = self.data_dir / ticker

        # 1. SEC Filings trimestrales (fuente primaria)
        df_q = self.sec_parser.parse_financials_json(
            ticker_dir / "financials_reported_quarterly.json"
        )

        # 2. SEC Filings anuales (complementar huecos)
        df_a = self.sec_parser.parse_financials_json(
            ticker_dir / "financials_reported_annual.json"
        )

        # 3. Basic financials series temporales
        bf_series, bf_pit = self.bf_parser.parse(
            ticker_dir / "basic_financials.json"
        )

        # Combinar: trimestrales tienen prioridad
        if df_q.empty and df_a.empty:
            log.warning(f"[{ticker}] Sin datos SEC — skip")
            return pd.DataFrame()

        if not df_q.empty:
            df = df_q.copy()
        else:
            df = df_a.copy()

        # Añadir filas anuales que no están en trimestrales
        if not df_q.empty and not df_a.empty:
            for dt in df_a.index:
                if dt not in df.index:
                    df.loc[dt] = df_a.loc[dt]
            df = df.sort_index()

        # Join con series de basic_financials (merge por fecha más cercana)
        if not bf_series.empty:
            df = pd.merge_asof(
                df.reset_index(),
                bf_series.reset_index().rename(columns={"index": "report_date"}),
                on="report_date",
                direction="backward",
                tolerance=pd.Timedelta("400D"),
            ).set_index("report_date")

        # Calcular ratios adicionales
        df = self._calculate_ratios(df)

        # Añadir point-in-time de basic_financials como columnas constantes
        for k, v in bf_pit.items():
            if k not in df.columns:
                df[k] = v

        # Limpiar infinitos
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ratios financieros derivados de los datos base."""
        df = df.copy()

        # Asegurar total_debt
        if "total_debt" not in df.columns:
            lt = df.get("long_term_debt", pd.Series(0, index=df.index)).fillna(0)
            st = df.get("short_term_debt", pd.Series(0, index=df.index)).fillna(0)
            df["total_debt"] = lt + st

        # EBITDA
        if "operating_income" in df.columns and "depreciation" in df.columns:
            df["ebitda"] = df["operating_income"] + df["depreciation"].fillna(0)

        # EPS calculado (si no viene del SEC)
        if "eps_diluted" not in df.columns or df["eps_diluted"].isna().all():
            if "net_income" in df.columns and "shares_diluted" in df.columns:
                df["eps_diluted"] = (
                    df["net_income"] / df["shares_diluted"].replace(0, np.nan)
                )
        # Estandarizar nombre eps
        if "eps_diluted" in df.columns and "eps" not in df.columns:
            df["eps"] = df["eps_diluted"]

        # shares estandarizado
        if "shares_diluted" not in df.columns and "shares_outstanding" in df.columns:
            df["shares_diluted"] = df["shares_outstanding"]

        # Ratios de rentabilidad
        if "net_income" in df.columns and "total_equity" in df.columns:
            df["roe"] = df["net_income"] / df["total_equity"].replace(0, np.nan)
        if "net_income" in df.columns and "total_assets" in df.columns:
            df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
        if "net_income" in df.columns and "revenue" in df.columns:
            df["net_margin"] = df["net_income"] / df["revenue"].replace(0, np.nan)
        if "gross_profit" in df.columns and "revenue" in df.columns:
            df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, np.nan)
        if "operating_income" in df.columns and "revenue" in df.columns:
            df["operating_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)
        if "operating_income" in df.columns and "total_assets" in df.columns:
            df["roi"] = df["operating_income"] / df["total_assets"].replace(0, np.nan)

        # ROIC: Operating Income / (Equity + Debt)
        if "operating_income" in df.columns and "total_equity" in df.columns:
            invested = df["total_equity"].fillna(0) + df["total_debt"].fillna(0)
            df["roic"] = df["operating_income"] / invested.replace(0, np.nan)

        # Ratios de liquidez / apalancamiento
        if "total_equity" in df.columns:
            df["debt_equity"] = df["total_debt"] / df["total_equity"].replace(0, np.nan)
        if "current_assets" in df.columns and "current_liabilities" in df.columns:
            df["current_ratio"] = (
                df["current_assets"] / df["current_liabilities"].replace(0, np.nan)
            )

        # Interest coverage
        if "operating_income" in df.columns and "interest_expense" in df.columns:
            ie = df["interest_expense"].replace(0, np.nan).abs()
            df["interest_coverage"] = df["operating_income"] / ie

        # FCF margin
        if "fcf" in df.columns and "revenue" in df.columns:
            df["fcf_margin"] = df["fcf"] / df["revenue"].replace(0, np.nan)

        # Debt to EBITDA
        if "ebitda" in df.columns:
            df["debt_to_ebitda"] = df["total_debt"] / df["ebitda"].replace(0, np.nan)
            df["ebitda_margin"] = df["ebitda"] / df.get("revenue", pd.Series(np.nan, index=df.index)).replace(0, np.nan)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def process_all_tickers(self, tickers: List[str]) -> None:
        """Procesa y guarda CSVs consolidados para todos los tickers."""
        ok = 0
        fail = 0
        for ticker in tickers:
            try:
                df = self.consolidate_ticker(ticker)
                if df.empty:
                    fail += 1
                    continue
                df = df.dropna(how="all", axis=1)
                out = self.output_dir / f"{ticker}.csv"
                df.to_csv(out)
                ok += 1
                log.debug(f"[{ticker}] consolidado: {len(df)} períodos, {len(df.columns)} cols")
            except Exception as e:
                log.warning(f"[{ticker}] Error consolidando: {e}")
                fail += 1
        log.info(f"FinnhubConsolidator: ok={ok}, fail={fail}")


# =============================================================================
# 4. PARSER DE EPS SURPRISES
# =============================================================================

class EPSSurprisesParser:
    """
    Extrae series temporales de sorpresas de EPS desde eps_surprises.json.
    """

    def parse(self, path: Path) -> pd.DataFrame:
        """
        Retorna DataFrame con columnas:
          period, eps_actual, eps_estimate, eps_surprise_abs, eps_surprise_pct
        Indexado por fecha (period).
        """
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
            actual   = item.get("actual")
            estimate = item.get("estimate")
            surprise = item.get("surprisePercent")
            records.append({
                "date":               dt,
                "eps_actual":         float(actual)   if actual   is not None else np.nan,
                "eps_estimate":       float(estimate) if estimate is not None else np.nan,
                "eps_surprise_pct":   float(surprise) if surprise is not None else np.nan,
                "eps_beat":           int(actual > estimate) if (actual is not None and estimate is not None) else np.nan,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]


# =============================================================================
# 5. PARSER DE RECOMMENDATION TRENDS
# =============================================================================

class RecommendationParser:
    """
    Extrae señales de consenso de analistas desde recommendation_trends.json.
    """

    def parse(self, path: Path) -> pd.DataFrame:
        """
        Retorna DataFrame con columnas:
          buy_ratio, bearish_score, analyst_dispersion, consensus_score
        Indexado por fecha.
        """
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

            strong_buy  = float(item.get("strongBuy",  0) or 0)
            buy         = float(item.get("buy",         0) or 0)
            hold        = float(item.get("hold",        0) or 0)
            sell        = float(item.get("sell",        0) or 0)
            strong_sell = float(item.get("strongSell",  0) or 0)
            total = strong_buy + buy + hold + sell + strong_sell

            if total == 0:
                continue

            bullish = strong_buy + buy
            bearish = sell + strong_sell

            # Consensus score: +2=strong buy, +1=buy, 0=hold, -1=sell, -2=strong sell
            consensus = (
                2 * strong_buy + 1 * buy + 0 * hold - 1 * sell - 2 * strong_sell
            ) / total

            records.append({
                "date":                 dt,
                "analyst_total":        total,
                "analyst_buy_ratio":    bullish / total,
                "analyst_bearish_score": bearish / total,
                "analyst_strong_buy_pct": strong_buy / total,
                "analyst_dispersion":   (strong_buy + strong_sell) / total,
                "analyst_consensus":    consensus,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]


# =============================================================================
# 6. PARSER DE INSIDER SENTIMENT (MSPR)
# =============================================================================

class InsiderSentimentParser:
    """
    Extrae MSPR mensual desde insider_sentiment.json.
    """

    def parse(self, path: Path) -> pd.DataFrame:
        """
        Retorna DataFrame con columnas mspr, change, monthlyNetBuying.
        Indexado por fecha (año-mes).
        """
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
            year  = item.get("year")
            month = item.get("month")
            mspr  = item.get("mspr")
            change = item.get("change")
            if year is None or month is None:
                continue
            try:
                dt = pd.Timestamp(f"{int(year)}-{int(month):02d}-01")
            except Exception:
                continue
            records.append({
                "date":              dt,
                "mspr":              float(mspr)   if mspr   is not None else np.nan,
                "insider_net_buy":   float(change) if change is not None else np.nan,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df[~df.index.duplicated(keep="last")]


# =============================================================================
# 7. PARSER DE INSIDER TRANSACTIONS
# =============================================================================

class InsiderTransactionsParser:
    """
    Extrae transacciones insider desde insider_transactions.json.
    """

    # Códigos de compra y venta según Form 4
    BUY_CODES  = {"P"}          # Purchase
    SELL_CODES = {"S", "S+"}    # Sale, Sale + (plan 10b5-1)

    def parse(self, path: Path) -> pd.DataFrame:
        """
        Retorna DataFrame con columnas:
          date, name, transaction_code, shares, is_buy, is_sell
        """
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

            code   = str(item.get("transactionCode", "")).strip()
            change = item.get("change", 0) or 0
            shares = abs(float(change)) if change else 0.0

            is_buy  = code in self.BUY_CODES  or change > 0
            is_sell = code in self.SELL_CODES or change < 0

            records.append({
                "date":             dt,
                "name":             item.get("name", ""),
                "transaction_code": code,
                "shares":           shares,
                "is_buy":           int(is_buy),
                "is_sell":          int(is_sell),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df


# =============================================================================
# 8. CONSTRUCTOR DEL COMPANIES_DF DESDE PROFILES
# =============================================================================

def build_companies_df(finnhub_data_dir: str, tickers: List[str]) -> pd.DataFrame:
    """
    Construye un DataFrame de sector/industry/market_cap leyendo
    los profile.json descargados por fetcher_finnhub.

    Reemplaza companies.csv del sistema legacy.

    Returns:
        DataFrame indexado por ticker con columnas:
          company_name, sector, industry, market_cap_mil, exchange, ipo_date, currency
    """
    data_dir = Path(finnhub_data_dir)
    records  = []

    for ticker in tickers:
        path = data_dir / ticker / "profile.json"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            continue

        market_cap = profile.get("marketCapitalization")
        records.append({
            "ticker":          ticker,
            "company_name":    profile.get("name", ""),
            "sector":          _normalize_sector(profile.get("finnhubIndustry", "Unknown")),
            "industry":        profile.get("finnhubIndustry", "Unknown"),
            "market_cap_mil":  float(market_cap) if market_cap else np.nan,
            "exchange":        profile.get("exchange", ""),
            "ipo_date":        profile.get("ipo", ""),
            "currency":        profile.get("currency", "USD"),
            "country":         profile.get("country", ""),
        })

    if not records:
        log.warning("[build_companies_df] No se encontraron profiles — companies_df vacío")
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("ticker")
    log.info(f"[build_companies_df] {len(df)} empresas — {df['sector'].nunique()} sectores")
    return df


def _normalize_sector(industry: str) -> str:
    """
    Mapea la industria Finnhub a un sector agregado más estándar.
    Finnhub usa finnhubIndustry como campo de industria, no sector.
    """
    SECTOR_MAP = {
        "Technology":                          "Technology",
        "Semiconductors":                      "Technology",
        "Software":                            "Technology",
        "Internet":                            "Technology",
        "Hardware":                            "Technology",
        "Financial Services":                  "Financials",
        "Banks":                               "Financials",
        "Insurance":                           "Financials",
        "Capital Markets":                     "Financials",
        "Asset Management":                    "Financials",
        "Healthcare":                          "Healthcare",
        "Pharmaceuticals":                     "Healthcare",
        "Biotechnology":                       "Healthcare",
        "Medical Devices":                     "Healthcare",
        "Consumer Cyclical":                   "Consumer Discretionary",
        "Retail":                              "Consumer Discretionary",
        "Automobiles":                         "Consumer Discretionary",
        "Consumer Defensive":                  "Consumer Staples",
        "Food":                                "Consumer Staples",
        "Beverages":                           "Consumer Staples",
        "Energy":                              "Energy",
        "Oil, Gas & Consumable Fuels":         "Energy",
        "Industrials":                         "Industrials",
        "Aerospace & Defense":                 "Industrials",
        "Transportation":                      "Industrials",
        "Materials":                           "Materials",
        "Chemicals":                           "Materials",
        "Utilities":                           "Utilities",
        "Real Estate":                         "Real Estate",
        "REITs":                               "Real Estate",
        "Communication Services":              "Communication Services",
        "Media":                               "Communication Services",
        "Telecommunications":                  "Communication Services",
    }
    for key, sector in SECTOR_MAP.items():
        if key.lower() in industry.lower():
            return sector
    return industry  # fallback: usar la industria como sector
