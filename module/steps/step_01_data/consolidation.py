"""Consolidation utilities for Finnhub datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from module.steps.step_01_data.parsers import FinnhubSECParser, BasicFinancialsParser

log = logging.getLogger(__name__)


class FinnhubConsolidator:
	"""
	Consolida todas las fuentes Finnhub por ticker en un CSV estandarizado.

	Fuentes (por orden de prioridad):
	  1. financials_reported_quarterly.json (10-Q: Q1, Q2, Q3)
	  2. financials_reported_annual.json   (10-K: ano completo, usado para derivar Q4)
	  3. basic_financials.json            (series de ratios anuales/trimestrales)
	"""

	FLOW_COLS = [
		"revenue", "gross_profit", "operating_income", "net_income",
		"interest_expense", "income_tax", "eps_diluted", "shares_diluted",
		"operating_cash_flow", "capex", "depreciation",
	]
	STOCK_COLS = [
		"total_assets", "current_assets", "current_liabilities", "total_liabilities",
		"total_equity", "long_term_debt", "short_term_debt", "cash", "shares_outstanding",
		"total_debt",
	]

	def __init__(self, finnhub_data_dir: str = "data_finnhub"):
		self.data_dir = Path(finnhub_data_dir)
		self.output_dir = self.data_dir / "consolidated"
		self.sec_parser = FinnhubSECParser()
		self.bf_parser = BasicFinancialsParser()
		self.output_dir.mkdir(parents=True, exist_ok=True)

	def consolidate_ticker(self, ticker: str) -> pd.DataFrame:
		ticker_dir = self.data_dir / ticker

		df_q = self.sec_parser.parse_financials_json(
			ticker_dir / "financials_reported_quarterly.json"
		)
		df_a = self.sec_parser.parse_financials_json(
			ticker_dir / "financials_reported_annual.json"
		)
		bf_series, bf_pit = self.bf_parser.parse(
			ticker_dir / "basic_financials.json"
		)

		if df_q.empty and df_a.empty:
			log.warning(f"[{ticker}] No data SEC — skip")
			return pd.DataFrame()

		if not df_q.empty and not df_a.empty:
			df = self._build_quarterly_series(df_q, df_a, ticker)
		elif not df_q.empty:
			df = df_q[df_q["report_type"] == "10-Q"].copy() if "report_type" in df_q.columns else df_q.copy()
		else:
			df = df_a.copy()

		if df.empty:
			return pd.DataFrame()

		if not bf_series.empty:
			bf_series.index.name = "report_date"
			left = df.reset_index()
			right = bf_series.reset_index()
			df = pd.merge_asof(
				left,
				right,
				on="report_date",
				direction="backward",
				tolerance=pd.Timedelta("100D"),
			).set_index("report_date")

		df = self._calculate_ratios(df)

		for k, v in bf_pit.items():
			if k not in df.columns:
				df[k] = v

		df = df.drop(columns=["report_type", "fiscal_quarter"], errors="ignore")
		df.replace([np.inf, -np.inf], np.nan, inplace=True)
		return df

	def _build_quarterly_series(
		self, df_q: pd.DataFrame, df_a: pd.DataFrame, ticker: str
	) -> pd.DataFrame:
		if "report_type" in df_q.columns:
			quarterly_rows = df_q[df_q["report_type"] == "10-Q"].copy()
		else:
			quarterly_rows = df_q.copy()

		if "report_type" in df_a.columns:
			annual_rows = df_a[df_a["report_type"] == "10-K"].copy()
		else:
			annual_rows = df_a.copy()

		records = list(quarterly_rows.reset_index().to_dict("records"))

		for ann_date, ann_row in annual_rows.iterrows():
			ann_ts = pd.Timestamp(ann_date)
			fiscal_year_start = ann_ts - pd.DateOffset(months=13)

			qs_in_year = quarterly_rows[
				(quarterly_rows.index > fiscal_year_start)
				& (quarterly_rows.index <= ann_ts)
			].sort_index()

			if len(qs_in_year) < 3:
				q4_row = ann_row.to_dict()
				q4_row["report_date"] = ann_ts
				q4_row["report_type"] = "10-Q"
				q4_row["fiscal_quarter"] = 4
				records.append(q4_row)
				log.debug(
					f"[{ticker}] Q4 estimado desde 10-K (sin 3 trimestrales) para {ann_ts.date()}"
				)
				continue

			q4_row: Dict = {}
			for col in self.FLOW_COLS:
				ann_val = ann_row.get(col)
				sum_q123 = qs_in_year[col].sum() if col in qs_in_year.columns else None
				if ann_val is not None and sum_q123 is not None and not np.isnan(sum_q123):
					q4_row[col] = ann_val - sum_q123
				else:
					q4_row[col] = np.nan

			for col in self.STOCK_COLS:
				q4_row[col] = ann_row.get(col, np.nan)

			ocf = q4_row.get("operating_cash_flow")
			capex = q4_row.get("capex")
			if ocf is not None and not np.isnan(ocf) and capex is not None and not np.isnan(capex):
				q4_row["fcf"] = ocf - abs(capex)
			else:
				q4_row["fcf"] = np.nan

			lt = q4_row.get("long_term_debt") or 0.0
			st = q4_row.get("short_term_debt") or 0.0
			q4_row["total_debt"] = (lt if not np.isnan(lt) else 0.0) + (st if not np.isnan(st) else 0.0)

			q4_row["report_date"] = ann_ts
			q4_row["report_type"] = "10-Q"
			q4_row["fiscal_quarter"] = 4

			records.append(q4_row)
			log.debug(f"[{ticker}] Q4 derivado para ano fiscal {ann_ts.year}")

		if not records:
			return pd.DataFrame()

		df = pd.DataFrame(records)
		df["report_date"] = pd.to_datetime(df["report_date"])
		df = df.set_index("report_date").sort_index()
		df = df[~df.index.duplicated(keep="last")]
		return df

	def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
		df = df.copy()

		if "total_debt" not in df.columns:
			lt = df.get("long_term_debt", pd.Series(0, index=df.index)).fillna(0)
			st = df.get("short_term_debt", pd.Series(0, index=df.index)).fillna(0)
			df["total_debt"] = lt + st

		if "eps_diluted" not in df.columns or df["eps_diluted"].isna().all():
			if "net_income" in df.columns and "shares_diluted" in df.columns:
				df["eps_diluted"] = df["net_income"] / df["shares_diluted"].replace(0, np.nan)
		if "eps_diluted" in df.columns and "eps" not in df.columns:
			df["eps"] = df["eps_diluted"]

		if "shares_diluted" not in df.columns and "shares_outstanding" in df.columns:
			df["shares_diluted"] = df["shares_outstanding"]

		if "operating_income" in df.columns and "depreciation" in df.columns:
			df["ebitda"] = df["operating_income"] + df["depreciation"].fillna(0)

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
		if "operating_income" in df.columns and "total_equity" in df.columns:
			invested = df["total_equity"].fillna(0) + df["total_debt"].fillna(0)
			df["roic"] = df["operating_income"] / invested.replace(0, np.nan)
		if "fcf" in df.columns and "revenue" in df.columns:
			df["fcf_margin"] = df["fcf"] / df["revenue"].replace(0, np.nan)
		if "ebitda" in df.columns and "revenue" in df.columns:
			df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace(0, np.nan)

		if "total_equity" in df.columns:
			df["debt_equity"] = df["total_debt"] / df["total_equity"].replace(0, np.nan)
		if "current_assets" in df.columns and "current_liabilities" in df.columns:
			df["current_ratio"] = df["current_assets"] / df["current_liabilities"].replace(0, np.nan)

		flow_ttm = [
			"revenue",
			"net_income",
			"operating_income",
			"gross_profit",
			"fcf",
			"ebitda",
			"interest_expense",
			"eps_diluted",
		]
		for col in flow_ttm:
			if col in df.columns:
				df[f"{col}_ttm"] = df[col].rolling(4, min_periods=4).sum()

		if "eps_diluted_ttm" in df.columns:
			df["eps_ttm"] = df["eps_diluted_ttm"]

		if "ebitda_ttm" in df.columns:
			df["debt_to_ebitda"] = df["total_debt"] / df["ebitda_ttm"].replace(0, np.nan)
		if "operating_income_ttm" in df.columns and "interest_expense_ttm" in df.columns:
			ie = df["interest_expense_ttm"].replace(0, np.nan).abs()
			df["interest_coverage"] = df["operating_income_ttm"] / ie

		df.replace([np.inf, -np.inf], np.nan, inplace=True)
		return df

	def process_all_tickers(self, tickers: List[str]) -> None:
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
				log.debug(f"[{ticker}] consolidado: {len(df)} periodos, {len(df.columns)} cols")
			except Exception as e:
				log.warning(f"[{ticker}] Error consolidando: {e}")
				fail += 1
		log.info(f"FinnhubConsolidator: ok={ok}, fail={fail}")


def build_companies_df(finnhub_data_dir: str, tickers: List[str]) -> pd.DataFrame:
	"""
	Construye un DataFrame de sector/industry/market_cap leyendo
	los profile.json descargados por Finnhub.
	"""
	data_dir = Path(finnhub_data_dir)
	records = []

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
			"ticker": ticker,
			"company_name": profile.get("name", ""),
			"sector": _normalize_sector(profile.get("finnhubIndustry", "Unknown")),
			"industry": profile.get("finnhubIndustry", "Unknown"),
			"market_cap_mil": float(market_cap) if market_cap else np.nan,
			"exchange": profile.get("exchange", ""),
			"ipo_date": profile.get("ipo", ""),
			"currency": profile.get("currency", "USD"),
			"country": profile.get("country", ""),
		})

	if not records:
		log.warning("[build_companies_df] No se encontraron profiles — companies_df vacio")
		return pd.DataFrame()

	df = pd.DataFrame(records).set_index("ticker")
	log.info(f"[build_companies_df] {len(df)} empresas — {df['sector'].nunique()} sectores")
	return df


def _normalize_sector(industry: str) -> str:
	sector_map = {
		"Technology": "Technology",
		"Semiconductors": "Technology",
		"Software": "Technology",
		"Internet": "Technology",
		"Hardware": "Technology",
		"Financial Services": "Financials",
		"Banks": "Financials",
		"Insurance": "Financials",
		"Capital Markets": "Financials",
		"Asset Management": "Financials",
		"Healthcare": "Healthcare",
		"Pharmaceuticals": "Healthcare",
		"Biotechnology": "Healthcare",
		"Medical Devices": "Healthcare",
		"Consumer Cyclical": "Consumer Discretionary",
		"Retail": "Consumer Discretionary",
		"Automobiles": "Consumer Discretionary",
		"Consumer Defensive": "Consumer Staples",
		"Food": "Consumer Staples",
		"Beverages": "Consumer Staples",
		"Energy": "Energy",
		"Oil, Gas & Consumable Fuels": "Energy",
		"Industrials": "Industrials",
		"Aerospace & Defense": "Industrials",
		"Transportation": "Industrials",
		"Materials": "Materials",
		"Chemicals": "Materials",
		"Utilities": "Utilities",
		"Real Estate": "Real Estate",
		"REITs": "Real Estate",
		"Communication Services": "Communication Services",
		"Media": "Communication Services",
		"Telecommunications": "Communication Services",
	}
	for key, sector in sector_map.items():
		if key.lower() in industry.lower():
			return sector
	return industry
