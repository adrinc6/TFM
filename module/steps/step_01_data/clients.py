"""HTTP clients for Finnhub and Yahoo Finance."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)


class FinnhubClient:
	"""Cliente HTTP para Finnhub con rate limiting conservador (plan free)."""

	BASE = "https://finnhub.io/api/v1"
	MIN_INTERVAL = 1.05  # ~57 req/min (limite free: 60/min)

	def __init__(self, api_key: str):
		self.api_key = api_key
		self._last_call = 0.0
		self.session = requests.Session()
		self.session.headers.update({"X-Finnhub-Token": api_key})

	def _get(self, endpoint: str, params: dict | None = None):
		params = params or {}
		url = f"{self.BASE}{endpoint}"

		wait = self.MIN_INTERVAL - (time.time() - self._last_call)
		if wait > 0:
			time.sleep(wait)

		try:
			resp = self.session.get(url, params=params, timeout=20)
			self._last_call = time.time()

			if resp.status_code == 429:
				log.warning("Rate limit — esperando 65 s...")
				time.sleep(65)
				return self._get(endpoint, params)

			if resp.status_code in (401, 403):
				log.debug(f"  {resp.status_code} -> {endpoint} (Premium / no autorizado)")
				return None

			resp.raise_for_status()
			data = resp.json()
			return data if data else None

		except requests.exceptions.RequestException as e:
			log.warning(f"  Error en {endpoint}: {e}")
			return None

	def company_profile2(self, symbol):
		"""Nombre, sector, market cap, exchange, IPO date, logo, web."""
		return self._get("/stock/profile2", {"symbol": symbol})

	def basic_financials(self, symbol):
		"""
		Ratios calculados point-in-time + series anuales:
		P/E, P/B, P/S, EV/FCF, ROE, ROA, ROIC, margenes, debt/equity,
		current ratio, interest coverage, beta, 52w high/low,
		avg volume, y series anuales de revenue, netIncome, EPS, FCF, etc.
		"""
		return self._get("/stock/metric", {"symbol": symbol, "metric": "all"})

	def financials_as_reported(self, symbol, freq: str = "annual"):
		"""
		Estados financieros RAW tal como fueron reportados al SEC.
		Secciones 'bs' (Balance Sheet), 'ic' (Income Statement), 'cf' (Cash Flow).
		freq: 'annual' | 'quarterly'
		"""
		return self._get("/stock/financials-reported", {
			"symbol": symbol,
			"freq": freq,
		})

	def eps_surprises(self, symbol, limit: int = 20):
		"""EPS real vs estimado + surprise% por trimestre (hasta 20Q)."""
		return self._get("/stock/earnings", {"symbol": symbol, "limit": limit})

	def earnings_calendar(self, from_date, to_date, symbol: str = ""):
		"""Fechas de resultados con EPS y revenue real vs estimado."""
		params = {"from": from_date, "to": to_date}
		if symbol:
			params["symbol"] = symbol
		return self._get("/calendar/earnings", params)

	def recommendation_trends(self, symbol):
		"""Tendencias de analistas: strongBuy/buy/hold/sell/strongSell por mes."""
		return self._get("/stock/recommendation", {"symbol": symbol})

	def insider_transactions(self, symbol, from_date, to_date):
		"""Compras y ventas de insiders. change > 0 = compra, < 0 = venta."""
		return self._get("/stock/insider-transactions", {
			"symbol": symbol, "from": from_date, "to": to_date,
		})

	def insider_sentiment(self, symbol, from_date, to_date):
		"""
		MSPR mensual (Monthly Share Purchase Ratio), rango -100 a 100.
		Senal predictiva documentada para los siguientes 30-90 dias.
		"""
		return self._get("/stock/insider-sentiment", {
			"symbol": symbol, "from": from_date, "to": to_date,
		})

	def company_news(self, symbol, from_date, to_date):
		"""Noticias recientes (descargadas pero no procesadas por el pipeline)."""
		return self._get("/company-news", {
			"symbol": symbol, "from": from_date, "to": to_date,
		})

	def peers(self, symbol):
		"""Peers en el mismo pais y sub-industria."""
		return self._get("/stock/peers", {"symbol": symbol})

	def quote(self, symbol):
		"""Precio actual, cambio dia, maximo/minimo diario, cierre anterior."""
		return self._get("/quote", {"symbol": symbol})



class YahooClient:
	"""
	Descarga OHLCV diario de Yahoo Finance usando la API v8 directamente.
	No requiere yfinance.
	"""

	BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

	HEADERS = {
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/120.0.0.0 Safari/537.36"
		),
		"Accept": "application/json",
	}

	def __init__(self):
		self.session = requests.Session()
		self.session.headers.update(self.HEADERS)
		self._last_call = 0.0

	def _dt(self, date_str: str) -> int:
		return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

	def ohlcv(self, ticker: str, start: str, end: str) -> Optional[dict]:
		wait = 0.5 - (time.time() - self._last_call)
		if wait > 0:
			time.sleep(wait)

		params = {
			"period1": self._dt(start),
			"period2": self._dt(end),
			"interval": "1d",
			"events": "div,splits",
			"includeAdjustedClose": "true",
		}

		url = f"{self.BASE}/{ticker}"

		try:
			resp = self.session.get(url, params=params, timeout=20)
			self._last_call = time.time()

			if resp.status_code == 429:
				log.warning(f"[{ticker}] Yahoo rate limit — esperando 30 s...")
				time.sleep(30)
				return self.ohlcv(ticker, start, end)

			if resp.status_code != 200:
				log.warning(f"[{ticker}] Yahoo status {resp.status_code}")
				return None

			raw = resp.json()
			result = raw.get("chart", {}).get("result")
			if not result:
				return None

			result = result[0]
			meta = result.get("meta", {})
			timestamps = result.get("timestamp", [])
			indicators = result.get("indicators", {})
			quote_data = indicators.get("quote", [{}])[0]
			adj_close = indicators.get("adjclose", [{}])[0].get("adjclose", [])

			records = []
			for i, ts in enumerate(timestamps):
				date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
				records.append({
					"date": date_str,
					"open": _safe(quote_data.get("open", []), i),
					"high": _safe(quote_data.get("high", []), i),
					"low": _safe(quote_data.get("low", []), i),
					"close": _safe(quote_data.get("close", []), i),
					"adj_close": _safe(adj_close, i),
					"volume": _safe(quote_data.get("volume", []), i),
				})

			events = result.get("events", {})
			dividends = [
				{"date": datetime.utcfromtimestamp(int(k)).strftime("%Y-%m-%d"),
				 "amount": v.get("amount")}
				for k, v in events.get("dividends", {}).items()
			]
			splits = [
				{"date": datetime.utcfromtimestamp(int(k)).strftime("%Y-%m-%d"),
				 "ratio": f"{v.get('numerator')}/{v.get('denominator')}"}
				for k, v in events.get("splits", {}).items()
			]

			return {
				"currency": meta.get("currency"),
				"exchange": meta.get("exchangeName"),
				"data": records,
				"dividends": dividends,
				"splits": splits,
			}

		except Exception as e:
			log.warning(f"[{ticker}] Yahoo error: {e}")
			return None


def _safe(lst: list, i: int):
	try:
		v = lst[i]
		return None if v != v else v
	except (IndexError, TypeError):
		return None
