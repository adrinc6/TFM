"""HTTP clients for Finnhub and Yahoo Finance."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Por qué una descarga de precios no devolvió serie. Distinguirlos importa: "el proveedor no
# reconoce el símbolo" y "la petición falló" producen el mismo síntoma (sin datos) pero significan
# cosas opuestas. Colapsarlos en `None` convierte una avería de red en mortalidad empresarial
# aparente y sobreestima el sesgo de supervivencia (ver docs/architecture.md,
# «Cobertura del universo»).
PRICE_FAILURE_REASONS = (
	"not_found",  # 404: el proveedor no sirve el símbolo (retirado o inexistente)
	"bad_request",  # 400: símbolo conocido pero rango no servible
	"rate_limited",  # 429 tras agotar reintentos: reintentable, NO es ausencia de datos
	"http_error",  # otros códigos no 200
	"transport_error",  # excepción de red/parseo
	"empty_series",  # respuesta 200 válida pero sin observaciones
)


class FinnhubClient:
	"""Cliente HTTP para Finnhub con rate limiting conservador (plan free)."""

	BASE = "https://finnhub.io/api/v1"
	MIN_INTERVAL = 1.05  # ~57 req/min (limite free: 60/min)
	MAX_429_RETRIES = 5

	def __init__(self, api_key: str):
		self.api_key = api_key
		self._last_call = 0.0
		self.session = requests.Session()
		self.session.headers.update({"X-Finnhub-Token": api_key})

	def _get(self, endpoint: str, params: dict | None = None, _attempt: int = 1):
		params = params or {}
		url = f"{self.BASE}{endpoint}"

		wait = self.MIN_INTERVAL - (time.time() - self._last_call)
		if wait > 0:
			time.sleep(wait)

		try:
			resp = self.session.get(url, params=params, timeout=20)
			self._last_call = time.time()

			if resp.status_code == 429:
				if _attempt >= self.MAX_429_RETRIES:
					log.error(f"  Rate limit exceeded {self.MAX_429_RETRIES} retries -> {endpoint}, giving up.")
					return None
				log.warning(f"  Rate limit — waiting 65 s... (attempt {_attempt}/{self.MAX_429_RETRIES})")
				time.sleep(65)
				return self._get(endpoint, params, _attempt=_attempt + 1)

			if resp.status_code in (401, 403):
				log.debug(f"  {resp.status_code} -> {endpoint} (Premium / unauthorised)")
				return None

			resp.raise_for_status()
			data = resp.json()
			return data if data else None

		except requests.exceptions.RequestException as e:
			log.warning(f"  Error on {endpoint}: {e}")
			return None

	def company_profile2(self, symbol):
		"""Name, sector, market cap, exchange, IPO date, logo, web."""
		return self._get("/stock/profile2", {"symbol": symbol})

	def basic_financials(self, symbol):
		"""
		Point-in-time calculated ratios + annual series:
		P/E, P/B, P/S, EV/FCF, ROE, ROA, ROIC, margins, debt/equity,
		current ratio, interest coverage, beta, 52w high/low,
		avg volume, and annual series for revenue, netIncome, EPS, FCF, etc.
		"""
		return self._get("/stock/metric", {"symbol": symbol, "metric": "all"})

	def company_news(self, symbol, from_date, to_date):
		"""Noticias recientes (descargadas pero no procesadas por el pipeline)."""
		return self._get("/company-news", {
			"symbol": symbol, "from": from_date, "to": to_date,
		})


class YahooClient:
	"""
	Descarga OHLCV diario de Yahoo Finance usando la API v8 directamente.
	No requiere yfinance.
	"""

	BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
	MAX_429_RETRIES = 5

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
		self.last_failure_reason: str | None = None

	def _dt(self, date_str: str) -> int:
		return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

	def ohlcv(self, ticker: str, start: str, end: str, _attempt: int = 1) -> Optional[dict]:
		"""Serie diaria, o `None` si no hay datos.

		Cuando devuelve `None`, el motivo queda en `self.last_failure_reason` (uno de
		`PRICE_FAILURE_REASONS`). El motivo NO se deduce del valor de retorno: un 404 del
		proveedor y un corte de red son indistinguibles desde fuera, pero solo el primero
		justifica excluir al ticker del universo.
		"""
		self.last_failure_reason: str | None = None
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
				if _attempt >= self.MAX_429_RETRIES:
					log.error(f"[{ticker}] Yahoo rate limit exceeded {self.MAX_429_RETRIES} retries, giving up.")
					self.last_failure_reason = "rate_limited"
					return None
				log.warning(f"[{ticker}] Yahoo rate limit — waiting 30 s... (attempt {_attempt}/{self.MAX_429_RETRIES})")
				time.sleep(30)
				return self.ohlcv(ticker, start, end, _attempt=_attempt + 1)

			if resp.status_code != 200:
				log.warning(f"[{ticker}] Yahoo status {resp.status_code}")
				self.last_failure_reason = {
					404: "not_found",
					400: "bad_request",
				}.get(resp.status_code, "http_error")
				return None

			raw = resp.json()
			result = raw.get("chart", {}).get("result")
			if not result:
				self.last_failure_reason = "empty_series"
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
			self.last_failure_reason = "transport_error"
			return None


def _safe(lst: list, i: int):
	try:
		v = lst[i]
		return None if v != v else v
	except (IndexError, TypeError):
		return None
