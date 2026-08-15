"""Cliente de SEC EDGAR: fechas REALES de publicación de los informes financieros.

Por qué existe este módulo: las series de fundamentales de Finnhub vienen fechadas por
`period` (el día en que la empresa CERRÓ el trimestre), no por el día en que ese dato se hizo
público. Usar un fundamental el día del cierre fiscal es lookahead: nadie lo conocía aún.
Apple cerró su trimestre el 2000-01-01 y lo publicó el 2000-02-01.

EDGAR da la fecha real (`filingDate`) desde 1993, es gratuita, oficial y no necesita clave.
Finnhub solo la da desde 2010, por lo que no sirve para simular desde el año 2000.

La SEC exige un User-Agent identificatorio y limita a ~10 req/s.
Ver `docs/metodologia.md` («Universo y datos point-in-time»).
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests

log = logging.getLogger(__name__)

# Solo informes periódicos: los que traen los fundamentales que usa el sistema. Se ignoran 8-K,
# S-1, etc.
#
# 20-F y 40-F son los equivalentes anuales de los emisores privados extranjeros (los canadienses
# usan 40-F bajo el sistema MJDS). Estuvieron fuera de esta tupla, y la consecuencia era que una
# empresa extranjera del índice —con CIK válido y sus cuentas publicadas puntualmente— quedaba sin
# ningún informe periódico y se contaba como si no existiera. Ese fallo se sumaba al sesgo de
# supervivencia sin ser mortalidad: la empresa estaba viva y sus datos estaban en EDGAR.
PERIODIC_FORMS = ("10-Q", "10-K", "20-F", "40-F")


class EdgarClient:
    """Cliente HTTP de SEC EDGAR con rate limiting conservador."""

    BASE = "https://data.sec.gov"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    TICKER_LOOKUP_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    MIN_INTERVAL = 0.15  # ~7 req/s (limite SEC: 10/s)

    def __init__(self, user_agent: str, cache_dir: Path, force_download: bool = False):
        self._last_call = 0.0
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.session = requests.Session()
        # La SEC rechaza peticiones sin User-Agent identificatorio.
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def _get(self, url: str, cache_path: Path):
        if cache_path.exists() and not self.force_download:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        wait = self.MIN_INTERVAL - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self.session.get(url, timeout=30)
            self._last_call = time.time()
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
        except requests.exceptions.RequestException as exc:
            log.warning("  Error EDGAR en %s: %s", url, exc)
            return None

    def _get_text(self, url: str, cache_path: Path) -> str | None:
        if cache_path.exists() and not self.force_download:
            return cache_path.read_text(encoding="utf-8")

        wait = self.MIN_INTERVAL - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            response = self.session.get(url, timeout=30)
            self._last_call = time.time()
            if response.status_code == 404:
                return None
            response.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(response.text, encoding="utf-8")
            return response.text
        except requests.exceptions.RequestException as exc:
            log.warning("  Error EDGAR en %s: %s", url, exc)
            return None

    def ticker_to_cik(self) -> dict[str, str]:
        """Mapeo ticker -> CIK (identificador de empresa en la SEC), con ceros a la izquierda.

        Solo lista emisores vivos y bajo su símbolo **actual**, así que una parte del universo
        histórico no resuelve aquí. Cuántos y por qué es una medida, no una interpretación: lo
        reparte `_ticker_resolution` en `module/data/ingest/pipeline.py` y queda en
        `universe_coverage.json`. No debe asumirse que los ausentes murieron —un cambio de ticker,
        una absorción o un emisor extranjero producen el mismo síntoma— y por eso existe el
        fallback `lookup_cik`.
        """
        data = self._get(self.TICKERS_URL, self.cache_dir / "company_tickers.json")
        if not data:
            return {}
        return {
            entry["ticker"].strip().upper(): str(entry["cik_str"]).zfill(10)
            for entry in data.values()
            if entry.get("ticker") and entry.get("cik_str")
        }

    def lookup_cik(self, ticker: str) -> str | None:
        """Resuelve un ticker con el buscador SEC si el mapa actual es ambiguo o no lo contiene.

        ``company_tickers.json`` solo lista emisores vivos y bajo su símbolo **actual**, así que
        falla en dos casos que no son mortalidad: una empresa que cambió de ticker y una que fue
        absorbida (su CIK y sus filings siguen en EDGAR para siempre). También puede apuntar a otra
        sociedad tras una reutilización de símbolo. El buscador de EDGAR cubre esos casos.

        No se filtra por tipo de formulario. Filtrar por ``10-Q`` excluía justo a los emisores
        extranjeros, que presentan 20-F o 40-F: el mismo defecto que tenía ``PERIODIC_FORMS``,
        repetido aquí, de modo que el fallback tampoco podía rescatarlos.
        """
        query = urlencode(
            {
                "action": "getcompany",
                "CIK": ticker,
                "owner": "exclude",
                "output": "atom",
            }
        )
        text = self._get_text(
            f"{self.TICKER_LOOKUP_URL}?{query}",
            self.cache_dir / ticker.replace("/", "-") / "ticker_lookup.xml",
        )
        return _cik_from_atom(text) if text else None

    def report_dates(self, ticker: str, cik: str) -> list[dict]:
        """Informes periódicos de una empresa: (form, period, filed_date).

        `period` es el cierre fiscal y `filed_date` el día en que se hizo público. EDGAR parte
        los filings en dos sitios cuando hay muchos: `filings.recent` y ficheros adicionales
        listados en `filings.files` (ahí están los años 90, que es justo lo que necesitamos).
        """
        ticker_dir = self.cache_dir / ticker.replace("/", "-") / f"CIK{cik}"
        submissions = self._get(
            f"{self.BASE}/submissions/CIK{cik}.json", ticker_dir / "submissions.json"
        )
        if not submissions:
            return []

        filings = submissions.get("filings", {})
        blocks = [filings.get("recent", {})]
        for extra in filings.get("files", []):
            name = extra.get("name")
            if name:
                historical = self._get(f"{self.BASE}/submissions/{name}", ticker_dir / name)
                if historical:
                    blocks.append(historical)

        rows: list[dict] = []
        for block in blocks:
            rows.extend(_extract_periodic_reports(block))
        return rows


def _extract_periodic_reports(block: dict) -> list[dict]:
    """Extrae los 10-Q/10-K de un bloque de filings (dict de listas paralelas).

    Se descartan las filas sin `reportDate`: algunos filings antiguos no lo traen y sin cierre
    fiscal no se pueden casar con las series de fundamentales.
    """
    forms = block.get("form") or []
    report_dates = block.get("reportDate") or []
    filing_dates = block.get("filingDate") or []
    accessions = block.get("accessionNumber") or []
    if not forms:
        return []

    rows: list[dict] = []
    for index, form in enumerate(forms):
        if form not in PERIODIC_FORMS:
            continue
        period = report_dates[index] if index < len(report_dates) else None
        filed = filing_dates[index] if index < len(filing_dates) else None
        if not period or not filed:
            continue
        rows.append(
            {
                "form": form,
                "period": period,
                "filed_date": filed,
                "accession": accessions[index] if index < len(accessions) else None,
            }
        )
    return rows


def _cik_from_atom(text: str) -> str | None:
    """Extrae el CIK del feed Atom de búsqueda sin depender de namespaces."""
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None
    for element in root.iter():
        if element.tag.rsplit("}", maxsplit=1)[-1] == "cik" and element.text:
            return element.text.strip().zfill(10)
    return None
