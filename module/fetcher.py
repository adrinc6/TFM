# =============================================================================
# Descarga y cachea todos los datos necesarios
# =============================================================================

import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import logging
from environment import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

## Gestión de histórico de actualizaciones
def load_history():
    """Carga el histórico de actualizaciones"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_history(history):
    """Guarda el histórico de actualizaciones"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def update_ticker_category(ticker, category, date=None):
    """Actualiza la fecha de una categoría para un ticker"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    history = load_history()
    if ticker not in history:
        history[ticker] = {}
    
    history[ticker][category] = date
    save_history(history)

def get_ticker_category_date(ticker, category):
    """Obtiene la fecha de una categoría específica"""
    history = load_history()
    if ticker not in history:
        return None
    return history[ticker].get(category, None)

def needs_update(ticker, category, days=1):
    """Verifica si una categoría de un ticker necesita actualización"""
    last_update = get_ticker_category_date(ticker, category)
    if last_update is None:
        return True
    
    last_date = datetime.strptime(last_update, "%Y-%m-%d")
    days_diff = (datetime.now() - last_date).days
    return days_diff >= days


## Descarga de tickers
def fetch_tickers() -> list[str]:
    tickers_raw = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "TSLA", "AVGO",
        "LLY", "JPM", "UNH", "XOM", "V", "MA", "PG", "COST", "HD", "JNJ",
        "ABBV", "NFLX", "ORCL", "BAC", "WMT", "CVX", "CRM", "ADBE", "MRK", "AMD",
        "PEP", "TMO", "KO", "ABT", "WFC", "CSCO", "DHR", "PLTR", "QCOM", "LIN",
        "INTU", "ACN", "GE", "TXN", "AMAT", "ISRG", "IBM", "UBER", "CAT", "PGR",
        "MS", "RTX", "AXP", "HON", "LOW", "PM", "SYK", "NEE", "BKNG", "SPGI",
        "UNP", "ETN", "VRTX", "C", "GILD", "TJX", "ADP", "BSX", "AMGN", "LRCX",
        "PANW", "MMC", "CB", "ADI", "MCD", "MDLZ", "MU", "CRWD", "SCHW", "DE",
        "REGN", "GEV", "PH", "APH", "ECL", "KLAC", "SNPS", "PLD", "CDNS", "ICE",
        "TT", "ZTS", "MAR", "ADSK", "SHW", "WM", "CMG", "MCO", "HOOD", "APP",
        "ARES", "TTD", "KKR", "CIEN", "IBKR", "EME", "GDDY", "VRT", "CRH", "CVNA",
        "MMM", "AOS", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL",
        "MO", "AMCR", "AEE", "AAL", "AEP", "AIG", "AMT", "AWK",
        "AMP", "AME", "ANSS", "AON", "APA", "APTV", "ACGL", "ADM",
        "AJG", "AIZ", "T", "ATO", "AZO", "AVB", "AVY", "BKR", "BALL",
        "BK", "BBWI", "BAX", "BDX", "BBY", "BIO", "TECH", "BIIB", "BLK", "BX",
        "BA", "BWA", "BXP", "BMY", "BR", "BRO", "BF-B", "BLDR",
        "BG", "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR",
        "CTLT", "CBOE", "CBRE", "CDW", "CE", "COR", "CNC", "CNP", "CF",
        "CHRW", "CRL", "CHTR", "CHD", "CI", "CINF",
        "CTAS", "CFG", "CLX", "CME", "CMS", "CTSH", "CL",
        "CMCSA", "CMA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW",
        "CPAY", "CTVA", "CSGP", "CTRA", "CCI", "CSX", "CMI", "CVS",
        "DRI", "DVA", "DAY", "DAL", "DVN", "DXCM", "FANG", "DLR", "DFS",
        "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DHI", "DTE", "DUK", "DD",
        "EMN", "EBAY", "EIX", "EW", "EA", "ELV", "EMR", "ENPH",
        "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY",
        "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", "FFIV", "FDS",
        "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "FMC",
        "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT",
        "GEHC", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GPN",
        "GL", "GS", "HAL", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES",
        "HPE", "HLT", "HOLX", "HRL", "HST", "HWM", "HPQ", "HUM",
        "HBAN", "HII", "IEX", "IDXX", "ITW", "ILMN", "INCY", "IR", "PODD",
        "INTC", "IFF", "IVZ", "INVH", "IQV", "IRM", "JBHT",
        "JKHY", "J", "JCI", "JNPR", "K", "KDP", "KEY", "KEYS",
        "KMB", "KIM", "KMI", "KR", "LHX", "LH",
        "LW", "LVS", "LDOS", "LEN", "LYV", "LKQ", "LMT", "L",
        "LULU", "LYB", "MTB", "MRO", "MPC", "MKTX", "MLM", "MAS",
        "MTCH", "MKC", "MCK", "MDT", "MET", "MTD",
        "MGM", "MCHP", "MAA", "MRNA", "MHK", "MOH", "TAP",
        "MPWR", "MNST", "MOS", "MSI", "MSCI", "NDAQ", "NTAP",
        "NEM", "NWL", "NWS", "NWSA", "NKE", "NI", "NDSN", "NSC", "NTRS",
        "NOC", "NCLH", "NRG", "NUE", "NVR", "NXPI", "ORLY", "OXY", "ODFL",
        "OMC", "ON", "OKE", "OTIS", "PCAR", "PKG", "PARA",
        "PAYX", "PAYC", "PYPL", "PNR", "PFE", "PCG", "PSX", "PNW",
        "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PRU",
        "PEG", "PTC", "PSA", "PHM", "QRVO", "PWR", "DGX", "RL", "RJF",
        "O", "REG", "RF", "RSG", "RMD", "RVTY", "RHI", "ROK",
        "ROL", "ROP", "ROST", "RCL", "SBAC", "SLB", "STX", "SRE",
        "NOW", "SPG", "SWKS", "SJM", "SNA", "SO", "LUV", "SWK", "SBUX",
        "STT", "STLD", "STE", "SYF", "SYY", "TMUS", "TROW", "TTWO",
        "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TXT",
        "TSG", "TOL", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB",
        "UDR", "ULTA", "UAL", "UPS", "URI", "UHS", "VLO", "VTR",
        "VRSN", "VRSK", "VZ", "VFC", "VICI", "VMC", "WAB",
        "WBA", "DIS", "WBD", "WAT", "WEC", "WELL", "WST",
        "WDC", "WRK", "WY", "WHR", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM",
        "ZBRA", "ZBH",
    ]
    # Eliminar duplicados preservando orden
    tickers = list(dict.fromkeys(tickers_raw))
    # tickers = [
    #     "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK-B",
    #     "WMT", "LLY", "JPM", "XOM", "JNJ", "V", "MU", "COST", "MA", "ORCL",
    #     "ABBV", "HD", "PG", "CVX", "BAC", "CAT", "GE", "AMD", "KO", "NFLX",
    #     "CSCO", "MRK", "LRCX", "PLTR", "AMAT", "PM", "GS", "MS", "RTX", "WFC",
    #     "UNH", "TMUS", "MCD", "GEV", "LIN", "PEP", "INTC", "AXP", "TMO", "CRM",
    #     "IBM", "QCOM", "TXN", "NOW", "INTU", "ISRG", "UBER", "DIS", "ADBE", "ABT",
    #     "PGR", "CAT", "SPGI", "BSX", "BA", "SCHW", "TJX", "NEE", "AMGN", "HON",
    #     "BLK", "C", "UNP", "GILD", "CMCSA", "ADP", "PFE", "SYK", "DE", "LOW",
    #     "ETN", "PANW", "DHR", "COF", "MMC", "VRTX", "COP", "ADI", "MDT", "CB",
    #     "PLD", "CI", "ADI", "PH", "LMT", "NOC", "MO", "HWM", "FDX", "JCI"
    # ]
    # tickers = [
    #     "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY",
    #     "JPM", "WMT", "XOM", "V", "MA", "UNH", "COST", "ORCL", "JNJ", "PG",
    #     "HD", "ABBV", "NFLX", "KO", "MRK", "CVX", "ABT", "PEP", "ADBE", "CRM",
    #     "AMD", "BAC", "MCD", "ACN", "TMO", "LIN", "CSCO", "INTU", "QCOM", "AMAT",
    #     "TXN", "DIS", "DHR", "VZ", "GE", "IBM", "PM", "ISRG", "NOW", "CAT",
    #     "UNP", "PFE", "GS", "HON", "SYK", "BKNG", "AMGN", "LOW", "RTX", "SPGI",
    #     "BSX", "TJX", "NEE", "BLK", "MS", "DE", "ADP", "VRTX", "MMC", "SCHW",
    #     "GILD", "LMT", "ETN", "CB", "PLTR", "PANW", "MU", "T", "CI", "MDT",
    #     "REGN", "ZTS", "PLD", "PGR", "CDNS", "SNPS", "LRCX", "KLAC", "C", "AON",
    #     "BSX", "FI", "WM", "APH", "ICE", "EQIX", "TGT", "MO", "MCK", "CMG",
    #     "SHW", "SLB", "CDW", "PH", "EOG", "NXPI", "MAR", "EMR", "TT", "PSX",
    #     "GD", "BDX", "HCA", "ITW", "USB", "CVS", "PNC", "TDG", "ORLY", "CTAS",
    #     "ROP", "MCO", "ECL", "MSI", "MPC", "WELL", "D", "NSC", "CARR", "DXCM",
    #     "F", "CSX", "ADSK", "FCX", "GD", "SRE", "AJG", "AZO", "EW", "BK",
    #     "HRE", "MET", "PSA", "AIG", "HUM", "SNPS", "TRV", "O", "TFC", "O",
    #     "TEL", "MNST", "DLR", "COR", "A", "OKE", "DDUX", "AMP", "PCAR", "BKR",
    #     "DHI", "CMI", "DFS", "ALL", "ROK", "HAL", "VRSK", "STZ", "KDP", "KEYS",
    #     "SYY", "GPN", "HWM", "KMB", "CEG", "ODFL", "LEN", "STT", "EXC", "ED",
    #     "PAYX", "VLO", "IQV", "FAST", "MTB", "ALGN", "AME", "CTSH", "OTIS", "DLTR",
    #     "PRU", "FITB", "KHC", "MCHP", "ACGL", "URI", "KR", "ROST", "GWW", "GLW",
    #     "IDXX", "VRSN", "EBAY", "HPQ", "VICI", "WDS", "MPWR", "WY", "AWK", "BBY",
    #     "ZBH", "WBD", "FE", "WEC", "SBAC", "MTD", "STE", "EFX", "CBRE", "WTW",
    #     "BLL", "FSLR", "ARE", "LH", "LUV", "VTR", "ES", "DTE", "INVH", "UAL",
    #     "EXPD", "TRGP", "EGP", "XEL", "MKC", "BG", "CHD", "DRI", "CAG", "RJF",
    #     "L", "TER", "STX", "TROW", "ATO", "BRO", "WRB", "PKG", "TYL", "IFF"
    # ]
    return tickers


## Precios históricos
def fetch_prices(tickers: list[str], start: str = START_DATE, end: str = END_DATE) -> None:
    """
    Descarga precios ajustados.
    Devuelve MultiIndex columnas: (ticker, field)
    """
    prices_dir = Path(DATA_DIR)/"prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    tickers_downloaded = [t for t in tickers if not needs_update(t, "prices", DAYS_UPDATE)]
    if not FORCE_DOWNLOAD:
        tickers = [t for t in tickers if t not in tickers_downloaded]
    log.info(f"Descargando precios ({start} → {end})")
    log.info(f"Ya hay {len(tickers_downloaded)} tickers actualizados")

    for idx, ticker in enumerate(tickers, 1):
        ticker_path = prices_dir / f"{ticker}.csv"
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                log.warning(f"  ⚠ {ticker}: no se han obtenido datos")
                continue

            # Normalizar MultiIndex de yfinance >=0.2
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Guardar con columnas simples (field)
            df.to_csv(ticker_path, index=True)
            update_ticker_category(ticker, "prices")
            log.info(f"[{idx}/{len(tickers)}] {ticker} descargado")
            time.sleep(0.1)

        except Exception as e:
            error_type = type(e).__name__
            log.warning(f"{error_type}: {str(e).lower()}")
            if error_type == "YFRateLimitError" or "rate limit" in str(e).lower():
                log.warning(f"  ⚠ {ticker}: rate limit: {e}")
            else:
                log.warning(f"  ❌ Error descargando precios de {ticker}: {e}")
                update_ticker_category(ticker, "prices")

    log.info("Descarga completada.")


## Fundamentales
def fetch_fundamentals(tickers: list[str]) -> None:
    
    fundamentals_dir = Path(DATA_DIR) / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)

    tickers_downloaded = [t for t in tickers if not needs_update(t, "fundamentals", DAYS_UPDATE)]
    if not FORCE_DOWNLOAD:
        tickers = [t for t in tickers if t not in tickers_downloaded]
    
    log.info(f"Descargando datos trimestrales")
    log.info(f"Ya hay {len(tickers_downloaded)} tickers actualizados")

    for idx, ticker in enumerate(tickers, 1):
        try:
            ticker_path = fundamentals_dir / f"{ticker}.csv"
            t = yf.Ticker(ticker)
            
            # Obtener datos
            balance = t.quarterly_balance_sheet
            cash = t.quarterly_cashflow
            income = t.quarterly_income_stmt
            financials = t.quarterly_financials
            
            if balance is not None and not balance.empty:
                balance = balance.T
            else:
                balance = pd.DataFrame()
                
            if cash is not None and not cash.empty:
                cash = cash.T
            else:
                cash = pd.DataFrame()
                
            if income is not None and not income.empty:
                income = income.T
            else:
                income = pd.DataFrame()
                
            if financials is not None and not financials.empty:
                financials = financials.T
            else:
                financials = pd.DataFrame()
            
            df = pd.concat([balance, cash, income, financials], axis=1)
            if df.empty:
                log.warning(f"  ⚠ {ticker}: sin datos disponibles")
                continue
            
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[~df.index.duplicated()]

            df.to_csv(ticker_path, index=True)
            update_ticker_category(ticker, "fundamentals")
            log.info(f"[{idx}/{len(tickers)}] {ticker} descargado")
            time.sleep(0.1)

        except Exception as e:
            error_type = type(e).__name__
            msg = str(e).lower()
            if error_type == "YFRateLimitError" or "rate limit" in msg:
                log.warning(f"  ⚠ {ticker}: rate limit: {e}")
            else:
                log.warning(f"  ❌ Error descargando datos de {ticker}: {e}")


## Transacciones de insiders
def fetch_insider_trades(tickers: list[str]) -> None:
    """
    Descarga transacciones insider desde yfinance.
    Columnas: date, transactionType, shares, value.
    """
    insider_dir = Path(DATA_DIR)/"insider"
    insider_dir.mkdir(parents=True, exist_ok=True)

    tickers_downloaded = [t for t in tickers if not needs_update(t, "insider", DAYS_UPDATE)]
    if not FORCE_DOWNLOAD:
        tickers = [t for t in tickers if t not in tickers_downloaded]
    log.info(f"Descargando insider trades")
    log.info(f"Ya hay {len(tickers_downloaded)} tickers actualizados")

    for idx, ticker in enumerate(tickers, 1):
        try:
            ticker_path = insider_dir / f"{ticker}.csv"
            t = yf.Ticker(ticker)
            df = t.insider_transactions

            if df is None or df.empty:
                log.warning(f"  ⚠ {ticker}: sin datos insider")
                continue

            # Normalizar columnas
            df = df.copy().reset_index()
            col_map = {
                "Start Date":  "date",
                "Date":        "date",
                "Shares":      "shares",
                "Value":       "value",
                "Text":        "transactionType",
                "Transaction": "transactionType",
            }
            existing = set(df.columns)
            rename = {k: v for k, v in col_map.items() if k in existing and v not in existing}
            df = df.rename(columns=rename)
            df = df.loc[:, ~df.columns.duplicated()]

            keep = [c for c in ["date", "transactionType", "shares", "value"] if c in df.columns]
            df = df[keep].copy()

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")

            df.to_csv(ticker_path, index=False)
            update_ticker_category(ticker, "insider")
            log.info(f"[{idx}/{len(tickers)}] {ticker} insider trades descargado")
            time.sleep(0.1)

        except Exception as e:
            error_type = type(e).__name__
            msg = str(e).lower()
            
            if error_type == "YFRateLimitError" or "rate limit" in msg:
                log.warning(f"  ⚠ {ticker}: rate limit: {e}")
            else:
                log.warning(f"  ❌ Error descargando insider trades de {ticker}: {e}")


## Estimaciones de analistas
def fetch_analyst_estimates(tickers: list[str]) -> None:
    analyst_dir = Path(DATA_DIR)/"analyst"
    analyst_dir.mkdir(parents=True, exist_ok=True)

    tickers_downloaded = [t for t in tickers if not needs_update(t, "analyst", DAYS_UPDATE)]
    if not FORCE_DOWNLOAD:
        tickers = [t for t in tickers if t not in tickers_downloaded]
    log.info(f"Descargando analyst estimates")
    log.info(f"Ya hay {len(tickers_downloaded)} tickers actualizados")

    for idx, ticker in enumerate(tickers, 1):
        try:
            ticker_path = analyst_dir / f"{ticker}.csv"
            t = yf.Ticker(ticker)
            ed = t.earnings_dates

            if ed is None or ed.empty:
                log.warning(f"  ⚠ {ticker}: sin datos analyst")
                continue

            df = ed.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            rename = {
                "EPS Estimate":  "eps_est",
                "Reported EPS":  "eps_reported",
                "Surprise(%)":   "eps_surprise_pct",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

            if "eps_est" in df.columns:
                df["eps_revision"] = df["eps_est"].pct_change()

            df = df[~df.index.duplicated()].replace([float("inf"), float("-inf")], float("nan"))

            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.to_csv(ticker_path, index=True)
            update_ticker_category(ticker, "analyst")
            log.info(f"[{idx}/{len(tickers)}] {ticker} analyst estimates descargado")
            time.sleep(0.1)

        except Exception as e:
            error_type = type(e).__name__
            msg = str(e).lower()
            
            if error_type == "YFRateLimitError" or "rate limit" in msg:
                log.warning(f"  ⚠ {ticker}: rate limit: {e}")
            else:
                log.warning(f"  ❌ Error descargando analyst estimates de {ticker}: {e}")


## Datos macroeconómicos
def fetch_macro_data() -> None:
    """
    Descarga VIX, S&P500, tipos 10Y y 2Y desde yfinance.
    Guarda cada serie en CSV individual.
    """
    macro_dir = Path(DATA_DIR)/"macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Descargando datos macro (VIX, S&P500, yields)")

    def _get_close(ticker: str, name: str, retries: int = 4, delay: float = 8.0) -> pd.Series | None:
        ticker_path = macro_dir / f"{name}.csv"
        
        # Si ya existe, cargar desde archivo
        if ticker_path.exists():
            try:
                df = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
                s = df.iloc[:, 0].copy()
                s.name = name
                log.info(f"  {name} cargado desde archivo")
                return s
            except Exception as e:
                log.warning(f"  Error leyendo {name}: {e}, re-descargando...")

        for attempt in range(retries):
            try:
                df = yf.download(
                    ticker,
                    start=START_DATE,
                    end=END_DATE,
                    auto_adjust=True,
                    progress=False
                )
                if df.empty:
                    log.warning(f"  ⚠ {ticker} vacío (intento {attempt+1}/{retries})")
                    time.sleep(delay)
                    continue
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                s = df["Close"].copy()
                s.name = name
                s.to_frame().to_csv(ticker_path, index=True)
                log.info(f"  {name} ({ticker}) descargado")
                update_ticker_category(name, "macro")
                return s
            
            except Exception as e:
                wait = delay * (attempt + 1)
                log.warning(f"  ❌ {ticker} intento {attempt+1}/{retries}: {e}. Esperando {wait}s...")
                time.sleep(wait)

        log.error(f"  ✗ No se pudo descargar {ticker} tras {retries} intentos")
        return None

    # Descargar series
    vix   = _get_close("^VIX",  "vix")
    sp500 = _get_close("^GSPC", "sp500")
    t10y  = _get_close("^TNX",  "t10y")
    t2y   = _get_close("^IRX",  "t2y")   # ^IRX = 13-week T-bill (proxy short-term rate)

    available = {
        n: s for n, s in {"vix": vix, "sp500": sp500, "t10y": t10y, "t2y": t2y}.items()
        if s is not None
    }
    
    if not available:
        log.error("No se pudo descargar ningún dato macro")
        return

    # Construir DataFrame consolidado
    macro = pd.concat(list(available.values()), axis=1)

    # Calcular indicadores derivados
    if "t10y" in macro.columns and "t2y" in macro.columns:
        macro["yield_curve"] = macro["t10y"] - macro["t2y"]
        yc_df = macro[["yield_curve"]].copy()
        yc_df.to_csv(macro_dir / "yield_curve.csv", index=True)
        log.info("  yield_curve calculado")
        update_ticker_category("yield_curve", "macro")

    if "sp500" in macro.columns:
        macro["sp500_momentum_3m"]  = macro["sp500"].pct_change(63)
        macro["sp500_momentum_12m"] = macro["sp500"].pct_change(252)
        
        mom3m_df = macro[["sp500_momentum_3m"]].copy()
        mom3m_df.to_csv(macro_dir / "sp500_momentum_3m.csv", index=True)
        
        mom12m_df = macro[["sp500_momentum_12m"]].copy()
        mom12m_df.to_csv(macro_dir / "sp500_momentum_12m.csv", index=True)
        
        log.info("  sp500_momentum_3m y sp500_momentum_12m calculados")
        update_ticker_category("sp500_momentum", "macro")

    # Guardar consolidado
    macro_consolidated = macro_dir / "macro_consolidated.csv"
    macro.to_csv(macro_consolidated, index=True)
    log.info("Macro consolidado guardado")
    update_ticker_category("consolidated", "macro")


## Unir CSV datasets
def unir_csvs_datasets():
    ds = Path(DATA_DIR) / "datasets"
    income = [str(ds / "us-income-quarterly.csv"), str(ds / "us-income-banks-quarterly.csv"), str(ds / "us-income-insurance-quarterly.csv")]
    unir_csvs(income, str(ds / "income-quarterly.csv"))
    cashflow = [str(ds / "us-cashflow-quarterly.csv"), str(ds / "us-cashflow-banks-quarterly.csv"), str(ds / "us-cashflow-insurance-quarterly.csv")]
    unir_csvs(cashflow, str(ds / "cashflow-quarterly.csv"))
    balance = [str(ds / "us-balance-quarterly.csv"), str(ds / "us-balance-banks-quarterly.csv"), str(ds / "us-balance-insurance-quarterly.csv")]
    unir_csvs(balance, str(ds / "balance-quarterly.csv"))

def unir_csvs(archivos: list[str], archivo_salida: str, valores_vacios=''):
    
    dataframes = [pd.read_csv(archivo, sep=';') for archivo in archivos]
    
    columnas_base = list(dataframes[0].columns)
    columnas_adicionales = set()
    for df in dataframes[1:]:
        columnas_adicionales.update(df.columns)
    
    columnas_adicionales = sorted(list(columnas_adicionales - set(columnas_base)))
    todas_las_columnas = columnas_base + columnas_adicionales
    
    dataframes_completos = [df.reindex(columns=todas_las_columnas) for df in dataframes]
    df_final = pd.concat(dataframes_completos, ignore_index=True)
    df_final = df_final.fillna(valores_vacios)
    df_final.to_csv(archivo_salida, index=False, sep=';')
    
    log.info(f"Archivos unidos exitosamente en '{archivo_salida}'")


## Preparar los datos para el modelo
class FinancialDataConsolidator:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.fundamentals_path = self.data_path / "fundamentals"
        self.balance_df = None
        self.cashflow_df = None
        self.income_df = None
        
        # Mapeo de equivalencias entre datasets
        self.column_mapping = {
            'revenue': ['Revenue', 'Total Revenue', 'Operating Revenue'],
            'net_income': ['Net Income', 'Net Income From Continuing And Discontinued Operation'],
            'operating_income': ['Operating Income (Loss)', 'Operating Income', 'EBIT'],
            'cogs': ['Cost Of Revenue', 'Cost of Goods Sold'],
            'gross_profit': ['Gross Profit'],
            'operating_cash_flow': ['Net Cash from Operating Activities', 'Operating Cash Flow'],
            'fcf': ['Free Cash Flow'],
            'total_assets': ['Total Assets'],
            'total_equity': ['Total Equity', 'Total Equity Gross Minority Interest', 'Stockholders Equity'],
            'total_liabilities': ['Total Liabilities', 'Total Liabilities Net Minority Interest'],
            'current_assets': ['Total Current Assets', 'Current Assets'],
            'current_liabilities': ['Total Current Liabilities', 'Current Liabilities'],
            'long_term_debt': ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'],
            'short_term_debt': ['Short Term Debt', 'Current Debt', 'Current Debt And Capital Lease Obligation'],
            'cash': ['Cash, Cash Equivalents & Short Term Investments', 'Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'],
            'shares_diluted': ['Shares (Diluted)', 'Diluted Average Shares'],
            'shares_basic': ['Shares (Basic)', 'Basic Average Shares'],
            'interest_expense': ['Interest Expense, Net'],
            'tax_expense': ['Income Tax (Expense) Benefit, Net', 'Tax Provision'],
            'capex': ['Capital Expenditure', 'Purchase Of PPE'],
            'depreciation': ['Depreciation & Amortization', 'Depreciation Amortization Depletion', 'Depreciation And Amortization'],
        }
        
    def load_source_data(self):
        """Carga los tres datasets principales"""
        try:
            self.balance_df = pd.read_csv(self.data_path / "datasets" / "balance-quarterly.csv", sep=";")
            print(f"✓ Balance cargado: {len(self.balance_df)} filas")
        except Exception as e:
            print(f"✗ Error cargando balance: {e}")
            
        try:
            self.cashflow_df = pd.read_csv(self.data_path / "datasets" / "cashflow-quarterly.csv", sep=";")
            print(f"✓ Cashflow cargado: {len(self.cashflow_df)} filas")
        except Exception as e:
            print(f"✗ Error cargando cashflow: {e}")
            
        try:
            self.income_df = pd.read_csv(self.data_path / "datasets" / "income-quarterly.csv", sep=";")
            print(f"✓ Income cargado: {len(self.income_df)} filas")
        except Exception as e:
            print(f"✗ Error cargando income: {e}")
    
    def _find_col(self, df, equivalent_names):
        """Busca una columna por nombres equivalentes"""
        for name in equivalent_names:
            if name in df.columns:
                return name
        return None
    
    def _get_value(self, row, equivalent_names):
        """Obtiene valor de una fila buscando por nombres equivalentes"""
        for name in equivalent_names:
            if name in row.index:
                val = row[name]
                if pd.notna(val):
                    return val
        return np.nan
    
    def get_ticker_fundamentals(self, ticker):
        """Lee el CSV de fundamentals para un ticker específico"""
        ticker_file = self.fundamentals_path / f"{ticker}.csv"
        if ticker_file.exists():
            df = pd.read_csv(ticker_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        return None
    
    def extract_ticker_data(self, ticker):
        """Extrae datos de los tres datasets para un ticker"""
        data = {}
        
        if self.balance_df is not None:
            balance = self.balance_df[self.balance_df['Ticker'] == ticker].copy()
            if len(balance) > 0:
                data['balance'] = balance
        
        if self.cashflow_df is not None:
            cashflow = self.cashflow_df[self.cashflow_df['Ticker'] == ticker].copy()
            if len(cashflow) > 0:
                data['cashflow'] = cashflow
        
        if self.income_df is not None:
            income = self.income_df[self.income_df['Ticker'] == ticker].copy()
            if len(income) > 0:
                data['income'] = income
        
        return data
    
    def consolidate_ticker(self, ticker):
        """Consolida todos los datos para un ticker, priorizando ticker CSV > otros datasets"""
        
        # Cargar datos
        fundamentals = self.get_ticker_fundamentals(ticker)
        other_data = self.extract_ticker_data(ticker)
        
        # Diccionario para almacenar períodos consolidados
        consolidated = {}
        
        # 1. PROCESAR DATOS DE LOS 3 CSVs (años pasados + recientes)
        
        # Income statement
        if 'income' in other_data:
            income = other_data['income']
            for _, row in income.iterrows():
                period = row['Report Date']
                if period not in consolidated:
                    consolidated[period] = {}
                
                consolidated[period].update({
                    'revenue': self._get_value(row, self.column_mapping['revenue']),
                    'cogs': self._get_value(row, self.column_mapping['cogs']),
                    'gross_profit': self._get_value(row, self.column_mapping['gross_profit']),
                    'operating_income': self._get_value(row, self.column_mapping['operating_income']),
                    'net_income': self._get_value(row, self.column_mapping['net_income']),
                    'interest_expense': self._get_value(row, self.column_mapping['interest_expense']),
                    'tax_expense': self._get_value(row, self.column_mapping['tax_expense']),
                    'depreciation': self._get_value(row, self.column_mapping['depreciation']),
                    'shares_diluted': self._get_value(row, self.column_mapping['shares_diluted']),
                    'shares_basic': self._get_value(row, self.column_mapping['shares_basic']),
                })
        
        # Balance sheet
        if 'balance' in other_data:
            balance = other_data['balance']
            for _, row in balance.iterrows():
                period = row['Report Date']
                if period not in consolidated:
                    consolidated[period] = {}
                
                consolidated[period].update({
                    'total_assets': self._get_value(row, self.column_mapping['total_assets']),
                    'current_assets': self._get_value(row, self.column_mapping['current_assets']),
                    'current_liabilities': self._get_value(row, self.column_mapping['current_liabilities']),
                    'total_liabilities': self._get_value(row, self.column_mapping['total_liabilities']),
                    'total_equity': self._get_value(row, self.column_mapping['total_equity']),
                    'long_term_debt': self._get_value(row, self.column_mapping['long_term_debt']),
                    'short_term_debt': self._get_value(row, self.column_mapping['short_term_debt']),
                    'cash': self._get_value(row, self.column_mapping['cash']),
                })
        
        # Cashflow
        if 'cashflow' in other_data:
            cashflow = other_data['cashflow']
            for _, row in cashflow.iterrows():
                period = row['Report Date']
                if period not in consolidated:
                    consolidated[period] = {}
                
                ocf = self._get_value(row, self.column_mapping['operating_cash_flow'])
                capex = self._get_value(row, self.column_mapping['capex'])
                
                consolidated[period].update({
                    'operating_cash_flow': ocf,
                    'capex': capex,
                    'fcf': ocf - capex if pd.notna(ocf) and pd.notna(capex) else np.nan,
                })
        
        # 2. PROCESAR FUNDAMENTALS (SOBREESCRIBE SI EXISTE - prioridad)
        if fundamentals is not None:
            for date, row in fundamentals.iterrows():
                period = date.strftime('%Y-%m-%d')
                if period not in consolidated:
                    consolidated[period] = {}
                
                # Aplicar mapping: buscar equivalencias de fundamentals en el mapping
                for standard_col, equivalents in self.column_mapping.items():
                    val = self._get_value(row, equivalents)
                    if pd.notna(val):
                        consolidated[period][standard_col] = val
        
        # 3. CONVERTIR A DATAFRAME Y CALCULAR RATIOS
        df = pd.DataFrame.from_dict(consolidated, orient='index')
        df.index.name = 'report_date'
        df = df.sort_index()
        
        # Asegurar que todas las columnas del mapping están presentes, aunque vacías
        for standard_col in self.column_mapping.keys():
            if standard_col not in df.columns:
                df[standard_col] = np.nan

        # Calcular ratios
        df = self._calculate_ratios(df)
        
        return df
    
    def _calculate_ratios(self, df):
        """Calcula ratios que se pueden calcular desde datos base"""
        
        # ROE: Net Income / Total Equity
        if 'net_income' in df.columns and 'total_equity' in df.columns:
            df['roe'] = df['net_income'] / df['total_equity']
        
        # ROA: Net Income / Total Assets
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa'] = df['net_income'] / df['total_assets']
        
        # Net Margin: Net Income / Revenue
        if 'net_income' in df.columns and 'revenue' in df.columns:
            df['net_margin'] = df['net_income'] / df['revenue']
        
        # Gross Margin: Gross Profit / Revenue
        if 'gross_profit' in df.columns and 'revenue' in df.columns:
            df['gross_margin'] = df['gross_profit'] / df['revenue']
        
        # Debt/Equity: Total Debt / Total Equity
        df['total_debt'] = df['long_term_debt'].fillna(0) + df['short_term_debt'].fillna(0)
        if 'total_equity' in df.columns:
            df['debt_equity'] = df['total_debt'] / df['total_equity']
        
        # Current Ratio: Current Assets / Current Liabilities
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            df['current_ratio'] = df['current_assets'] / df['current_liabilities']
        
        # Quick Ratio: (Current Assets - Inventory) / Current Liabilities
        # Sin datos de inventario disponibles, se omite para no duplicar current_ratio
        
        # ROI: Operating Income / Total Assets
        if 'operating_income' in df.columns and 'total_assets' in df.columns:
            df['roi'] = df['operating_income'] / df['total_assets']
        
        # ROIC: Operating Income / (Total Equity + Total Debt)
        if 'operating_income' in df.columns:
            df['roic'] = df['operating_income'] / (df['total_equity'].fillna(0) + df['total_debt'])
        
        # EPS: Net Income / Shares Diluted
        if 'net_income' in df.columns and 'shares_diluted' in df.columns:
            df['eps'] = df['net_income'] / df['shares_diluted']
        
        # EBITDA = Operating Income + Depreciation
        if 'operating_income' in df.columns and 'depreciation' in df.columns:
            df['ebitda'] = df['operating_income'] + df['depreciation']
        
        # FCF Margin: FCF / Revenue
        if 'fcf' in df.columns and 'revenue' in df.columns:
            df['fcf_margin'] = df['fcf'] / df['revenue']
        
        # Debt to EBITDA
        if 'ebitda' in df.columns:
            df['debt_to_ebitda'] = df['total_debt'] / df['ebitda']
        
        return df
    
    def process_all_tickers(self, tickers):
        """Procesa lista de tickers y genera CSVs consolidados"""
        output_path = self.data_path / "consolidated"
        output_path.mkdir(exist_ok=True, parents=True)
        
        for ticker in tickers:
            log.info(f"Procesando {ticker}...")
            try:
                df = self.consolidate_ticker(ticker)
                
                if len(df) == 0:
                    log.warning(f"  ✗ Sin datos para {ticker}")
                    continue
                
                # Limpiar columnas vacías
                df = df.dropna(how='all', axis=1)
                
                # Reordenar columnas: índice primero, luego datos base, luego ratios
                base_cols = ['revenue', 'net_income', 'operating_income', 'total_assets', 
                            'total_equity', 'total_debt', 'fcf', 'operating_cash_flow']
                ratio_cols = [col for col in df.columns if col not in base_cols]
                
                cols_ordered = [col for col in base_cols if col in df.columns] + ratio_cols
                df = df[cols_ordered]
                
                # Guardar
                output_file = output_path / f"{ticker}.csv"
                df.to_csv(output_file)
                log.info(f"  ✓ Guardado: {len(df)} períodos | Columnas: {len(df.columns)}")
                
            except Exception as e:
                log.error(f"  ✗ Error procesando {ticker}: {e}")
                import traceback
                traceback.print_exc()
        
        log.info(f"✓ Proceso completado")