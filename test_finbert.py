"""
SEC Pipeline — Versión Optimizada
==================================
Optimizaciones aplicadas:

  1. BATCH INFERENCE   FinBERT procesa N fragmentos a la vez en lugar de 1.
                       Speedup típico: x4-x10 en GPU, x2-x4 en CPU.

  2. FP16 (solo GPU)   Media precisión. ~2x más rápido en GPUs modernas
                       con pérdida de precisión despreciable en sentimiento.

  3. torch.compile     Fusión de kernels (PyTorch >= 2.0). Primera ejecución
                       lenta (compila JIT), las siguientes ~20-30% más rápidas.

  4. PARALELISMO I/O   Descarga y limpieza de varios tickers a la vez con
                       ThreadPoolExecutor (I/O bound → hilos suficientes).
                       FinBERT es secuencial para no saturar GPU/CPU.

  5. LIMPIEZA AUTOMÁTICA
                       Los raw se borran en cuanto el clean está generado.
                       Los clean se borran en cuanto el sentimiento está
                       calculado. Uso de disco siempre al mínimo.

Guía de BATCH_SIZE según hardware:
  GPU  8 GB VRAM  →  32
  GPU 16 GB VRAM  →  64
  CPU 16 GB RAM   →   8
  CPU  8 GB RAM   →   4
"""

import os
import re
import glob
import json
import shutil
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pandas as pd
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Rutas ──────────────────────────────────────────────────────────────────
DIR_RAW        = "data/sec/raw"
DIR_CLEAN      = "data/sec/clean"
DIR_SENTIMENTS = "data/sec/sentiments"

# ── Parámetros de rendimiento ──────────────────────────────────────────────
BATCH_SIZE     = 32    # fragmentos por batch (ajusta según HW)
USE_FP16       = True  # fp16 si hay GPU disponible
USE_COMPILE    = True  # torch.compile si PyTorch >= 2.0 y GPU disponible
MAX_WORKERS_IO = 4     # hilos paralelos para descarga y limpieza
BORRAR_RAW     = True  # borrar archivo crudo tras generar el clean
BORRAR_CLEAN   = True  # borrar archivo clean tras calcular sentimiento
FORMS          = ["8-K"]  # formularios a procesar — cámbialo aquí o en __main__


# ══════════════════════════════════════════════════════════════════════════
#  ESTADO / IDEMPOTENCIA
# ══════════════════════════════════════════════════════════════════════════

def ruta_estado(ticker):
    return os.path.join(DIR_SENTIMENTS, ticker, f"{ticker}_estado.json")

def cargar_estado(ticker):
    r = ruta_estado(ticker)
    if os.path.exists(r):
        with open(r, encoding="utf-8") as f:
            return json.load(f)
    return {"descargados": [], "limpiados": [], "analizados": []}

def guardar_estado(ticker, estado):
    r = ruta_estado(ticker)
    os.makedirs(os.path.dirname(r), exist_ok=True)
    with open(r, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
#  PASO 1 — DESCARGA  →  data/sec/raw/
# ══════════════════════════════════════════════════════════════════════════

def paso_descarga(ticker, nombre_usuario, email_usuario, anios_atras, estado):
    fecha_fin = datetime.now()
    fecha_ini = fecha_fin.replace(year=fecha_fin.year - anios_atras)
    dl = Downloader(nombre_usuario, email_usuario, DIR_RAW)

    for form in FORMS:
        print(f"  📥 [{ticker}] Descargando {form}...")
        try:
            dl.get(form, ticker,
                   after=fecha_ini.strftime("%Y-%m-%d"),
                   before=fecha_fin.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"  ❌ [{ticker}] Error en {form}: {e}")
            continue

        patron = os.path.join(DIR_RAW, "sec-edgar-filings",
                              ticker, form, "*", "full-submission.txt")
        for archivo in glob.glob(patron):
            accession = os.path.basename(os.path.dirname(archivo))
            clave = f"{form}_{accession}"
            if clave not in estado["descargados"]:
                estado["descargados"].append(clave)

    guardar_estado(ticker, estado)
    return estado


# ══════════════════════════════════════════════════════════════════════════
#  PASO 2 — LIMPIEZA  →  data/sec/clean/  +  borrado del raw
# ══════════════════════════════════════════════════════════════════════════

def limpiar_dirs_vacios(*rutas_raiz):
    """Elimina recursivamente cualquier directorio vacío bajo las rutas dadas."""
    for raiz in rutas_raiz:
        if not os.path.exists(raiz):
            continue
        for dirpath, dirnames, filenames in os.walk(raiz, topdown=False):
            if not dirnames and not filenames:
                try:
                    os.rmdir(dirpath)
                    print(f"🗑️  Dir vacío eliminado: {dirpath}")
                except OSError:
                    pass

def extraer_exhibit_991(contenido_sgml):
    """
    Extrae el bloque EX-99.1 del paquete SGML de un 8-K.
    El full-submission.txt es un fichero multi-documento con la estructura:
        <DOCUMENT>
        <TYPE>EX-99.1
        ...contenido HTML del press release...
        </DOCUMENT>
    Si no encuentra el exhibit, devuelve el contenido completo como fallback.
    """
    patron = re.compile(
        r"<DOCUMENT>\s*<TYPE>\s*EX-99\.1.*?</DOCUMENT>",
        re.IGNORECASE | re.DOTALL,
    )
    coincidencia = patron.search(contenido_sgml)
    if coincidencia:
        return coincidencia.group(0), True
    return contenido_sgml, False

def _limpiar_un_archivo(args):
    """
    Worker para el ThreadPoolExecutor.
    Para 8-K extrae primero el Exhibit 99.1 antes de parsear el HTML.
    Devuelve (clave, ruta_crudo, dir_accession, exito).
    """
    archivo_crudo, ruta_limpio, clave, dir_accession, form_type = args
    try:
        with open(archivo_crudo, "r", encoding="utf-8", errors="ignore") as f:
            contenido = f.read()

        # Para 8-K: quedarse solo con el Exhibit 99.1 (press release)
        exhibit_encontrado = False
        if "8-K" in form_type.upper():
            contenido, exhibit_encontrado = extraer_exhibit_991(contenido)
            if exhibit_encontrado:
                print(f"    📎 EX-99.1 extraído de {os.path.basename(archivo_crudo)}")
            else:
                print(f"    ⚠️  EX-99.1 no encontrado en {os.path.basename(archivo_crudo)}, usando doc completo")

        soup  = BeautifulSoup(contenido, "lxml")
        texto = soup.get_text(separator=" ", strip=True)
        os.makedirs(os.path.dirname(ruta_limpio), exist_ok=True)
        with open(ruta_limpio, "w", encoding="utf-8") as f:
            f.write(texto)
        return clave, archivo_crudo, dir_accession, True, exhibit_encontrado
    except Exception as e:
        print(f"  ⚠️  Error limpiando {archivo_crudo}: {e}")
        return clave, archivo_crudo, dir_accession, False, False

def paso_limpieza(ticker, estado):
    tareas = []

    for form in FORMS:
        patron = os.path.join(DIR_RAW, "sec-edgar-filings",
                              ticker, form, "*", "full-submission.txt")
        for archivo_crudo in glob.glob(patron):
            accession     = os.path.basename(os.path.dirname(archivo_crudo))
            clave         = f"{form}_{accession}"
            dir_accession = os.path.dirname(archivo_crudo)

            if clave in estado["limpiados"]:
                if BORRAR_RAW and os.path.exists(dir_accession):
                    shutil.rmtree(dir_accession, ignore_errors=True)
                continue

            ruta_limpio = os.path.join(DIR_CLEAN, ticker, form,
                                       f"{ticker}_{form}_{accession}_clean.txt")
            tareas.append((archivo_crudo, ruta_limpio, clave, dir_accession, form))

    if not tareas:
        print(f"  ⏭️  [{ticker}] Limpieza ya completada.")
        return estado

    print(f"  🧹 [{ticker}] Limpiando {len(tareas)} archivo(s) en paralelo...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as pool:
        for clave, archivo_crudo, dir_accession, exito, exhibit in pool.map(_limpiar_un_archivo, tareas):
            if exito:
                estado["limpiados"].append(clave)
                estado.setdefault("exhibit_991", {})[clave] = exhibit
                guardar_estado(ticker, estado)
                if BORRAR_RAW:
                    shutil.rmtree(dir_accession, ignore_errors=True)
                    print(f"    🗑️  Raw borrado: {dir_accession}")

    return estado


# ══════════════════════════════════════════════════════════════════════════
#  FINBERT — CARGA CON OPTIMIZACIONES
# ══════════════════════════════════════════════════════════════════════════

LABEL_MAP = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}

def cargar_finbert():
    """Carga FinBERT con fp16 y torch.compile opcionales."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️  Cargando FinBERT en {str(device).upper()}...")

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model     = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    if USE_FP16 and device.type == "cuda":
        model = model.half()
        print("   ✅ fp16 activado")

    model = model.to(device).eval()

    if USE_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("   ✅ torch.compile activado (1ª inferencia será lenta, las demás más rápidas)")
        except Exception:
            print("   ⚠️  torch.compile no disponible, continuando sin él")

    return model, tokenizer, device


@torch.inference_mode()
def inferir_batch(textos, model, tokenizer, device):
    """
    Devuelve las probabilidades (softmax) de cada clase para cada texto.
    Shape: list of dicts {"POSITIVE": float, "NEGATIVE": float, "NEUTRAL": float}
    """
    inputs = tokenizer(
        textos,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    probs    = torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist()
    id2label = model.config.id2label
    return [
        {LABEL_MAP[id2label[i].lower()]: p for i, p in enumerate(fila)}
        for fila in probs
    ]


# ══════════════════════════════════════════════════════════════════════════
#  PASO 3 — SENTIMIENTO  →  data/sec/sentiments/  +  borrado del clean
# ══════════════════════════════════════════════════════════════════════════

def extraer_mda(texto, form_type):
    """
    Extrae la sección relevante según el tipo de formulario:
      - 10-K: Item 7 (MD&A)
      - 10-Q: Item 2 (MD&A)
      - 8-K:  el texto ya es el EX-99.1 limpio, se usa entero
    """
    if "10-K" in form_type.upper():
        patron = re.compile(
            r"item\s+7\.(?:\s|&nbsp;)*management\'?s\s+discussion.*?item\s+8\.",
            re.IGNORECASE | re.DOTALL,
        )
    elif "10-Q" in form_type.upper():
        patron = re.compile(
            r"item\s+2\.(?:\s|&nbsp;)*management\'?s\s+discussion.*?item\s+3\.",
            re.IGNORECASE | re.DOTALL,
        )
    else:
        # 8-K u otros: el clean ya contiene solo el EX-99.1, se analiza entero
        return texto, True

    coincidencias = patron.findall(texto)
    if coincidencias:
        seccion = max(coincidencias, key=len)
        if len(seccion) > 5000:
            return seccion, True
    return texto, False


def analizar_archivo(ruta_txt, ticker, form_type, accession, model, tokenizer, device, exhibit_991):
    try:
        with open(ruta_txt, "r", encoding="utf-8") as f:
            texto_completo = f.read()

        if not texto_completo.strip():
            print("  ⚠️  Archivo vacío, saltando.")
            return None

        if "8-K" in form_type.upper() and not exhibit_991:
            print("  ⏭️  Sin EX-99.1, saltando.")
            # El clean no tiene valor, borrarlo para liberar disco
            if os.path.exists(ruta_txt):
                os.remove(ruta_txt)
                print(f"  🗑️  Clean sin exhibit borrado: {ruta_txt}")
            return None

        texto, mda_ok = extraer_mda(texto_completo, form_type)
        print(f"  {'🎯 MD&A extraído' if mda_ok else '⚠️  Texto completo (MD&A no encontrado)'}")

        # ~370 palabras ≈ 512 tokens con margen de seguridad
        palabras   = texto.split()
        fragmentos = [" ".join(palabras[i:i+370]) for i in range(0, len(palabras), 370)]
        total      = len(fragmentos)
        n_batches  = -(-total // BATCH_SIZE)  # ceil division
        print(f"  -> {total} fragmentos | {n_batches} batches (size={BATCH_SIZE})")

        # Acumular probabilidades brutas de cada bloque
        probs_acum = {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0}
        for i in range(0, total, BATCH_SIZE):
            batch = fragmentos[i : i + BATCH_SIZE]
            print(f"\r  -> Batch {i//BATCH_SIZE + 1}/{n_batches}...", end="", flush=True)
            for fila_probs in inferir_batch(batch, model, tokenizer, device):
                for label, p in fila_probs.items():
                    probs_acum[label] += p
        print()

        # Promediar sobre el total de fragmentos → score continuo por clase
        pos = round(probs_acum["POSITIVE"] / total, 4)
        neg = round(probs_acum["NEGATIVE"] / total, 4)
        neu = round(probs_acum["NEUTRAL"]  / total, 4)

        # Índice de polaridad continuo: rango (-1, +1)
        # Interpretación: +1 = puramente positivo, -1 = puramente negativo
        polaridad = round(pos - neg, 4)

        print(f"  ✅ P(pos):{pos:.3f}  P(neg):{neg:.3f}  P(neu):{neu:.3f}  →  Polaridad: {polaridad:+.3f}")

        return {
            "Ticker":            ticker,
            "Formulario":        form_type,
            "Accession_Number":  accession,
            "MDA_Extraido":      mda_ok,
            "P_Positivo":        pos,
            "P_Negativo":        neg,
            "P_Neutro":          neu,
            "Total_Bloques":     total,
            "Indice_Polaridad":  polaridad,
        }
    except Exception as e:
        print(f"\n  ❌ Error en {ruta_txt}: {e}")
        return None


def paso_sentimiento(ticker, estado, model, tokenizer, device):
    pendientes = []
    for form in FORMS:
        patron = os.path.join(DIR_CLEAN, ticker, form,
                              f"{ticker}_{form}_*_clean.txt")
        for ruta in glob.glob(patron):
            partes    = os.path.basename(ruta).replace("_clean.txt", "").split("_")
            accession = partes[2] if len(partes) > 2 else "desconocido"
            clave     = f"{form}_{accession}"
            if clave not in estado["analizados"]:
                pendientes.append((ruta, form, accession, clave))

    if not pendientes:
        print(f"  ⏭️  [{ticker}] Sentimiento ya calculado.")
        return estado

    dir_salida = os.path.join(DIR_SENTIMENTS, ticker)
    ruta_csv   = os.path.join(dir_salida, f"{ticker}_sentimientos.csv")
    os.makedirs(dir_salida, exist_ok=True)

    resultados = (
        pd.read_csv(ruta_csv, encoding="utf-8").to_dict("records")
        if os.path.exists(ruta_csv) else []
    )

    total = len(pendientes)
    for idx, (ruta, form, accession, clave) in enumerate(pendientes, 1):
        print(f"\n📊 [{idx}/{total}] {ticker} | {form} | {accession}")
        fila = analizar_archivo(ruta, ticker, form, accession, model, tokenizer, device,
                                exhibit_991=estado.get("exhibit_991", {}).get(clave, False))
        if fila:
            resultados.append(fila)
            estado["analizados"].append(clave)
            # Guardado parcial tras cada archivo: si el proceso peta, no se pierde nada
            pd.DataFrame(resultados).to_csv(ruta_csv, index=False, encoding="utf-8")
            guardar_estado(ticker, estado)
            print(f"  💾 CSV actualizado → {ruta_csv}")

            # Borrar el clean solo si NO tiene exhibit (los que sí tienen se conservan)
            if BORRAR_CLEAN and not estado.get("exhibit_991", {}).get(clave, False):
                if os.path.exists(ruta):
                    os.remove(ruta)
                    print(f"  🗑️  Clean borrado: {ruta}")

    return estado


# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════

def pipeline(tickers, nombre_usuario, email_usuario,
             anios_atras=6,
             hacer_descarga=True,
             hacer_limpieza=True,
             hacer_sentimiento=True):
    """
    Ejecuta los tres pasos para una lista de tickers.
      - Descarga y limpieza: en paralelo entre tickers (I/O bound).
      - Sentimiento: secuencial (un modelo en GPU/CPU a la vez).
    """

    # PASO 1 + 2 en paralelo entre tickers ─────────────────────────────────
    def _io_pipeline(ticker):
        estado = cargar_estado(ticker)
        if hacer_descarga:
            print(f"\n── DESCARGA [{ticker}] {'─'*40}")
            estado = paso_descarga(ticker, nombre_usuario, email_usuario,
                                   anios_atras, estado)
        if hacer_limpieza:
            print(f"\n── LIMPIEZA [{ticker}] {'─'*40}")
            estado = paso_limpieza(ticker, estado)
        return ticker, estado

    estados = {}
    if hacer_descarga or hacer_limpieza:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS_IO, len(tickers))) as pool:
            futures = {pool.submit(_io_pipeline, t): t for t in tickers}
            for fut in as_completed(futures):
                ticker, estado = fut.result()
                estados[ticker] = estado

    # PASO 3: secuencial (una sola instancia del modelo) ───────────────────
    if hacer_sentimiento:
        model, tokenizer, device = cargar_finbert()
        for ticker in tickers:
            estado = estados.get(ticker) or cargar_estado(ticker)
            print(f"\n── SENTIMIENTO [{ticker}] {'─'*38}")
            paso_sentimiento(ticker, estado, model, tokenizer, device)

    # Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  ✅ PIPELINE COMPLETADO")
    for ticker in tickers:
        csv    = os.path.join(DIR_SENTIMENTS, ticker, f"{ticker}_sentimientos.csv")
        existe = "✔" if os.path.exists(csv) else "✘"
        print(f"  {existe} {ticker:8s} → {csv}")
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MI_EMPRESA = "Universidad Politécnica de Madrid"
    MI_CORREO  = "adrian.nunez.costa@alumnos.upm.es"

    TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
    FORMS   = ["8-K"]   # sobrescribe el global para esta ejecución

    pipeline(
        tickers           = TICKERS,
        nombre_usuario    = MI_EMPRESA,
        email_usuario     = MI_CORREO,
        anios_atras       = 6,
        hacer_descarga    = True,
        hacer_limpieza    = True,
        hacer_sentimiento = True,
    )

    limpiar_dirs_vacios(DIR_RAW, DIR_CLEAN)