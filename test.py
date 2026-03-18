# import os
# import requests
# from bs4 import BeautifulSoup
# from transformers import pipeline
# from pypdf import PdfReader

# def extraer_texto(entrada):
#     """Detecta si la entrada es un PDF, una URL o texto plano, y extrae el contenido."""
    
#     # 1. Comprobar si es un archivo PDF local
#     if entrada.lower().endswith('.pdf'):
#         if os.path.exists(entrada):
#             print(f"📄 Detectado archivo PDF en: {entrada}. Extrayendo texto...")
#             try:
#                 reader = PdfReader(entrada)
#                 texto = ""
#                 for page in reader.pages:
#                     extraido = page.extract_text()
#                     if extraido:
#                         texto += extraido + " "
                
#                 if not texto.strip():
#                     return "Error: El PDF está vacío o parece ser una imagen escaneada sin texto."
#                 return texto
#             except Exception as e:
#                 return f"Error al leer el PDF: {e}"
#         else:
#             return "Error: No se ha encontrado ningún archivo en esa ruta."

#     # 2. Comprobar si es una URL
#     elif entrada.startswith('http://') or entrada.startswith('https://'):
#         print("🌐 Detectada una URL. Extrayendo texto de la web...")
#         try:
#             headers = {'User-Agent': 'Mozilla/5.0'}
#             respuesta = requests.get(entrada, headers=headers)
#             respuesta.raise_for_status()
            
#             soup = BeautifulSoup(respuesta.content, 'html.parser')
#             parrafos = soup.find_all('p')
#             texto = ' '.join([p.get_text() for p in parrafos])
            
#             if not texto.strip():
#                 return "Error: No se pudo encontrar texto relevante en esta URL."
#             return texto
#         except Exception as e:
#             return f"Error al procesar la URL: {e}"
            
#     # 3. Si no es nada de lo anterior, asumimos que es texto plano
#     else:
#         print("📝 Detectado texto plano.")
#         return entrada


# def analizar_finbert(entrada):
#     """Analiza el texto por fragmentos usando FinBERT y calcula la media del sentimiento."""
#     texto = extraer_texto(entrada)
    
#     if texto.startswith("Error"):
#         print(texto)
#         return

#     print("Cargando el modelo FinBERT...")
#     analizador = pipeline(
#         "text-classification", 
#         model="ProsusAI/finbert", 
#         top_k=None
#     )

#     # DIVISIÓN DEL TEXTO (Chunking)
#     # FinBERT solo lee 512 tokens. Dividimos el texto en bloques de 400 palabras por seguridad.
#     palabras = texto.split()
#     tamano_bloque = 400
#     fragmentos = [' '.join(palabras[i:i + tamano_bloque]) for i in range(0, len(palabras), tamano_bloque)]
    
#     print(f"El texto ha sido dividido en {len(fragmentos)} bloque(s) para su análisis.")

#     # Diccionario para sumar las probabilidades de todos los bloques
#     totales_sentimiento = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}

#     # Analizamos cada fragmento
#     for i, fragmento in enumerate(fragmentos):
#         resultados = analizador(fragmento, truncation=True, max_length=512)
        
#         # Sumamos las probabilidades de este bloque a los totales
#         for resultado in resultados[0]:
#             etiqueta = resultado['label'].upper()
#             totales_sentimiento[etiqueta] += resultado['score']

#     print("\n" + "*"*50)
#     print("RESULTADOS GLOBALES DEL ANÁLISIS DE SENTIMIENTO:")
#     print("*"*50)
    
#     # Calculamos la media y printeamos
#     for etiqueta, suma_total in totales_sentimiento.items():
#         media_probabilidad = (suma_total / len(fragmentos)) * 100
#         print(f"-> {etiqueta}: {media_probabilidad:.2f}%")
#     print("*"*50 + "\n")

# # ==========================================
# # Zona de pruebas (Puedes cambiar esto)
# # ==========================================

# if __name__ == "__main__":
    
#     # 1. Prueba con URL
#     url_prueba = "https://finance.yahoo.com/news/stock-market-news-today-december-6-2023-131105436.html"
#     print("--- PRUEBA 1: URL ---")
#     # analizar_finbert(url_prueba)
    
#     # 2. Prueba con TEXTO PLANO
#     texto_prueba = "Revenue decreased by 20% compared to last year. The outlook remains uncertain."
#     print("--- PRUEBA 2: TEXTO PLANO ---")
#     # analizar_finbert(texto_prueba)
    
#     # 3. Prueba con PDF (¡Cambia la ruta por una real en tu ordenador!)
#     # Ejemplo en Windows: r"C:\Usuarios\TuNombre\Documentos\informe.pdf"
#     # Ejemplo en Mac/Linux: "/Users/TuNombre/Documentos/informe.pdf"
#     ruta_pdf_prueba = r"C:\Users\PC\Downloads\e361e58a-7483-44f5-bc62-a9080ae6ec72.pdf"
#     print("--- PRUEBA 3: ARCHIVO PDF ---")
#     analizar_finbert(ruta_pdf_prueba) # Descomenta esta línea cuando pongas una ruta válida


import os
import time
import glob
from datetime import datetime
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

def configurar_directorios(base_path, ticker, form_type):
    """Crea la estructura de carpetas solicitada si no existe."""
    ruta_limpia = os.path.join(base_path, ticker, form_type)
    os.makedirs(ruta_limpia, exist_ok=True)
    return ruta_limpia

def limpiar_html_a_texto(ruta_archivo_crudo, ruta_archivo_limpio):
    """Extrae el texto plano de un documento SEC eliminando etiquetas HTML/XML y ruido."""
    try:
        # Los archivos de la SEC pueden tener diferentes codificaciones, utf-8 suele funcionar
        with open(ruta_archivo_crudo, 'r', encoding='utf-8', errors='ignore') as f:
            contenido = f.read()

        # Usamos lxml porque es mucho más rápido que html.parser para archivos grandes (como un 10-K)
        soup = BeautifulSoup(contenido, 'lxml')
        
        # Extraemos el texto, separando los bloques con un espacio y eliminando espacios extra
        texto_plano = soup.get_text(separator=' ', strip=True)
        
        # Guardamos el resultado en un nuevo archivo .txt
        with open(ruta_archivo_limpio, 'w', encoding='utf-8') as f:
            f.write(texto_plano)
            
        return True
    except Exception as e:
        print(f"⚠️ Error al limpiar el archivo {ruta_archivo_crudo}: {e}")
        return False

def pipeline_descarga_sec(ticker, nombre_usuario, email_usuario, anios_atras=6):
    """Pipeline principal: Descarga reportes y extrae el texto para NLP."""
    
    # Configuración de fechas
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin.replace(year=fecha_fin.year - anios_atras)
    
    fecha_fin_str = fecha_fin.strftime("%Y-%m-%d")
    fecha_inicio_str = fecha_inicio.strftime("%Y-%m-%d")
    
    # Ruta base solicitada
    ruta_base = "data/sec/raw_filings"
    
    print(f"🚀 Iniciando pipeline para {ticker} ({fecha_inicio_str} a {fecha_fin_str})")
    
    # 1. INICIALIZAR EL DOWNLOADER (Cumplimiento estricto de la SEC)
    # sec-edgar-downloader maneja automáticamente el límite de 10 req/sec en sus peticiones internas.
    dl = Downloader(nombre_usuario, email_usuario, ruta_base)
    
    formularios = ["10-Q", "10-K"]
    
    for form in formularios:
        print(f"\n📥 Descargando formularios {form}...")
        try:
            # Descarga masiva
            dl.get(form, ticker, after=fecha_inicio_str, before=fecha_fin_str)
            print(f"✅ Descarga de {form} completada.")
        except Exception as e:
            print(f"❌ Error al descargar {form}: {e}")
            continue
            
        # 2. PROCESAMIENTO Y LIMPIEZA DE TEXTO (NLP Prep)
        print(f"🧹 Procesando y limpiando texto de los {form}...")
        
        # Ruta donde sec-edgar-downloader guarda los archivos crudos por defecto
        # Estructura: data/raw_filings/sec-edgar-filings/<TICKER>/<FORM>/<ACCESSION-NUMBER>/full-submission.txt
        ruta_crudos = os.path.join(ruta_base, "sec-edgar-filings", ticker, form, "*", "full-submission.txt")
        archivos_crudos = glob.glob(ruta_crudos)
        
        # Crear tu estructura de carpetas personalizada para los archivos limpios
        ruta_limpia_form = configurar_directorios(ruta_base, ticker, form)
        
        for archivo in archivos_crudos:
            # Extraemos el número de acceso (ID único del reporte) de la ruta del archivo
            accession_number = os.path.basename(os.path.dirname(archivo))
            archivo_destino = os.path.join(ruta_limpia_form, f"{ticker}_{form}_{accession_number}_clean.txt")
            
            # Si el archivo ya fue procesado antes, lo saltamos (Idempotencia)
            if os.path.exists(archivo_destino):
                continue
                
            exito = limpiar_html_a_texto(archivo, archivo_destino)
            
            if exito:
                print(f"   -> Guardado texto limpio: {archivo_destino}")
            
            # Pequeño retraso de cortesía para no saturar la CPU / Disco en procesamientos masivos
            time.sleep(0.5)

    print("\n🎉 Pipeline ETL completado con éxito. Tus datos están listos para FinBERT.")

# ==========================================
# Ejecución del Script
# ==========================================
if __name__ == "__main__":
    # ¡IMPORTANTE! Cambia esto por tus datos reales para evitar baneos de la SEC.
    MI_EMPRESA = "Universidad Politécnica de Madrid" 
    MI_CORREO = "adrian.nunez.costa@alumnos.upm.es"
    TICKER_OBJETIVO = "AAPL"
    
    pipeline_descarga_sec(
        ticker=TICKER_OBJETIVO,
        nombre_usuario=MI_EMPRESA,
        email_usuario=MI_CORREO,
        anios_atras=6
    )