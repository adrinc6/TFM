# Model Study · TFM

Sistema *point-in-time* que estudia si cinco agentes especializados —**quality, value, growth,
momentum y risk**— aprenden a ordenar transversalmente las acciones del S&P 500 de forma que esa
ordenación mantenga capacidad predictiva **fuera de muestra**.

No es un buscador de la cartera más rentable del pasado. Es un experimento sobre si la señal
sobrevive fuera de la muestra en la que se eligió.

---

## La pregunta, y qué cuenta como respuesta

La variable central es el **Rank-IC**: la correlación de Spearman entre la puntuación que el sistema
produce en una fecha y el retorno que esa acción obtuvo después, medida solo cuando la etiqueta ya
está cerrada.

**Rank-IC es lo único que decide.** Alfa, Information Ratio, rentabilidad, turnover y perfiles se
calculan, se publican y se discuten, pero **ninguno eligió jamás una configuración**. Esa separación
es deliberada: mezclarlas llevaría a seleccionar reglas de cartera porque casualmente funcionaron
mejor en la historia conocida, que es exactamente el error que este diseño existe para evitar.

Las cifras concretas no están en este README ni en ningún otro documento: viven en los artefactos de
`results/studies/<study_id>/`, y [docs/results.md](docs/results.md) explica cómo leerlas.

---

## Qué tiene de particular

Si vas a mirar un solo repositorio de este tipo, estas son las decisiones que lo diferencian de un
backtest convencional:

- **Datos estrictamente point-in-time, con fechas reales de publicación.** Los fundamentales se
  incorporan según su `filingDate` real de la SEC (EDGAR, desde 1993) más un lag de ejecución. Usar
  la fecha del periodo fiscal —lo habitual en las APIs comerciales— sería lookahead puro.
- **Universo histórico del S&P 500, día a día.** Una señal fechada en 2000 solo ve los miembros del
  índice en 2000, con guarda contra símbolos reciclados.
- **El sesgo de supervivencia se mide y se publica, no se declara.** El proyecto distingue el sesgo
  de *composición* (resuelto) del de *cobertura de datos* (declarado, cuantificado y **no**
  resuelto), y documenta cómo leer los resultados a la luz de él.
- **Una sola frontera temporal.** 2024 separa lo que decide de lo que solo confirma. **2025–2026 es
  estrés reservado**: no participa en ninguna decisión, de ninguna pasada, y se evalúa exactamente
  una vez sobre el ganador ya congelado, publicando el resultado sea cual sea.
- **Catálogo cerrado de parámetros.** No se puede probar una configuración que el catálogo no
  declare. Es una defensa estructural contra el *p-hacking* por configuración libre.
- **Selección con puerta pareada y placebos de etiqueta.** Las comparaciones son cohorte a cohorte
  con bootstrap por bloques; los placebos comprueban que el sistema no «acierta» con etiquetas
  barajadas.
- **Umbrales de cartera económicos, no de percentil.** Una venta solo se emite si el destino del
  dinero es mejor que la posición **después de costes**. Un percentil no dice cuánto se espera ganar
  y por tanto no puede compararse contra lo que cuesta operar.
- **Corrección por multiplicidad.** Deflated Sharpe Ratio (Bailey y López de Prado) sobre las series
  de IC candidatas, porque con decenas de evaluaciones el mejor resultado es alto aunque ninguna
  configuración tenga capacidad real.

---

## El flujo

```text
main.py ingest
   │  Finnhub (fundamentales) + Yahoo (precios) + SEC EDGAR (fechas reales de publicación)
   ▼
data/prepared/<hash>/        panel point-in-time inmutable y compartido
   │
   ▼
cinco agentes especialistas  →  meta-agente causal  →  meta-score
   │
   ▼
Model Study: optimización secuencial por Rank-IC
   Temporal → Representación → Modelo → Meta → Ganador congelado
   │
   ├─ Robustez (bootstrap, eras, semillas, placebos)
   ├─ Atribución (factores, Newey-West, Deflated Sharpe, confirmación 2025-2026)
   └─ Ocho perfiles de inversor
   │
   ▼
Portfolio Study: rejilla cartesiana por Information Ratio
   │  (sin reentrenar; 2025-2026 no se calcula durante la rejilla)
   ▼
results/studies/<study_id>/  →  informe
```

---

## Arranque rápido

```powershell
python -m pip install -r requirements.txt

Copy-Item .env.example .env      # y poner FINNHUB_API_KEY y EDGAR_USER_AGENT

python main.py ingest            # descarga y consolida los datos crudos (horas)
python main.py serve             # API y dashboard
```

Abrir `http://127.0.0.1:8765/`. Los estudios se lanzan desde el dashboard, no desde la línea de
comandos.

Dos caminos cortos antes de esperar horas: `http://127.0.0.1:8765/dev` es un modo visual con
fixtures que no entrena nada, y `RUN_SCOPE=dev` ejecuta el flujo real con solo cinco tickers.

La guía completa —configuración, lanzar estudios, compilar el manuscrito y problemas frecuentes—
está en [docs/usage.md](docs/usage.md).

---

## El repositorio

| Directorio | Qué contiene |
|---|---|
| `main.py` | Único punto de entrada: `ingest` y `serve` |
| `environment.py` | Constantes científicas y `Settings` congelado y validado |
| `module/data` | Ingesta, universo histórico y panel point-in-time |
| `module/modeling` | Features, los cinco agentes y el meta-agente |
| `module/evaluation` | Cartera, backtest, perfiles y estadística |
| `module/research` | Robustez, atribución, capacidad y sensibilidad a costes |
| `module/studies` | Catálogo cerrado, runner, selección y Portfolio Study |
| `module/storage` | Datasets preparados, caché, evidencia y persistencia |
| `module/web` + `app` | API HTTP (biblioteca estándar) y dashboard sin build |
| `tests` | Tests de contrato; no necesitan datos ni credenciales |
| `latex` | Manuscrito, exportador de activos y compilación |
| `results/studies` | Los artefactos de los cuatro estudios del trabajo |

---

## El manuscrito

`latex/TFM.pdf` (la memoria) y `latex/TFM_ppt.pdf` (la defensa) **están versionados**: se pueden
leer directamente sin compilar nada ni ejecutar el sistema.

Para compilarlos: `python latex/build.py`, con interruptores al principio del fichero. Regenerar los
activos desde los artefactos solo funciona en la máquina donde se ejecutaron los estudios, porque los
`.parquet` de evidencia no están versionados.

---

## Documentación

| Documento | Para qué |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Qué hace cada pieza del código y **por qué** está decidida así |
| [docs/usage.md](docs/usage.md) | Instalar, configurar, ejecutar y compilar |
| [docs/results.md](docs/results.md) | Qué produce el sistema y cómo auditarlo |
| [latex/COMO_COMPILAR.md](latex/COMO_COMPILAR.md) | Detalle de la compilación y convenciones de escritura |

---

## Verificación

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

---

## Estado, límites y licencia

Este es un **Trabajo de Fin de Máster académico**. No es asesoramiento de inversión, ni un producto,
ni una recomendación de compra o venta de ningún valor. Los resultados históricos simulados no
garantizan resultados futuros.

Limitaciones que el propio trabajo declara:

- **Sesgo de supervivencia por cobertura de datos**: medido y publicado, no resuelto. Decae con el
  tiempo y por eso los resultados se leen era por era.
- **Las cifras dentro de la ventana de selección son una cota superior optimista**: el ganador es el
  mejor de un conjunto probado sobre los mismos datos. La era reservada 2025–2026 es la única medida
  libre de sesgo de selección.
- **Los `.parquet` de evidencia no están versionados** (`.gitignore`), así que un clon recibe las
  decisiones, los ganadores y los informes, pero no las series numéricas.
- **Reproducirlo de cero requiere una clave de Finnhub** y varias horas de ingesta.

No hay fichero de licencia: por defecto, todos los derechos reservados.
