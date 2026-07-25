# Informe acumulativo de resultados

## Estado

La arquitectura Model Study dispone de un smoke real completo y auditable. El Full Study científico
continúa pendiente; por tanto, no se extraen todavía conclusiones de inversión.

## Reglas de trazabilidad

Toda cifra futura deberá incluir:

- `study_id`;
- `winner_run_id`;
- hash del catálogo;
- hash del dataset;
- periodo;
- métrica y unidad;
- ruta del artefacto fuente;
- papel de la cifra: selección, estrés conocido o diagnóstico.

## Smoke de cinco tickers

Identidad:

- Study: `study-20260725-132255-c49da9ff`.
- Run de evidencia ganador: `run-578e46af72bc`.
- Dataset: `56aed7c788b723d39054b4a9bbb8bb1e982498350ec16c424c618e7689b81d99`.
- Universo: AAPL, MSFT, NVDA, JPM y XOM; SPY como benchmark.
- Periodo dev: 2016–2024.
- Duración de extremo a extremo: 16 minutos y 19 segundos.
- 27 runs físicos, todos `succeeded`; 53 eventos persistidos.
- Evidencia total del Study: 872.775 bytes.

**Limitación conocida de este smoke:** se ejecutó antes de la limpieza del 2026-07-25 que activó
de forma incondicional los factores de momentum multi-horizonte y medias móviles
(`mom_acceleration`, `mom_reversal_1m`, `ma_price_vs_sma6`, `ma_price_vs_sma12`,
`ma_distance_to_high12`), ya declarados en el catálogo de factores pero antes inalcanzables desde
cualquier Study real (ver `docs/bitacora.md`). Las cifras de este smoke no incluyen esas columnas.
Un nuevo Full Study las incluirá.

El ganador conservó horizonte 12 meses, lag PIT de 60 días y meta rolling acotado 10–50 %. Su
Rank-IC medio fue 0,0335 sobre 23 cohortes, con 56,5 % positivas. Ningún challenger fue elegible:
todos incumplieron el suelo de Rank-IC de al menos una era. El horizonte de seis meses alcanzó
Rank-IC medio 0,1849, pero se rechazó correctamente por dependencia temporal.

La robustez dev produjo un intervalo bootstrap al 95 % de [−0,0609; 0,0899] y permutación
`p = 0,345` con 199 iteraciones. Las semillas 7 y 2026 dieron Rank-IC 0,0765 y 0,0736. Los dos
placebos dieron 0,0257 y −0,1758. Las carteras aleatorias quedaron por debajo del modelo en esta
muestra, pero el tamaño transversal y las iteraciones reducidas impiden una interpretación
confirmatoria.

Los ocho perfiles y las nueve comparaciones de cartera se materializaron. La cartera balanced
obtuvo CAGR 38,7 % frente a 14,5 % de SPY, pero con turnover anualizado de 1.247 %. Estas cifras
son deliberadamente solo diagnósticas: cinco acciones, sesgo muestral y configuración dev hacen que
no sean evidencia de alfa generalizable.

Conclusión del smoke: el sistema ejecuta, selecciona por Rank-IC, conserva decisiones, publica todas
las vistas, reanuda sin duplicar y termina el worker. No confirma que la IA aprenda; sí demuestra
que el protocolo ya puede medirlo de forma no degenerada en un Full Study.

## Full Study

Pendiente de ejecutar después de validar el smoke. Las conclusiones distinguirán capacidad
predictiva, estabilidad estadística y traducción económica sin seleccionar retrospectivamente por
alfa.
