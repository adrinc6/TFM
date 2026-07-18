# Informe final — Sistema de IA para selección de acciones

> **Estado: pendiente de la ejecución en curso.** El estudio completo se está **reejecutando** con
> el orquestador de dos fases y la era reservada 2025-2026 (ver [docs/doc.md](doc.md) §5 y §8). Este
> informe se rellenará con las cifras finales cuando esa ejecución termine; hasta entonces no
> recoge resultados numéricos, para no presentar como definitivos los números de un run anterior.

## Qué contendrá este informe

Cuando el estudio cierre, este documento presentará el resultado en **dos planos separados**, sin
mezclarlos, coherente con el criterio de honestidad del proyecto:

1. **El aprendizaje (rank-IC OOS del `meta_final`).** La respuesta directa a la pregunta de
   investigación: ¿aprende el sistema a ordenar acciones de forma estable y significativa fuera de
   muestra? Se reportará el rank-IC puntual, su intervalo de confianza por **bootstrap por
   bloques**, el **p-valor del placebo** (permutación de etiquetas), la estabilidad
   **leave-one-year-out** y el rank-IC del finalista en la **era reservada 2025-2026** (donde nunca
   se optimizó). Un rank-IC no distinguible del azar es un resultado válido y así se reportará.

2. **La rentabilidad como consecuencia.** CAGR real de la cartera frente al SPY, beat rate,
   drawdown y turnover, **por perfil de inversor**, con la **guarda anti-artefactos** activa para
   que las cifras sean honestas (sin retornos mensuales imposibles como el +953 % espurio de 2010
   que se documentó en la bitácora). La rentabilidad nunca se usa como selector de configuración:
   la selección se hace por aprendizaje y estabilidad.

## Trazabilidad

Todas las cifras que aparezcan aquí procederán del manifiesto y los artefactos de
`results/studies/<study_id>/` y serán
reproducibles con un único comando (`RUN_MODE=full_study`, ver [docs/doc.md](doc.md) §9). El hilo
de decisiones que llevó a esta configuración está en [docs/bitacora.md](bitacora.md).
