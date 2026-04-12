# LÓGICA Y PROCESO

## Cómo funciona el sistema multi-agente de selección de acciones

---

## El problema que resolvemos

Imagina que gestionas un fondo de inversión y necesitas seleccionar 10 acciones del S&P 500 cada trimestre. Tienes acceso a toda la información pública: estados financieros, precios históricos, recomendaciones de analistas, operaciones de insiders... pero son más de 500 empresas y cientos de variables por cada una. ¿Cómo decides de forma sistemática y repetible?

La inversión activa —es decir, seleccionar acciones específicas en lugar de comprar el índice completo— es uno de los problemas más difíciles de las finanzas. La mayoría de los gestores profesionales no consiguen superar al S&P 500 de forma consistente. Pero la evidencia académica muestra que ciertos patrones sí tienen capacidad predictiva: empresas con fundamentales sólidos tienden a hacerlo mejor, el momentum de precios persiste a corto plazo, y las acciones infravaloradas eventualmente convergen a su valor intrínseco.

El reto no es encontrar *un* patrón que funcione, sino integrar *múltiples señales complementarias* en una decisión coherente, sin caer en la trampa del sobreajuste (encontrar patrones en datos pasados que no se repiten en el futuro). Este sistema aborda exactamente ese reto.

---

## La idea central: un comité de analistas virtuales

El sistema funciona como un comité de inversión automatizado. En lugar de un modelo monolítico que intenta aprender todo a la vez, hay seis "analistas virtuales" (agentes), cada uno especializado en una dimensión del análisis financiero. Un séptimo componente —el meta-learner— actúa como el director del comité, ponderando las opiniones de cada analista para tomar la decisión final.

¿Por qué este enfoque? Por la misma razón por la que los fondos cuantitativos organizan sus equipos por especialidad: un analista que entiende profundamente los flujos de caja libres no necesita ser experto en análisis técnico, y viceversa. La diversificación de perspectivas reduce el riesgo de que un error en una dimensión domine la decisión.

---

## Los seis analistas y lo que cada uno "piensa"

### El Analista Fundamental

Este agente se pregunta: *"¿Es esta empresa financieramente sólida?"*

Examina la rentabilidad (ROE, ROA, márgenes), la evolución de los beneficios (crecimiento de ingresos, EPS), la calidad de los resultados (ratio de accruals, Piotroski F-Score) y las tendencias a largo plazo (pendiente de ROE y márgenes en los últimos 2-3 años). Utiliza un modelo XGBoost porque las interacciones entre variables fundamentales son cruciales: una empresa con alto ROE *y* bajo endeudamiento es diferente a una con alto ROE *por* alto apalancamiento.

### El Analista de Valoración

Este agente se pregunta: *"¿Está esta acción barata respecto a lo que vale?"*

Analiza múltiplos clásicos (P/E, P/B, EV/EBITDA) y los compara con sus medianas históricas de 5 años. Una acción que cotiza a 10× beneficios cuando su media histórica es 20× puede ser una oportunidad si la caída no está justificada por un deterioro fundamental. También incorpora yields (FCF yield, earnings yield) y sorpresas de beneficios.

### El Analista de Momentum

Este agente se pregunta: *"¿Tiene esta acción el viento a favor?"*

Analiza indicadores técnicos: RSI (para detectar sobrecompra/sobreventa), MACD (para identificar cambios de tendencia), medias móviles (SMA 20, 50, 200 para contexto de corto, medio y largo plazo) y el momentum puro de precios (retorno en 1, 3, 6 y 12 meses). También incorpora señales de momentum de beneficios: ¿la empresa está batiendo consistentemente las estimaciones de los analistas?

Lo interesante del momentum es que captura la inercia del mercado: las acciones que suben tienden a seguir subiendo a corto-medio plazo, y las que bajan tienden a seguir cayendo. Es un fenómeno bien documentado académicamente que complementa las dimensiones fundamentales.

### El Analista de Riesgo (Bear Agent)

Este agente se pregunta: *"¿Hay señales de alerta que los otros analistas podrían estar ignorando?"*

Funciona de forma única en el sistema: combina una capa de reglas explícitas con un modelo de machine learning. Las reglas son 10 señales de riesgo con umbrales concretos: deuda creciendo más del 20% anual, ratio deuda/EBITDA superior a 5×, flujo de caja libre negativo, pérdidas consecutivas, insiders vendiendo agresivamente, etc. Cada señal tiene un peso que refleja su gravedad.

El componente ML aprende patrones de riesgo más sutiles que las reglas no capturan. El score final es una media ponderada de ambas capas: 50% reglas, 50% ML.

Este agente produce un *score de riesgo* (no de oportunidad): un score alto significa "peligro". El pipeline invierte este score antes de pasarlo al meta-learner (`bear_score = 1 - bear_risk_score`), de modo que en el stacking final todos los scores siguen la misma convención: 1 = señal positiva. El meta-learner lo utiliza como filtro: si el riesgo es extremo (> 90%), la acción queda automáticamente excluida independientemente de lo que digan los otros agentes.

### El Analista de Sentimiento

Este agente se pregunta: *"¿Qué opinan los que saben más que el mercado?"*

Procesa dos fuentes de información privilegiada (no en sentido ilegal, sino de mayor profundidad):
- **Analistas de Wall Street**: ratio de recomendaciones de compra, dispersión de opiniones (cuando los analistas están muy divididos, hay incertidumbre), cambios recientes en el consenso.
- **Insiders**: ratio neto de compras/ventas de directivos en los últimos 90 días. Cuando los directivos compran acciones de su propia empresa, es una señal positiva significativa.

También incorpora el historial de sorpresas de EPS: empresas que baten consistentemente las estimaciones tienden a seguir haciéndolo.

### El Analista Sectorial

Este agente se pregunta: *"¿Va a hacerlo bien este sector en su conjunto?"*

Opera a un nivel más macro que los otros agentes. En lugar de evaluar empresas individuales, evalúa *sectores completos*. Agrega las medianas de las métricas fundamentales, de valoración y de momentum de todas las empresas de un sector, y predice si ese sector superará al S&P 500 en el próximo trimestre.

La lógica es top-down: si el sector tecnológico va a hacerlo bien, todas las acciones tecnológicas reciben un bonus; si el sector energético va a tener un trimestre difícil, incluso las mejores petroleras recibirán una penalización.

---

## Cómo el meta-learner combina las opiniones

El meta-learner recibe los seis scores (uno por agente, entre 0 y 1) y los combina en una decisión final. Pero no es una simple media: utiliza un *stacking* de dos modelos:

1. **Logistic Regression**: proporciona una combinación lineal interpretable. Sus coeficientes revelan directamente cuánto peso tiene cada agente.
2. **Gradient Boosting Machine**: captura interacciones no lineales. Por ejemplo, puede aprender que "alta calidad fundamental + infravaloración" es más que la suma de sus partes.

Los pesos de ambos modelos se calibran por rendimiento en validación cruzada temporal: si la LR tiene mejor AUC, recibe más peso, y viceversa.

Además, el meta-learner genera features de consenso: ¿cuántos agentes son alcistas? ¿Cuánta dispersión hay entre las opiniones? ¿La media de los scores base está por encima de 0.55? Estas señales de consenso ayudan a distinguir convicciones fuertes (todos los agentes de acuerdo) de convicciones débiles (opiniones divididas).

Finalmente, se aplican ajustes de robustez:
- **Dispersion shrink**: si un agente produce scores muy concentrados (todos los tickers con scores entre 0.48 y 0.52), su señal se contrae hacia 0.5 porque no es informativa.
- **Prior sectorial**: el score final se ajusta suavemente hacia la predicción del SectorRotationAgent, con una confianza proporcional al número de empresas en el sector.
- **Hard risk gate**: si el BearAgent detecta un riesgo extremo, el ticker queda excluido.

---

## Cómo se evita el sobreajuste: el walk-forward

El mayor riesgo en finanzas cuantitativas no es construir un modelo que funcione en el pasado —eso es trivial— sino construir uno que funcione *hacia adelante*. El backtesting convencional tiende a sobreestimar el rendimiento porque el investigador, consciente o inconscientemente, optimiza para los datos que ya conoce.

El walk-forward resuelve esto simulando exactamente cómo habría operado el sistema en tiempo real:

1. **Ventana de entrenamiento móvil**: en enero de 2020, el modelo solo conoce datos de 2010-2019. Entrena con esos datos.
2. **Evaluación estricta**: en el primer trimestre de 2020, el modelo selecciona sus Top 10 acciones. Se mide el retorno *real* de esas acciones.
3. **Avance temporal**: la ventana se desplaza un trimestre. En abril de 2020, el modelo se re-entrena con datos de 2010-2020Q1 y selecciona para Q2.
4. **Repetición**: este proceso se repite para cada trimestre del periodo de análisis.

Cada fold es independiente: el modelo nunca ve el futuro. Y hay capas adicionales de protección:

- **Snapshot lag de 60 días**: cuando analizamos Q1 (enero-marzo), el modelo no actúa el 31 de marzo, sino el 30 de mayo, simulando que los resultados financieros tardan en publicarse.
- **Filing date map**: las features fundamentales se alinean por la fecha en que el filing fue registrado en la SEC, no por la fecha del periodo contable.
- **OOF para el meta-learner**: los scores que usa el meta-learner para entrenarse se generan con validación cruzada temporal interna, no directamente con los scores de train (lo que sería leakage).
- **Política de features**: solo se permiten ratios y magnitudes normalizadas; valores absolutos como ingresos en dólares se bloquean porque su escala cambia con el tiempo y entre empresas de distinto tamaño.

---

## Las variantes del sistema

El pipeline es configurable en múltiples dimensiones:

### Frecuencia de análisis
- **Trimestral**: un fold cada 3 meses: rotación más frecuente, mayor coste de transacción, pero captura cambios rápidos.
- **Anual**: un fold por año, con un holding period de 12 meses. Menor rotación, menor coste, adecuado para un enfoque tipo Buffett.

### Universo de inversión
- **Dinámico** (recomendado): el universo de tickers se construye a partir del histórico de miembros del S&P 500, evitando el sesgo de supervivencia (incluir solo las empresas que *hoy* están en el índice).
- **Estático**: lista fija de ~500 tickers definida manualmente.
- **Top-N por market cap**: si se activa, solo se incluyen las N empresas de mayor capitalización por año.

### Modos de caché
- **Caché completa**: reutiliza el dataset maestro, los datos derivados del router y el resumen del walk-forward si la configuración no ha cambiado.
- **Sin caché**: recalcula todo desde cero (útil cuando se modifican features o datos).
- **Caché parcial**: reutiliza solo ciertos artefactos (dataset maestro pero no walk-forward, por ejemplo).

### Ponderación de cartera
- **Equiponderada**: todos los tickers seleccionados reciben el mismo peso.
- **Score-weighted**: el ticker con mayor score recibe aproximadamente el doble de peso que el de menor score, con una distribución lineal.

---

## Cómo interpretar la salida

Al finalizar una ejecución, el sistema produce:

### Métricas globales
Un resumen con Sharpe Ratio, retorno acumulado, maximum drawdown, Sortino, Calmar, y hit rate (porcentaje de folds rentables). Estos se comparan directamente con el benchmark (S&P 500) y las estrategias baseline.

### Detalle por fold
Para cada trimestre o año analizado, se reportan las acciones seleccionadas, sus scores por agente, y el retorno obtenido. Esto permite identificar en qué condiciones de mercado el sistema funciona mejor (por ejemplo, ¿es mejor en mercados laterales que en crashes?).

### Explicaciones por ticker
Para cada acción seleccionada, se genera una explicación textual basada en SHAP: *"AAPL fue seleccionada principalmente por su high ROE (SHAP +0.12), strong momentum_6m (SHAP +0.08) y low debt_to_ebitda (SHAP +0.05)"*. Esto permite al gestor evaluar si las razones del modelo son coherentes con su visión.

### Equity curve en USD
Si el backtest monetario está activado, se genera una curva de equity que muestra la evolución del capital invertido, incluyendo costes de transacción y slippage. El capital se encadena entre folds: si en Q1 el portfolio creció de 1000 USD a 1050 USD, en Q2 se invierte 1050 USD.

### Para tomar una decisión de inversión
Un usuario interpretaría los resultados de la siguiente manera:

1. **¿El sistema supera al benchmark?** Si el Sharpe Ratio es superior al S&P 500 en un periodo de 5+ años, hay evidencia de alpha.
2. **¿Es consistente?** Un hit rate del 60%+ y un maximum drawdown inferior al índice sugieren robustez.
3. **¿Las explicaciones son sensatas?** Si el último fold seleccionó una acción "porque tiene alta deuda" y el bear score es bajo, algo no cuadra.
4. **¿Supera al azar?** Si el rendimiento está por debajo del percentil 75 del baseline random, el modelo no aporta valor.
5. **¿Qué dicen los baselines de factor?** Si el sistema no supera al momentum puro de 12 meses, la complejidad adicional de los 6 agentes no está justificada.

La clave es que este sistema no es una caja negra: cada decisión puede rastrearse hasta los scores de los agentes individuales y, desde ahí, hasta las features financieras específicas que los motivaron. Esto lo hace no solo más confiable, sino también más útil como herramienta de apoyo a la decisión.

---

## Robustez y seguridad del pipeline

### Validación de entradas

El sistema valida todas las entradas externas antes de procesarlas:

- **Tickers**: el `DataRouter` aplica un filtro de formato (regex `[A-Za-z0-9.\-]{1,10}`) y verifica que la ruta resultante no escape del directorio de datos. Esto previene ataques de *path-traversal* donde un ticker malicioso como `../../etc/passwd` podría acceder a archivos fuera del directorio de datos.
- **Parámetros de configuración**: `environment.py` centraliza todos los parámetros y aplica tipos explícitos. Valores inválidos como una frecuencia de análisis desconocida provocan errores claros.

### Manejo de errores en OOF

La generación de scores Out-of-Fold captura excepciones por agente y split, registrando el error completo con traceback. En caso de fallo, el split afectado usa un score neutro de 0.5, evitando que un agente defectuoso corrompa todo el entrenamiento del meta-learner.

### Debug condicional

La exportación de datos de auditoría por ticker (e.g. inputs del SentimentAgent para AAPL) está controlada por el flag `DEBUG_EXPORT_AGENT_INPUTS`, desactivado por defecto. Esto evita I/O innecesario y posible contaminación del directorio de resultados en ejecuciones de producción.
