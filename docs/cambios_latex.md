# Deuda con el manuscrito LaTeX

> El manuscrito (`latex/main.tex`, `latex/presentacion.tex`, `latex/assets/*.tex`) está **congelado
> entre migraciones**: los cambios de código no lo editan. Cada uno deja aquí una entrada con qué
> cambió, a qué capítulos, tablas o figuras afecta y qué artefacto lo respalda. Cuando se ordene la
> migración, este fichero es el contexto de partida.
>
> **Aquí no se copian cifras.** Las cifras viven en `results/studies/<study_id>/`; esta lista dice
> qué hay que ir a buscar y dónde ponerlo.

---

## Pendiente

### 2026-08-15 · La cadena vigente queda derogada

**Qué pasa.** El usuario va a relanzar la cadena entera (tres Model Studies + Portfolio Study) con
el panel corregido. Todas las cifras del manuscrito proceden de
`study-20260814-095144-5ec17b78` y `study-20260814-135754-fdbdf2c5`, y **ninguna sobrevive**: cambia
el `dataset_hash`, cambian los `study_id`, cambian los ganadores y cambian las ~175 cifras escritas
a mano en la prosa.

**Alcance.** Es una migración completa, no un retoque. El procedimiento está en `latex/plan_tfm.md`
y consiste en regenerar activos (`latex/scripts/export_study_assets.py --study-id <NUEVO>`),
sustituir identificadores (`latex/main.tex` macro `\studyid`, `latex/assets/a_reproducibilidad.tex`),
reescribir la prosa capítulo a capítulo y revalidar las cinco afirmaciones vertebradoras de
`t01_afirmaciones.tex`. Los activos huérfanos del study anterior hay que borrarlos a mano;
`latex/scripts/verify_latex_assets.py` los detecta.

---

### 2026-08-15 · Tabla de cobertura anual: dos columnas nuevas

**Artefacto**: `attribution.json` → `universe_coverage` (ahora con `sp500_members` y
`panel_coverage_fraction`).
**Afecta a**: `latex/assets/t03_cobertura_anual.tex` (regenerada por el exportador, ya modificado) y
la prosa de `latex/assets/03_datos_y_universo.tex` que la comenta.

La tabla publicaba «fracción utilizable» al 100 % —calidad **dentro** del panel— sin decir qué
fracción del índice llegaba a él. Ahora trae los miembros reales del S&P 500 por año y la cobertura
efectiva. La prosa debe explicar la diferencia entre ambas columnas: una mide calidad, la otra
alcance, y solo la segunda habla del sesgo de supervivencia.

---

### 2026-08-15 · Corregir la afirmación sobre el tamaño del universo ← **error factual**

**Afecta a**: `latex/assets/t09_limitaciones.tex:48` y `latex/assets/08_limitaciones.tex:53-57`.

La fila «Universo restringido» dice hoy: *«S&P 500 estadounidense; 278 tickers en 2003 frente a más
de 500 en años recientes»*. Insinúa que el índice creció, y **el S&P 500 tiene 500 miembros desde
1957**. Lo que la cifra describe es cobertura perdida del panel, no el tamaño del índice.

La fila debe pasar a llamarse **«Cobertura incompleta del universo»** y enunciarse como un defecto
medido, con la cobertura efectiva por año de la tabla anterior. Es la corrección más urgente de esta
lista: un tribunal que conozca el índice detecta el error de inmediato.

---

### 2026-08-15 · Auditoría de resolución de tickers: material nuevo

**Artefacto**: `universe_coverage.json` → `ticker_resolution` (bloque nuevo, se genera en la
ingesta).
**Encaje**: tabla nueva en `latex/assets/03_datos_y_universo.tex`.

Reparte el universo histórico entre el panel y cada motivo de exclusión (`recycled_ticker`,
`missing_price`, `missing_cik`, `missing_reports`, `no_metric_period_match`,
`missing_fundamentals`). Permite sustituir una disculpa por una medición y, sobre todo, **impide
presentar los símbolos no resueltos como mortalidad**: un cambio de ticker, una absorción o un
emisor extranjero producen el mismo síntoma que una quiebra.

Requiere una función `table` nueva en `latex/scripts/export_study_assets.py` con prefijo `tNN_`, o
el verificador la marcará como huérfana.

---

### 2026-08-15 · Emisores extranjeros: el panel cambia de tamaño

**Afecta a**: `latex/assets/03_datos_y_universo.tex` y a la limitación de cobertura.

`PERIODIC_FORMS` excluía 20-F y 40-F, de modo que las empresas extranjeras del índice quedaban sin
informes periódicos pese a tener CIK válido y cuentas publicadas. Corregido, junto con el filtro
`type=10-Q` del fallback `lookup_cik`, que repetía el mismo defecto. El manuscrito debe declarar el
cambio y comparar la cobertura antes y después: es una mejora del panel, no un cambio de método.

---

### 2026-08-15 · Versiones de catálogo heterogéneas: verificar si la limitación sigue existiendo

**Afecta a**: `latex/assets/t09_limitaciones.tex` (fila «Versiones de catálogo heterogéneas»),
`latex/assets/08_limitaciones.tex:106-108` y `latex/assets/t08_versiones_catalogo.tex`.

Si la cadena nueva corre entera bajo la misma versión de catálogo, **esta limitación desaparece** y
sus tres apariciones se retiran en vez de reescribirse.

Si por cualquier motivo volviera a haber versiones mezcladas, la redacción actual es imprecisa y hay
que corregirla: decir que el cambio «no altera ninguna medición» es cierto, pero incompleto. Un
cambio en la tabla de simplicidad **sí invierte decisiones tomadas por empate técnico** y, con
ellas, el baseline de las pasadas siguientes. Está documentado en `docs/metodologia.md` («Empates
técnicos y versión de catálogo») y verificado en el `decisions.json` de la cadena derogada, donde la
primera pasada eligió `execution_lag_days = 30` con regla `tie_simplicity`.

---

### 2026-08-15 · Tabla de sensibilidad a costes y rebaja de su limitación

**Artefacto**: `cost_sensitivity.json` (todavía no existe; se generará con el primer Portfolio Study
posterior al relanzamiento — diseño en `docs/plan_pendiente.md`, paso 2.1).
**Afecta a**: capítulo económico (`latex/assets/07_resultados_economicos.tex`) y la fila «Costes
constantes» de `latex/assets/t09_limitaciones.tex`, más su desarrollo en
`latex/assets/08_limitaciones.tex:61-65`.

**Tabla nueva**, con tres escenarios sobre la cartera adoptada: bruto (coste 0), estándar (5 + 10 pb)
y equilibrio, este último expresado en pb, en porcentaje y en ida y vuelta. La prosa debe enunciar el
resultado como *«el sistema bate al índice mientras operar cueste menos de X pb por operación»*, y
declarar las cuatro salvedades que el artefacto trae dentro: el escenario bruto no lo realiza nadie,
el equilibrio de ruta congelada es conservador, el exceso de la ventana de selección ya es una cota
superior optimista, y los costes nunca seleccionan.

**La fila de limitaciones cambia de naturaleza**: hoy dice «5 pb de comisión y 10 pb de slippage, sin
impacto de mercado ni capacidad, sobre un turnover del 324 %» sin ninguna cifra que la acote. Pasará
a declarar el margen medido. Si el equilibrio queda muy por encima de cualquier coste plausible para
gran capitalización —la estimación previa lo sitúa un orden de magnitud por encima—, la severidad
baja de Media a Baja y hay que decirlo, en vez de mantener una cautela que la evidencia no sostiene.

La prosa debe explicar además **por qué hay dos familias de cifras**: el coste entra en la
contabilidad y en los umbrales de decisión, así que «sin costes» no es la misma cartera sin
comisiones. Está desarrollado en `docs/metodologia.md`, «El coste entra dos veces».

---

## Aplicado

*(vacío: nada de lo anterior se ha llevado todavía al manuscrito)*
