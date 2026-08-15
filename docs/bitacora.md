# Bitácora

## 2026-08-15 · Reorganización documental y corrección del sesgo de cobertura

Dos frentes: qué documentos existen y dónde estaba realmente el sesgo de supervivencia.

### La documentación se reduce a cuatro ficheros

`docs/` pasa de seis documentos solapados a cuatro con papel disjunto: `metodologia.md` (el cómo,
en profundidad), `bitacora.md` (la agenda), `cambios_latex.md` (la deuda con el manuscrito) y
`plan_pendiente.md` (lo planificado y no hecho). Se borran `guia_decisiones.md`,
`gestion_cartera.md`, `informe_resultados.md` y `plan_estudios_encadenados.md`, absorbiendo en
metodología lo que debía sobrevivir: la estrategia de estudios encadenados, el Portfolio Study y la
doctrina operativa de cartera (orden de decisión, umbrales derivados y casuísticas).

**El manuscrito LaTeX queda congelado entre migraciones.** Un cambio de código ya no lo edita: deja
una entrada en `cambios_latex.md`. Mezclar ambos ritmos era lo que obligaba a retocar la prosa del
TFM cada vez que se tocaba una función.

**Las cifras dejan de vivir en documentos.** Viven en `results/studies/<study_id>/` y se leen de
ahí; toda afirmación numérica cita `study_id` y ruta. `informe_resultados.md` era precisamente una
segunda verdad que se desincronizaba —la auditoría del mismo día encontró en él cifras de un study
derogado—, así que la regla que obligaba a mantenerlo se sustituye por la que apunta al artefacto.

De paso se retira una contradicción antigua de `AGENTS.md`, que seguía describiendo una `cash_policy`
con valores `fully_invested`/`opportunity_cash` eliminada del catálogo, y cuatro referencias del
código a `docs/plan_fases.md`, un fichero que no existe —una de ellas dentro de un mensaje de error
que ve el usuario final—.

### El sesgo de supervivencia no estaba donde se creía

La composición del índice ya era point-in-time y se intentaban descargar los 1.206 tickers que
pertenecieron al S&P 500 alguna vez, con guarda contra símbolos reciclados. El sesgo entraba por la
**cobertura de datos**: el panel tenía muchos menos tickers que el índice en los años tempranos, y
el manuscrito lo describía como si el índice hubiera sido más pequeño entonces. **El S&P 500 tiene
500 miembros desde 1957**, así que esa diferencia es cobertura perdida, no un universo menor.

Tres correcciones:

1. **`_coverage_by_year` no tenía denominador.** Publicaba «fracción utilizable» —calidad dentro del
   panel— que puede valer 100 % mientras falta media lista del índice. Ahora compara contra
   `members_at()` y publica la cobertura efectiva. El exportador de LaTeX se actualiza para
   reflejarlo, pero **no se ejecuta**: queda anotado en `cambios_latex.md`.
2. **El 715 de 1.206 nunca se midió, se interpretó.** El docstring de `ticker_to_cik` afirmaba que
   los ausentes eran «en su mayoría quebrados o absorbidos». No hay evidencia de eso: un cambio de
   símbolo, una absorción o un emisor extranjero producen el mismo síntoma que una quiebra. Se añade
   `_ticker_resolution`, que reparte el universo por motivo de exclusión con precedencia fija —de
   modo que los recuentos suman el universo y ningún ticker se cuenta dos veces— y declara en el
   propio artefacto que `missing_cik` es una **cota superior** de la mortalidad.
3. **Dos filtros excluían a los emisores extranjeros, y uno tapaba al otro.** `PERIODIC_FORMS` solo
   admitía 10-K y 10-Q, así que una empresa extranjera del índice, con CIK válido y cuentas
   publicadas, quedaba sin ningún informe periódico. El fallback `lookup_cik`, que debía rescatar
   esos casos, consultaba con `type=10-Q` y repetía el mismo defecto. Se añaden 20-F y 40-F y se
   retira el filtro del fallback.

Ninguna de estas tres cosas es una decisión científica: son defectos de medición. El efecto real
sobre la cobertura solo se conocerá al reejecutar la ingesta, y esa comparación —antes y después—
es parte del plan.

### Validación

- Suite completa: 96 tests, todos superados, incluidos 7 nuevos de contrato de cobertura.
- `ruff check .`: 88 avisos antes del cambio y 88 después, idénticos salvo desplazamiento de línea.
  Ninguno nuevo. Son preexistentes y quedan como estaban.
- `node --check app/js/app.js` sin avisos.
- `grep` de los cuatro documentos borrados y de `plan_fases`: sin referencias vivas fuera de las
  menciones históricas de esta bitácora.

### Lo que queda

En `docs/plan_pendiente.md`. El siguiente paso lo ejecuta el usuario: relanzar la cadena con el
panel corregido y las tres pasadas bajo la misma versión de catálogo. Los diagnósticos de coste y
capacidad, y la lectura de la auditoría de tickers, van después.

## 2026-08-15 · Auditoría de trazabilidad y reorganización del LaTeX

Revisión completa de `main.tex` y `presentacion.tex` cotejando **cada cifra de la prosa contra la
tabla que la genera y contra los artefactos de `results/studies/`**. El linter
`verify_latex_assets.py` pasaba limpio: todo lo encontrado era semántico.

**Cifras que contradecían a sus propias tablas.** La más grave estaba en el desglose de órdenes del
capítulo 7, que citaba la tabla anterior al Portfolio Study —rotación 56,5 % del flujo, `rebalance`
20,2 %— cuando la vigente dice `rebalance` 38,2 % y rotación 37,0 %. No es un decimal: invertía el
diagnóstico del capítulo, que atribuía la rotación a re-decidir cada mes sobre señal a doce meses.
Con la tabla real hay **dos fuentes de tamaño comparable y naturaleza distinta**, cada una con su
palanca, y una de ellas —la corrección de deriva— es justo la variable que la rejilla considera casi
inerte para el IR. También: rango entre semillas 0,0020 → **0,0015** (el artefacto y la figura ya
decían 0,0015), placebos «dos órdenes de magnitud» → **un** orden, percentil aleatorio 0,761 → 76,1,
contratos «casi el 60 %» → 53 %, y la afirmación de que la cartera ganadora es «más concentrada» que
la del modelo, que es **falsa**: ambas tienen ocho posiciones, y lo que cambia es el efectivo, la
tenencia mínima y la deriva.

**Un solo relato de la cadena.** El capítulo 6 atribuía a la tercera pasada el cambio del meta a
`stacked_rolling_free`; los `winner.json` y `t06_cadena_config` lo sitúan en la **segunda**. Se
reescribe para que la tercera lo confirme, conservando la ventaja pareada de +0,00827 como lo que
es: la ratificación de una decisión anterior.

**Cifras sin origen.** Se declara de dónde sale el 0,1177 de la neutralización —mismas 117 cohortes,
subconjunto de filas con los catorce controles disponibles—, y se citan `attribution.json`,
`decisions.json` y `portfolio_winner.json` donde faltaba. Se añade lo que el artefacto dice y el
texto suavizaba: entre semillas el exceso geométrico **cruza cero** y `economic_conclusion_stable`
es `false`. Se explica que las diecisiete decisiones no cubren las veintiuna variables predictivas
porque cuatro corrieron en modo `fixed`, y por qué el Portfolio Study optimiza seis de las doce
variables de cartera. Los tres ejemplos con cifras inventadas pasan al entorno `ejemplo`. Entran en
bibliografía Grinold (1989), Newey y West (1987) y Bailey y López de Prado (2014), y el Deflated
Sharpe se reatribuye a este último.

**Reorganización.** La cadena de estudios sube del Anexo D al comienzo del capítulo 5, donde el
lector la necesita para entender qué es la pasada de referencia. El capítulo 8 deja de reenunciar
limitaciones que los capítulos 6 y 7 ya dan junto a su resultado. Los capítulos 2 y 3 dejan de abrir
con un resumen que adelanta todo y luego lo repite. El capítulo 1 pasa de cuatro enumeraciones
solapadas a dos objetivos y cinco preguntas. Cada explicación repetida —secuencial contra
cartesiano, serie recortada, escalera plana, trayectoria de pesos— se da entera en un sitio y se
remite desde los demás.

**Sobre la extensión y sobre dónde vive la trazabilidad.** La verificación de que cada cifra la
respaldan las ejecuciones es una exigencia sobre el trabajo, no algo que deba escribirse en el
manuscrito: el cuerpo y las diapositivas no dicen de qué fichero sale cada dato. Se retiró todo ese
aparato —leyendas de fuente, el macro `\fuente` de la presentación con sus trece llamadas, los
identificadores de estudio del cierre y las menciones a rutas en la prosa—, conservando íntegra la
explicación metodológica que las acompañaba, que es lo que el lector sí necesita: qué mide cada
columna, sobre qué ventana y con qué cartera. Los anexos de reproducibilidad y evidencia
complementaria se mantienen, porque existen precisamente para eso.

El balance de extensión queda en **−78 líneas de fuente, 12 figuras (antes 14) y 24 tablas (antes
25)**. Bajar más exigiría recortar evidencia, no redundancia.

**Pendiente.** Las secciones 2 y 3 de `docs/informe_resultados.md` describen el study derogado
`study-20260803-201234-b4d7a8d8` (rank-IC 0,1004, IC-IR 0,744, DSR 0,930), no la cadena vigente. Se
han corregido las tres tablas de robustez verificables contra JSON; el resto necesita su propia
pasada.

## 2026-08-15 · Presentación de defensa y reenfoque del manuscrito a dos objetivos

Dos trabajos encadenados, ambos de cómo se **cuenta** el TFM. Ninguna cifra cambia.

**La presentación de defensa.** No existía ninguna: `latex/presentacion.tex`, Beamer 16:9 con
XeLaTeX, quince diapositivas para diez minutos más once de reserva para el turno de preguntas, y
`latex/guion_defensa.md` con el guion hablado cronometrado. Vive junto a `main.tex` y no en una
subcarpeta, decisión deliberada: así las figuras se referencian con las mismas rutas `assets/` que
el manuscrito, el linter las valida sin excepciones y no se duplica ningún PNG. El tema reutiliza la
paleta de `export_study_assets.py`, que es la de las quince figuras, para que diapositivas y
gráficos parezcan una sola pieza. `verify_latex_assets.py` escanea ahora también la presentación.

**El reenfoque, que es el cambio de fondo.** El manuscrito decía en `09_conclusiones.tex` que «el
hallazgo central del trabajo, sin embargo, no está en el modelo sino en lo que ocurre al llevarlo a
una cartera», y `00_resumen.tex` y `plan_tfm.md` repetían esa jerarquía. El autor la corrige: el
trabajo son **dos objetivos sucesivos**, no un hallazgo único que desplaza al modelo.

1. **Que el sistema aprenda a ordenar** acciones fuera de muestra. Lo demuestran los tres Model
   Studies.
2. **Que las variables de cartera importen** y que optimizando por Information Ratio se construya
   una buena. Lo demuestra el Portfolio Study.

La clave es que durante los tres Model Studies **la cartera es secundaria porque no se había
optimizado todavía**: se mantuvo la configuración por defecto del catálogo precisamente para que
ninguna decisión predictiva pudiera apoyarse en ella. Con ese marco, el −11,29 % de la era reservada
deja de ser un giro incómodo y pasa a ser el punto de partida del Objetivo 2 y la medida de cuánto
depende el resultado de la gestión.

**El hueco real que se cierra no era de tono.** El Portfolio Study se incorporó el 2026-08-14 y
`01_introduccion.tex` nunca se actualizó: sus cinco objetivos operativos eran panel, agentes, meta,
selección y auditoría —todos del Objetivo 1— y H3 hablaba de «una traducción prudente de la señal a
cartera», no de optimizarla. Ahora hay **H1–H4** (H1 y H2 del Objetivo 1; H3, que las variables de
cartera producen diferencias materiales, y H4, que la cartera elegida conserva ventaja fuera de la
ventana, del Objetivo 2) y un **sexto objetivo operativo** para la rejilla cartesiana.

Ficheros tocados: `00_resumen.tex` (resumen y abstract, mantenidos equivalentes),
`01_introduccion.tex`, `07_resultados_economicos.tex`, `09_conclusiones.tex`,
`t01_afirmaciones.tex` (las cinco afirmaciones agrupadas 3 + 2 por objetivo) y `plan_tfm.md`.

**Lo que el reenfoque no tocó, a propósito.** Ninguna cifra, ningún matiz y ninguna limitación: se
conservan el Deflated Sharpe en 0,682, las seis cohortes reservadas, «la ganadora es la mejor de
1.728 evaluadas» y que `risk` por separado (0,1227) bate al meta (0,1090). La convención de que
ningún resultado favorable se presenta sin su salvedad se respeta sin excepción.

## 2026-08-15 · Erratas de cifra en el capítulo 6 y secciones 7–9 del informe

Corrección de las discrepancias detectadas al preparar la defensa. Todas verificadas contra
`robustness.json`, `winner.json`, `attribution.json` y `portfolio_winner.json` antes de tocar nada.

**En `06_resultados_predictivos.tex`, cuatro cifras y un signo.** El percentil de carteras
aleatorias de riesgo emparejado decía 97,4 cuando es **96,8**; el percentil 95 de CAGR del escenario
general decía 102,28 % cuando es **75,06 %**; y la frase que explicaba por qué ese escenario no es
informativo hablaba del «percentil 65» cuando el modelo queda en el **76,1**. La peor invertía un
signo de titular: «El meta queda en $-0{,}0119$, prácticamente en cero» en la era reservada, cuando
`evidence/summary.json` del ganador da **+0,0441**. El −0,0119 procedía de `evidence_baseline`, es
decir, del baseline y no del ganador, contra la regla de procedencia. La corrección además gana un
argumento: el meta aguanta positivo porque ya había concentrado en `risk`, mientras la ponderación
uniforme cae a −0,0735. También `growth` en esa era decía −0,1352 y la tabla generada da −0,1333, y
la configuración ganadora citaba «un máximo de doce por agente» cuando `winner.json` dice **20**.

**El origen del «doce» estaba en `docs/informe_resultados.md`, y era peor de lo que parecía.** El
aviso de derogación cubría las secciones 1 a 6, pero **las secciones 7, 8 y 9 estaban igual de
obsoletas y nadie las había marcado**, pese a ser las más citables del documento: la «configuración
ganadora» daba `max_features_per_agent` = 12, `lgbm_min_child_samples` = 50, `meta_method` =
`stacked_rolling_bounded` y una cartera de 12 posiciones con 25 % de efectivo —cuando el ganador
real es 20, 20, `stacked_rolling_free` y una cartera de 8 posiciones sin efectivo—, y «Qué se puede
afirmar hoy» estaba entero en cifras del study derogado (rank-IC 0,1004, DSR 0,930, era reservada
+14,21 %, transferencia 0,247). Las tres secciones se reescriben contra los artefactos vigentes, la
8 reorganizada además por los dos objetivos, y el aviso se amplía para declarar explícitamente qué
está actualizado y qué no.

Lección que conviene retener: **un aviso de derogación parcial es una trampa**. Marcar «las
secciones 1 a 6» dejó tres secciones sin marcar que parecían vigentes precisamente por no estar
marcadas, y de ahí se filtró una cifra al manuscrito.

**Adaptación al borrado de `f07_perfiles_tradeoff`.** La entrada siguiente elimina esa figura y la
sustituye por la explicación del mecanismo de los perfiles. La diapositiva de reserva que la usaba
pasa a mostrar `t08_perfiles_cartera` y a contar el hallazgo que la acompaña —el orden entre perfiles
se predice desde el Rank-IC de los agentes que cada uno pondera—, que además es mejor material de
defensa que el gráfico. El guion se actualiza en consecuencia.

## 2026-08-15 · Los perfiles de inversor, por fin explicados

El manuscrito presentaba una tabla con ocho perfiles —`garp`, `contrarian`, `defensive`…— y sus
resultados, pero **en ningún punto decía qué hace cada uno**. El lector veía que `momentum` obtiene
IR 0,017 y `defensive` 0,570 sin ninguna forma de saber por qué, ni qué distingue a uno de otro.

Se añade la explicación del mecanismo, transcrita de `PROFILE_WEIGHTS`: un perfil no es un modelo ni
una cartera distinta, sino una reordenación de la señal congelada en dos pasos —acota el universo al
percentil 60 del `meta_rank`, y reordena ese conjunto con pesos fijos sobre los rangos de los cinco
agentes—. Se documenta que los pesos pueden ser negativos (el `contrarian` apuesta contra el
momentum) y la convención de que en `risk` el rango alto significa *menos* riesgo. Nueva tabla
`t08_perfiles_def` con la hipótesis de estilo y los pesos de los ocho.

Con el mecanismo explícito aparece un hallazgo que la tabla de resultados ya contenía y nadie había
leído: **el orden entre perfiles se predice desde el Rank-IC de los agentes que cada uno pondera**.
El mejor de los siete que reordenan es `defensive` (IR 0,570), que carga 0,60 en `risk`, el agente
de mayor Rank-IC (0,1227); el peor es `momentum` (IR 0,017), que carga 0,75 en el agente de Rank-IC
0,0005 y además penaliza a `risk`. Los cuatro intermedios ponderan agentes de Rank-IC 0,01–0,02 y
quedan agrupados entre 0,112 y 0,312. Un perfil no añade información: cambia el peso relativo de la
que ya existe, y empeora en proporción a lo malo que sea el agente al que se lo desplaza.

Se elimina `f07_perfiles_tradeoff`: el turnover que mostraba ya está en la tabla, y el dato relevante
(`momentum` rota 5,98 veces al año, casi el doble que `balanced`, para exceso negativo) se dice ahora
en el texto.

**Metadiscurso podado.** Tres pasajes del capítulo 7 anunciaban lo que iban a decir antes de decirlo
o repetían la misma advertencia sobre las cajas marginales en tres párrafos. El contenido se
conserva; desaparece el rodeo.

## 2026-08-15 · Cifras falsas en la prosa, poda de 21 activos y regla de procedencia estricta

Auditoría a fondo del manuscrito con dos objetivos: bajarlo de ~90 a 60–80 páginas y hacerlo
defendible. El segundo resultó ser el urgente.

**La prosa contradecía a las tablas.** Seis cifras del texto no existían en ningún artefacto; eran
residuos de studies anteriores que sobrevivieron a las reescrituras. La peor invertía un signo: el
manuscrito afirmaba que el alfa factorial de la era reservada era «0,84 % con *t* = 2,53» ---
positivo y significativo --- cuando `attribution.json` da **−0,48 % con *t* = −3,50**, y construía
sobre ello un párrafo entero. Las demás: intervalo bootstrap `[0,0335; 0,1695]` frente al real
`[0,0425; 0,1723]`; exclusión de eras «0,0730–0,1262» frente a 0,0780–0,1350; placebos
«−0,0061/+0,0008» frente a −0,0081/+0,0013; Rank-IC de 2022–2024 citado como 0,1621 cuando la tabla
dice 0,1788; y una contradicción interna en el capítulo 6, que decía «61 %» en la línea 101 y «52 %»
en la 180 para la misma mejora. **El abstract inglés entero era de otro study** (Rank-IC 0.1004,
DSR 0.930, era reservada +14,21 %, transferencia 0.247) y contradecía al resumen español de su misma
página. Todo verificado contra los JSON y corregido; ahora un script contrasta las doce cifras clave
contra sus artefactos.

**`t09_limitaciones.tex` estaba roto en cuatro sitios** y nadie lo había visto porque es un fichero
manual, no generado: un `\t` comido por un escape renderizaba «extit{risk}» en el PDF; declaraba
«catálogo v5, código en v6» cuando los studies son v6/v6/v7; y mantenía como limitación vigente
«13 de 23 variantes del barrido superan al ganador», un barrido que el Portfolio Study sustituyó.
Reescrita entera. También el Anexo B decía «catorce de las diecinueve decisiones» cuando
`decisions.json` tiene **17** (15 por `robust_rank_ic`, 2 por `tie_simplicity`).

**Regla de procedencia aplicada sin excepciones.** La cadena y el Portfolio Study se explican como
procedimiento, pero las cifras salen solo de los ganadores: lo predictivo del Model Study 3, lo
económico de `evidence_best_full/`. La consecuencia fue eliminar contenido que describía la cartera
**descartada**: la sección «Transferencia» (25 líneas sobre el coeficiente 0,328 de la cartera por
defecto, siete de ellas explicando que la cifra no describe la adoptada), su figura, y la fila de
confirmación de la regresión factorial. Se conservan el Deflated Sharpe y las carteras aleatorias
porque miden el procedimiento de búsqueda, no una cartera concreta, y presentarlos con la cartera
menos favorable es conservador.

**Poda de activos: de 60 a 39.** Trece no estaban referenciados en ninguna parte del texto y varios
pares tabla+figura mostraban los mismos datos. Fuera 16 figuras y 5 tablas; con ellas, 17 funciones
y 2 parámetros muertos del generador, detectados por AST. Se conserva el único par cuyo texto
argumenta que la figura aporta algo que la tabla no puede dar (dispersión frente a mediana).

**Repetición eliminada.** El hallazgo «*risk* aislado bate al meta» se contaba tres veces en el
capítulo 6; la era reservada, tres veces en el 7, que además tenía tres finales. El capítulo 6
fundió dos secciones separadas por subsecciones intercaladas y el 7 quedó con un solo cierre.

Resultado: 4.019 → 3.697 líneas de fuente y 60 → 39 activos, sin perder ninguna afirmación ni
ninguna cifra correcta. Verificación: 12/12 cifras cuadran con los artefactos, cero refs rotas, cero
labels duplicados, cero activos huérfanos, cero autorreferencias de capítulo, cero duplicación
literal, sin código muerto, 89 tests.

## 2026-08-15 · Cierre de la reestructuración: cada cosa una sola vez y en su sitio

La reestructuración anterior movió los capítulos al orden correcto pero dejó las costuras a la
vista. Esta pasada las cierra. El criterio es que cada elemento aparezca **una sola vez**, en el
capítulo cuya pregunta responde, sin recortar profundidad: el manuscrito pasa de 3.892 a 4.019
líneas porque lo eliminado se sustituye por transiciones y análisis de procedencia.

**Duplicaciones literales eliminadas.** El capítulo económico repetía dos bloques enteros palabra
por palabra: los años adversos y la lectura de la era reservada aparecían en las líneas 404–428 y
otra vez en las 449–474. La era reservada llegó a contarse **tres veces**. El capítulo predictivo
repetía la neutralización de estilo completa (0,1177 → 0,1019 y el 86,62 %) en dos secciones
distintas. Se conserva en cada caso la versión más extensa y se traslada a ella cualquier detalle
exclusivo de la otra: el alfa de 0,27 % con *t* = 1,44, que solo estaba en la versión corta, se
integra en la desarrollada.

**Frontera entre capítulos.** El capítulo de arquitectura adelantaba resultados que aún no podían
justificarse —hiperparámetros ganadores, el meta finalmente elegido, el peso de `risk` por encima de
0,95—. Se sustituyen por la disyuntiva de catálogo que representan: `stacked_rolling_bounded` protege
una propiedad cualitativa del diseño que ninguna métrica mide, `stacked_rolling_free` maximiza la
que sí se mide. El ejemplo numérico se reescribe con un peso hipotético de 0,70 para ilustrar el
mecanismo sin revelar el desenlace. El capítulo de protocolo afirmaba que «ninguna cifra de
resultados aparece aquí» y contenía la traza completa de decisiones; ahora distingue entre trazas
auditables (que sí van ahí, porque sin ellas el protocolo no es comprobable) e interpretación (que
no). También presentaba una cartera de 12 posiciones con tope de efectivo del 25 % como si fuera la
adoptada, cuando la ganadora tiene 8 y cero efectivo: pasa a describir las seis variables y su
rejilla, sin valores.

**Tres defectos que impedían la lectura seguida.** El capítulo de resultados predictivos se
**citaba a sí mismo** tres veces («el Capítulo 6 comprobará…» dentro del capítulo 6). La
configuración ganadora no aparecía en ningún punto del cuerpo —las tablas de la cadena vivían solo
en el Anexo D—, de modo que el lector llegaba a los resultados sin saber qué se estaba evaluando;
`t06_cadena` y `t06_cadena_config` se trasladan al cuerpo y el anexo remite a ellas. Al hacerlo
aparecieron **labels duplicados** (`tab:cadena`, `tab:cadena-config` definidos en dos ficheros), que
habrían roto la compilación: se eliminan las copias del anexo.

**Procedencia verificada, no supuesta.** Se auditó el generador en lugar de confiar en los captions.
`f07_equity`, `f07_drawdown`, `f07_alfa_anual`, `f07_ordenes`, `t07_anual` y las dos figuras de
perfiles salen de `evidence_best_full/` —cartera ganadora— y sus captions lo declaran ahora de forma
uniforme. `f07_factores` y `f07_transferencia` salen de `attribution.json`, que se calculó sobre la
cartera del Model Study y no puede regenerarse: se marcan **expresamente como diagnósticas**. El
texto que rodea a `f07_transferencia` explicaba una sola fuente de discrepancia con la tabla anual
(la agregación) cuando hay dos: también cambia la cartera.

**Dato corregido.** La sección de rotación citaba un turnover del 359 % que no existe en ningún
artefacto: es un residuo de la cartera anterior. El valor de la ventana de selección es 324,43 % y
el de la curva completa 333,35 %, ambos verificados contra `evidence_best_full/summary.json`.

**Código muerto.** El parámetro `profiles` de `write_tables` se leía del disco y se pasaba por dos
funciones sin usarse desde que la tabla de perfiles se emite del Portfolio Study. Eliminado.

Verificación: `verify_latex_assets.py` correcto, cero labels duplicados, cero referencias rotas,
cero mojibake, 89 tests en verde. La validación visual del PDF queda pendiente de Overleaf.

## 2026-08-14 · Reestructuración narrativa del manuscrito

El documento se leía en zigzag: la historia del desarrollo interrumpía dos veces la explicación del
sistema, los resultados predictivos y económicos convivían en un único capítulo de 761 líneas, y la
arquitectura de los agentes venía mezclada con sus resultados —que sólo se pueden justificar después
de explicar el protocolo de selección—.

**Orden nuevo**: problema → datos observables → sistema → protocolo → resultados predictivos →
resultados económicos → límites → conclusión.

| Antes | Ahora |
|---|---|
| `03_datos` + `04_diseno_metodologico` | `03_datos_y_universo` (universo, fuentes, PIT, features) |
| `06_agentes` (arquitectura + resultados) | `04_agentes_y_meta_agente` (sólo arquitectura, 112 líneas) |
| `07_diseno_experimental` + diseño del Portfolio Study | `05_protocolo_experimental` (303 líneas) |
| `09_resultados` (761 líneas, dos preguntas mezcladas) | `06_resultados_predictivos` (583) + `07_resultados_economicos` (501) |
| `05_desarrollo_metodo` + `08_desarrollo_cartera` | **Anexo D** `d_auditoria_desarrollo` (373 líneas, íntegro) |

**Decisiones de fondo:**

- **El Anexo D conserva las dos historias completas**, no un resumen. Cambian de lugar, no de
  contenido: recorrer cronológicamente errores ya corregidos interrumpe la explicación del sistema
  final, pero omitirlos sería peor porque muchas reglas del protocolo existen porque algo falló.
- **La tabla de las cinco afirmaciones se traslada a las conclusiones.** La introducción plantea las
  cinco *preguntas*; las respuestas llegan cuando ya hay evidencia detrás. Antes el documento
  revelaba el desenlace en la página 2.
- **Los capítulos 4 y 5 no contienen ni una cifra de resultados.** Describen mecanismo y regla de
  decisión; su evidencia vive en los capítulos 6 y 7. Cada uno anuncia explícitamente dónde se
  analiza lo que describe.
- **El capítulo 7 abre con una advertencia epistemológica**: la cartera se eligió por IR sobre
  2015-2024, de modo que sus cifras de selección describen la configuración elegida pero no son
  confirmación independiente. La única lectura económica reservada es 2025-2026.

**El documento no encogió**: 3.160 → 3.892 líneas (+23 %), por las transiciones nuevas y el material
añadido hoy. Ningún bloque de análisis se eliminó; sólo se eliminó duplicación literal tras las
migraciones, sustituida por referencias cruzadas.

**Verificación**: `verify_latex_assets` correcto, 0 etiquetas duplicadas, 0 referencias sin destino,
entornos balanceados, sin mojibake y sin capítulos huérfanos. Las etiquetas `chap:metodologia`,
`chap:desarrollo-metodo` y `chap:desarrollo-cartera` se conservan o redirigen al Anexo D para no
romper las referencias existentes.

## 2026-08-14 · Cartera equivocada en tres activos, y tres tablas vacías sustituidas

### El error: la tabla de las tres ventanas describía la cartera del modelo

`t07_selec_conf_full` («Las tres ventanas de evaluación») afirmaba que en 2025-2026 la cartera perdía
un 11,29 % con IR −1,167 y batía al índice el **0 %** de los años. Eso es la cartera del Model
Study, no la ganadora del Portfolio Study, así que el documento **se contradecía a sí mismo** en
cuatro capítulos.

Causa: `write_tables_robustness` construía sus bloques desde `summary["summary"]`, es decir
`evidence/summary.json`. Nunca recibió el Portfolio Study. Corregido con un parámetro `portfolio`
que redirige a `evidence_best_full/summary.json`, que tiene la misma forma (`summary`,
`confirmation`, `full_curve`).

| Métrica | Antes (modelo) | Ahora (ganadora) |
|---|---|---|
| Exceso, selección | 2,61 % | **6,97 %** |
| Exceso, reservada | −11,29 % | **+2,56 %** |
| IR, reservada | −1,167 | **+0,304** |
| Años que baten, reservada | 0 % | **50 %** |

Al arreglarlo cayó otra afirmación falsa: la prosa decía que «el efectivo medio pasa de 9,08 % a
25,04 %» y que el mejor resultado coincidía con la mayor posición de liquidez. La cartera ganadora
**no sostiene efectivo** (0 % en las tres ventanas), así que su exceso no puede atribuirse a haber
estado fuera del mercado. Reescrito.

### Perfiles: había dos tablas con ganadores distintos

`t07_perfiles` (cartera del modelo) y `t08_perfiles_cartera` (cartera ganadora) coexistían, y **no
dan el mismo orden**:

| Perfil | Cartera modelo | Cartera ganadora |
|---|---|---|
| `value` | **0,340 (1.º)** | 0,190 (5.º) |
| `balanced` | 0,339 (2.º) | **0,844 (1.º)** |
| `defensive` | 0,259 | 0,570 (2.º) |

Con la cartera vieja gana `value` por 0,001 —el «empate por ruido» que el manuscrito describía—;
con la buena, `balanced` domina por margen amplio. **La afirmación 5 se sostiene mejor de lo que
decía el documento.** Decisión del autor: presentar **sólo la cartera ganadora**. Eliminado
`t07_perfiles.tex` y regeneradas las dos figuras desde `portfolio_profiles.parquet`.

### Tablas vacías sustituidas por evidencia real

**La escalera de decisiones era una columna de ceros.** 17 filas con Rank-IC 0,1005 y ventaja
0,0000, salvo la última. No era un error: es lo esperable de la última pasada de una cadena
convergida, porque cada variable llega ya en su mejor valor. Pero enseñaba la columna equivocada.

`t06_decisiones` se reescribe como **«qué se rechazó y cuánto costaba»**, ordenada por coste:
neutralizar por sector −0,0429, pesos de recencia −0,0373, horizonte de 6 meses −0,0322, preset
`core` −0,0290… hasta un grupo con costes por debajo de 0,005 (todos los hiperparámetros de
LightGBM) que demuestra que el sistema no debe su capacidad al ajuste fino. Y marca los dos empates
técnicos: en `lgbm_min_child_samples` la alternativa medía **+0,0004 mejor** que el ganador.

Eliminados `t06_escalera.tex` y `f06_escalera.png` (eran casi la misma tabla que `t06_decisiones`,
ambas llenas de ceros) y con ellos `decision_records` y `draw_decision_ladder`.

**El top-20 de la rejilla tampoco discriminaba** (IR de 0,844 a 0,703, 20 variaciones de la misma
cartera). Sustituido por `t08_cartera_influencia`: cuánto separa cada variable el IR.

| Variable | Diferencia mejor-peor |
|---|---|
| Posiciones objetivo | **0,300** |
| Tope de efectivo | **0,202** |
| Suelo de cobertura | 0,107 |
| Tenencia mínima | 0,039 |
| Tolerancia de deriva | 0,016 |
| Reparto de pesos | **0,009** |

Contiene además un argumento que faltaba: el mejor `target_size` **marginal** es 5, pero la ganadora
usa **8**. Optimizar variable a variable no habría encontrado la ganadora — que es justo por qué la
búsqueda es cartesiana y no secuencial.

### Lo que no se puede arreglar y se declara

`t07_factores`, `f07_transferencia` y el Deflated Sharpe salen de `attribution.json`, calculado
sobre la cartera del Model Study. El Portfolio Study **no recalcula** atribución factorial ni
robustez: su `evidence_best_full/` sólo tiene equity, órdenes, posiciones y métricas anuales. La
corrección es de honestidad: cada caption declara ahora que esas cifras corresponden a la cartera
del Model Study.

**Regla que queda fijada**: todo activo económico se lee de `evidence_best_full/` o de
`portfolio_profiles.parquet`; lo predictivo, del Model Study; lo que no se pueda regenerar, se
declara en el caption.

## 2026-08-14 · Explicabilidad: `risk` no es baja volatilidad, y la salvaguarda nunca se activó

### Por qué domina `risk` (la pregunta que las conclusiones dejaban abierta)

`agent_local_attribution.parquet` (1,3 M filas) estaba persistido pero sin explotar. Agregado,
responde a lo que el manuscrito declaraba como su resultado menos explicado:

| Variable de `risk` | Contrib. media | Encabeza |
|---|---|---|
| `gap_21d` | 0,0216 | **51,2 %** |
| `range_63d` | 0,0136 | 26,8 % |
| `max_drawdown_252d` | 0,0092 | 4,2 % |

**No es la prima clásica de baja volatilidad.** Mandan la microestructura de precio a corto plazo
—huecos de apertura y rango de negociación, 78 % de las observaciones entre las dos—, mientras que
`beta_252d`, el candidato obvio, es el séptimo (2,9 %). Esto explica mecánicamente por qué la
neutralización por 14 controles de estilo retiene el 86,62 %: si el agente dominante replicara el
factor de volatilidad, debería destruir mucho más.

**Segundo hallazgo**: la variable que gobierna a `risk` cambia con el régimen. `range_63d` domina
2017-2020 (incluido el desplome de 2020) y luego se desvanece; `gap_21d` crece hasta encabezar el
64-77 % desde 2023. Aunque el meta concentre >95 % del peso en `risk`, ese agente **no es una
apuesta fija**: reasigna atención por dentro. Suaviza —sin eliminarla— la objeción de depender de un
único especialista.

Añadido también que `quality` encabeza con **29 variables distintas** frente a 10-12 del resto, y es
de los peores por Rank-IC (0,0096). Se presenta como asociación sugerente, no como causalidad: son
cinco agentes, no una muestra.

### Corrección: la salvaguarda de la curva de alfa no se activa nunca

El manuscrito afirmaba en tres sitios que la salvaguarda «se activa en **4.384 de 20.545 filas**,
algo más de una de cada cinco», y lo declaraba amenaza a la validez con severidad Media. Era cifra
del study derogado. En el study 3, `signal_calibration.parquet` reparte sus 60.380 filas en
`horizon` 43.639 (80 %) y `era` 10.905 (20 %); las 5.836 filas `none` son **sólo de 2015-2016** y
tienen alfa nulo — el arranque sin cohortes cerradas, no la salvaguarda.

Es decir: **no se activa ni una vez**. Corregido en `08_desarrollo_cartera.tex`,
`10_limitaciones.tex` y `t09_limitaciones.tex`, rebajando la severidad a Baja. La crítica de diseño
se conserva (un supuesto *a priori* que se dispararía justo cuando la evidencia dice que no hay
señal sigue siendo mala idea), pero pasa a ser rama muerta y no defecto medido.

### Decisiones de presentación

- **Sección, no capítulo.** Va en `06_agentes_y_meta_agente.tex` §«Qué mira realmente cada agente»,
  donde *cierra* un argumento abierto. Un capítulo propio quedaría descolgado: el material es
  descriptivo, no aporta evidencia sobre las cinco afirmaciones y prometería más de lo que demuestra.
- **No se codifica el signo** de la contribución en figura ni tabla: la media con signo es de orden
  10⁻⁴ (se cancela entre acciones) y colorear por ella sugeriría un sesgo direccional inexistente.
  Se sustituyó por «Encabeza» (fracción de veces que la variable es la primera), que sí es legible.
- **`signal_calibration` descartado** como sección: el alfa esperado medio es negativo en 7 de 11
  años mientras la cartera batía al índice. No es contradicción —es calibración de percentil a
  retorno, no predicción de cartera— pero explicarlo cuesta más de lo que aporta y se presta a
  malinterpretación en una defensa. Queda documentado en el anexo.

Activos nuevos: `f05_atribucion_risk.png`, `f05_atribucion_anual.png`, `t05_atribucion.tex`.
`load_agent_attribution` lee el parquet **una sola vez** y devuelve tres agregados pequeños (0,9 s).

## 2026-08-14 · El TFM pasa a la cadena de 4 estudios; la cartera decide el signo del resultado

### Qué se ha hecho

El manuscrito LaTeX documentaba `study-20260803-201234-b4d7a8d8` (catálogo v5, ~175 cifras a mano).
Se ha trasladado entero a la cadena vigente: tres Model Studies encadenados
(`1b104667` → `aa733655` → `5ec17b78`) más el Portfolio Study `fdbdf2c5`. **No queda ni una cifra
del study anterior en `latex/`**, verificado por grep.

La regla dura de `latex/plan_tfm.md` («el TFM se redacta exclusivamente con este study») exigía
registrar cualquier cambio allí y aquí. Queda registrado: el study antiguo sale del documento.

### El hallazgo que cambia la tesis

La cadena mejora monótonamente **dentro** de la ventana de selección (IR 0,189 → 0,294 → 0,339) y se
degrada monótonamente **fuera** (IR +0,898 → +0,476 → **−1,167**). Con la cartera del catálogo, el
study 3 —el mejor de la cadena en selección— perdía un 11,29 % frente al índice en la era reservada
sin batirla ni un año. Es la firma clásica del sobreajuste por búsqueda.

**Pero la degradación no venía del modelo.** Con la cartera ganadora del Portfolio Study —misma
señal, mismos scores congelados, sin reentrenar nada— la era reservada da **+2,56 % de exceso e IR
+0,304**, batiendo al índice 1 de 2 años. El Rank-IC de esa era es **+0,0441 en ambos casos**,
porque no depende de la cartera: la ordenación nunca se rompió, lo que fallaba era su traducción a
posiciones.

La tesis del TFM pasa a ser eso: en un sistema de esta clase la construcción de cartera no es un
detalle de implementación, es una variable capaz de decidir el signo del resultado fuera de muestra.
Cautelas declaradas en el texto: 6 cohortes cerradas, ~1,41 años de cartera, y la ganadora es la
mejor de 1.728.

### Otros resultados que obligaron a matizar el manuscrito

- **`risk` solo (0,1227) bate al meta (0,1090)** en la ventana de selección, y el meta —ya con
  `stacked_rolling_free`, sin tope— acaba asignándole **más del 95 %** del peso. La arquitectura
  multi-agente se matiza en lugar de darse por demostrada.
- **Deflated Sharpe baja a 0,682**. Encadenar estudios compra Rank-IC y encarece cualquier
  afirmación de rentabilidad ajustada por selección. Se reporta como el contraste no superado.
- **Sólo 1 de 17 decisiones** desplazó al incumbente en el study 3 (escalera plana en 0,1005 hasta
  `meta_method`): es lo que se espera de la última pasada de una cadena convergida.
- **2 decisiones se resolvieron por `tie_simplicity`** y se declara explícitamente que sus valores
  no son hallazgos empíricos.
- La cadena no corrió entera bajo el mismo catálogo: studies 1 y 2 con v6, study 3 con **v7**.

### Conflicto metodológico resuelto

El manuscrito argumentaba **en contra** del cartesiano (`07_diseno_experimental.tex`) y registraba
como limitación no haber optimizado la cartera (`09_resultados.tex`). Se distinguen los dos planos:
el argumento secuencial se mantiene para lo **predictivo** (donde la búsqueda infla el Rank-IC que
el trabajo reporta) y el cartesiano se justifica para la **cartera** (que no toca el Rank-IC y cuya
rejilla se calcula sobre scores recortados en 2024, de modo que la era reservada no llega a
calcularse). La vieja limitación desaparece y la sustituye una de multiplicidad.

### Panel

Cada perfil y la cartera ganadora tienen botón **«Ver run»** con las mismas vistas que un run normal
(Rendimiento y Cartera: snapshots, posiciones con `$$CASH$$`, órdenes, efectivo). Fuentes nuevas
`portfolio-winner` y `portfolio-profile:<nombre>` en [module/web/queries.py](../module/web/queries.py),
confinadas bajo `profiles/`. No se ofrecen Aprendizaje ni Acciones porque un perfil no reentrena el
modelo y esos artefactos no existen.

### Limpieza para Overleaf

Eliminados por obsoletos: `f07_barrido_cartera.png`, `t07_cartera_barrido.tex`,
`draw_portfolio_sweep`, `write_tables_portfolio` y `latex/scripts/__pycache__`. `plan_tfm.md` pasa
de 497 a 185 líneas: su guion capítulo a capítulo reproducía cifras del study derogado, y mantener
dos copias garantiza que una quede obsoleta. `latex/` queda sin activos huérfanos.

`export_study_assets.py` acepta ahora `--chain-study-id` (repetible) y `--portfolio-study-id`; el
manifiesto separa fuentes predictivas de económicas.

## 2026-08-14 · La escritura atómica reintenta ante bloqueos transitorios (WinError 5)

### Problema

Un Portfolio Study abortó a las **7 de 1.728 combinaciones** con
`PermissionError: [WinError 5] Acceso denegado` al renombrar `.study.json.<rnd>.tmp` sobre
`study.json`. No es un problema de permisos: en Windows un antivirus o el indexador pueden retener
el fichero destino unas decenas de milisegundos, y `os.replace` falla. Ya había ocurrido antes en
un Model Study, pero ahí era raro; el Portfolio Study lo convierte en casi seguro porque escribía
el estado **en cada combinación**, es decir 1.728 oportunidades de tropezar.

### Solución, en dos frentes

1. **Reintento en `write_json`** ([module/common/utils.py](../module/common/utils.py)):
   `_replace_with_retry` reintenta el rename hasta 6 veces con espera creciente (50 ms, 100 ms…).
   Si el bloqueo es permanente el error se propaga: no se silencia nada.
2. **Menos escrituras** ([module/studies/portfolio_study.py](../module/studies/portfolio_study.py)):
   el estado se persiste cada `STATUS_EVERY = 10` combinaciones en vez de en todas. Con ~6 s por
   combinación eso refresca el panel cada minuto, resolución de sobra, y reduce la exposición en un
   90 %.

Las dos son necesarias: la primera hace el fallo recuperable, la segunda lo hace raro.

### Verificación

`pytest` 87/87, con dos tests nuevos en `tests/test_workflow_contract.py`: uno comprueba que un
bloqueo transitorio se supera reintentando y no deja temporales huérfanos; el otro, que un bloqueo
permanente sigue propagando la excepción.

## 2026-08-14 · Portfolio Study: reanudación incremental y perfiles con la cartera ganadora

### Decisión

Dos añadidos al Portfolio Study, ambos previos a lanzar la rejilla definitiva:

1. **Reanudación real.** La rejilla vuelca `portfolio_grid.parquet` cada 25 combinaciones y, al
   arrancar, salta las que ya figuran en él reconstruyendo el mejor vigente.
2. **Los ocho perfiles de inversor se evalúan con la cartera ganadora**, al terminar la rejilla, en
   `portfolio_profiles.parquet`.

### Motivo

**La reanudación era un agujero real**: `WORKERS.pause` mata el proceso y `/resume` relanza el
worker, pero `run_portfolio_study` recorría la rejilla desde el principio. A diferencia del Model
Study —que reutiliza runs por `logical_key`— la rejilla no persiste runs individuales, así que
pausar tiraba todo el cómputo. Se comprobó en la práctica: el primer lanzamiento se detuvo en
412/1728 y, al no existir todavía el volcado periódico, esas ~40 minutos se perdieron.

**Los perfiles quedan fuera del cartesiano** por una razón de método, no de coste. `apply_profile`
([module/evaluation/profiles.py](../module/evaluation/profiles.py)) **reordena la señal**:
sustituye el `meta_rank` y recalibra el alfa. Las seis variables de la rejilla, en cambio, solo
gestionan la cartera ya elegida. Son dos planos distintos y el TFM los mantiene separados.
Incluirlos multiplicaría la rejilla por ocho (13.824 combinaciones, ~23 h) y, sobre todo,
**elegiría el estilo de inversor por su rentabilidad conocida**: en el study 3, `value` (IR 0,3405)
y `balanced` (0,3389) se separan por 0,0016, que es ruido. Los perfiles responden a otra pregunta
—«cómo le habría ido a cada tipo de inversor»— y siguen siendo diagnóstico informativo.

### Detalles

- `combination_key` serializa en el orden fijo de `PORTFOLIO_STUDY_VARIABLES`, no en el de
  inserción del diccionario, para que la identidad de una combinación no dependa de cómo se
  construyó.
- `_resume` empieza de cero si el parquet está corrupto —volcado interrumpido a medias—: es
  preferible repetir trabajo a arrastrar un estado inconsistente.
- Cada perfil se evalúa **dos veces**, igual que el ganador: sobre la serie recortada en 2024 para
  las cifras de selección y sobre la completa para la era reservada. La comparación entre perfiles
  nunca se contamina con 2025-2026.
- Un perfil que exija rangos de agentes ausentes se registra como `applicable=False` en vez de
  romper el estudio.
- Vista `portfolio-profiles` en la API y tabla nueva en la página del Portfolio Study, con el aviso
  explícito de que ningún perfil se elige por su IR.

### Verificación

`pytest` 85/85 (10 en `tests/test_portfolio_study_contract.py`, incluidos uno que comprueba que la
reanudación no repite combinaciones, otro que `combination_key` no depende del orden y otro que los
perfiles usan la cartera ganadora contra las dos series), `ruff` y `node --check` limpios.

### Pendiente

El usuario reanuda la rejilla. Después: fases 1 y 2 de la actualización del TFM, ya planificadas.

## 2026-08-14 · Portfolio Study: segundo optimizador, por Information Ratio

### Decisión

Se añade un **segundo tipo de estudio** que no toca el modelo: parte de un Model Study ganador,
reutiliza sus scores congelados y recorre el **producto cartesiano** de las seis variables de
cartera que gobiernan el riesgo, eligiendo por **Information Ratio**. Vive en
[module/studies/portfolio_study.py](../module/studies/portfolio_study.py) y se lanza desde un
conmutador «Modelo / Cartera» en Inicio.

### Motivo

El Model Study optimiza Rank-IC, que mide la calidad de la **ordenación**. Pero se observó que
Rank-IC y rentabilidad pueden divergir: en la cadena de studies hubo un ganador con el mejor
Rank-IC y a la vez el peor coeficiente de transferencia. Optimizar la ordenación no optimiza la
cartera, así que la cartera necesita su propio criterio. El IR lo es porque premia el exceso **por
unidad de riesgo**, no el alfa bruto.

### Diseño

- **Cartesiano y no greedy**: las seis variables interactúan (el suelo de diversificación sale de
  `target_size` **y** `max_cash_weight`; el efecto de `coverage_percentile_floor` depende de si la
  plaza liberada se recompra o queda en efectivo). Un recorrido greedy no ve esas interacciones.
- **Barato**: cada combinación reutiliza los scores del ganador vía `run_profile_evaluation` y solo
  rehace el backtest. Medido: **~5,2 s por combinación**, frente a los ~146 s de un run predictivo.
  Las 1.728 combinaciones de la rejilla completa son ~2,5 h, no 70.
- **Retención (regla 5)**: cada combinación deja una fila de resumen en `portfolio_grid.parquet`;
  la evidencia completa es **solo la del mejor vigente** en `evidence_best/`, que se sustituye
  cuando otra combinación lo supera. Al terminar queda exactamente una carpeta.
- **Qué no se optimiza**: `commission_bps` y `slippage_bps` se fijan a un único valor porque son
  *supuestos de coste*, no decisiones —optimizarlos sería elegir el mundo en el que la estrategia
  luce mejor—; las `price_only_*` y los umbrales en pb gobiernan cuándo se opera bajo información
  incompleta y se estresan aparte. La validación lo impone: exactamente un valor cada una.
- **2025-2026 no se calcula durante la rejilla**: el backtest de cada combinación se **corta en
  2024** recortando los scores antes de simular (`selection_evidence`). No basta con filtrar el
  resumen al elegir: la cartera es secuencial, así que si la simulación entra en la era reservada su
  resultado existe, y basta con mirarlo para caer en la tentación de elegir por él. Cortando la
  serie, ese resultado **no se ha calculado**. Solo la combinación ya ganadora se reevalúa una vez
  sobre la serie completa, y esa evidencia se guarda aparte en `evidence_best_full/`. La cartera de
  partida contra la que se mide la mejora usa la **misma** serie recortada; compararla sobre la
  completa mediría ventanas distintas y la mejora sería ficticia.

### Riesgo declarado

Optimizar por IR sobre la ventana de selección **añade multiplicidad**: 1.728 carteras probadas
sobre los mismos datos. Es exactamente el riesgo que la era reservada existe para detectar, y el
manuscrito debe reportar el IR de 2025-2026 de la cartera elegida junto al de selección, nunca solo
el segundo.

### Verificación

`pytest` 82/82 (7 tests nuevos en `tests/test_portfolio_study_contract.py`, incluidos uno que
comprueba que la evidencia conservada es la del mejor y no la de la última evaluada, y otro que
verifica que la rejilla no puede ver la era reservada), `ruff`, `node --check`.

Prueba de extremo a extremo con rejilla reducida: el ganador (`target_size=8`, `max_cash_weight=0`)
da **IR 0,4455 en la ventana de selección** frente a 0,3389 de la cartera del modelo (**+0,107**), y
la evidencia de la rejilla termina en 2024 con **cero filas** de 2025-2026, mientras la del ganador
llega a 2026.

**Y el aislamiento demostró servir para algo**: ese mismo ganador obtiene **IR −0,508 en la era
reservada**. La cartera que mejor rinde en la ventana con la que se elige es peor fuera de ella, que
es exactamente el sobreajuste que la separación existe para detectar. El manuscrito debe reportar
las dos cifras juntas, nunca solo la de selección.

## 2026-08-14 · Cadena de tres studies completada; el study 3 pasa a ser la referencia del TFM

### Resultado

La cadena de tres Model Studies encadenados (§ `docs/plan_estudios_encadenados.md`) terminó. **El
study de referencia del TFM pasa a ser `study-20260813-232458-05b4d236`** (ganador
`run-d304f6074665`), el más optimizado. Los tres corrieron bajo catálogo v6 y son comparables.

| Métrica | Study 1 | Study 2 | Study 3 |
|---|---|---|---|
| Rank-IC medio (selección) | 0,1000 | 0,1074 | **0,1090** |
| IC-IR | 0,735 | 0,835 | **0,851** |
| Cohortes positivas | 70,94 % | 74,36 % | 74,36 % |
| Coeficiente de transferencia | 0,178 | 0,234 | **0,049** |
| Information Ratio | 0,189 | 0,294 | **0,121** |
| Deflated Sharpe | 0,844 | 0,867 | **0,584** |

### Por qué se encadenó

La optimización es **greedy secuencial**, así que el ganador depende del punto de partida: cada
variable se evalúa sobre el incumbent acumulado, no sobre todas las combinaciones posibles. Usar el
ganador de cada pasada como baseline de la siguiente es un ascenso por coordenadas, y se hizo para
intentar alcanzar una configuración mejor que la que alcanza una sola pasada desde el baseline del
catálogo.

### Qué demuestra

**Funciona en la métrica de selección**: Rank-IC sube de forma monótona (+9,0 % acumulado) e IC-IR
de 0,735 a 0,851. Y **converge**: solo cambia una variable por pasada (`meta_method` en la 2.ª,
`execution_lag_days` en la 3.ª) y las otras 19 se mantienen, señal de que el óptimo greedy es
estable frente al punto de partida.

**No se traduce en economía, y hay que decirlo**: con el mejor Rank-IC de las tres, el study 3 tiene
el peor coeficiente de transferencia (0,049 frente a 0,234) y peor IR (0,121 frente a 0,294). La
tercera pasada ordena mejor el universo y a la vez convierte peor esa ordenación en rentabilidad.
Es evidencia directa de que el cuello de botella está en la traducción señal → cartera, no en la
capacidad predictiva. El Deflated Sharpe también empeora (0,584), consecuencia esperada de la
multiplicidad que introduce encadenar pasadas.

### Consecuencias para el manuscrito

- La quinta afirmación vertebradora («el perfil `balanced` es el mejor») **es falsa**: no gana en
  ninguna pasada. El mejor perfil es `defensive` en 1 y 2, y `value` en la 3.
- El agente `risk` por sí solo (Rank-IC 0,1197) **sigue superando al meta-agente** (0,1058).
- El contraste de carteras aleatorias no se supera en el escenario general (percentil 56,7 en el
  study 3); sí en el emparejado por riesgo (92,4), aunque por debajo del umbral de 0,95.

## 2026-08-14 · `execution_lag_days`: se invierte la tabla de simplicidad (catálogo v7)

### Decisión

`simplicity` de `execution_lag_days` pasa de `(30, 45, 60)` a **`(60, 45, 30)`**, de modo que en un
empate técnico gana el **lag mayor**. `CATALOG_VERSION` sube de 6 a **7**.

### Motivo

El desempate por simplicidad (`tie_simplicity` en
[module/studies/selection.py](../module/studies/selection.py)) elegía sistemáticamente el lag
**menor** cuando la evidencia no distinguía. Pero un lag menor no es una hipótesis más simple sino
**más fuerte**: afirma que el fundamental ya estaba disponible para operar antes. La tabla estaba
premiando, en caso de duda, la opción con más riesgo de lookahead — justo lo contrario de la regla 2
de `CLAUDE.md` y del criterio con el que el catálogo recomienda 60.

Se auditaron las 21 variables predictivas: **es el único caso** con este defecto de validez. En las
de LightGBM (`lgbm_max_depth`, `lgbm_n_estimators`, `lgbm_learning_rate`, `lgbm_min_child_samples`)
el orden es correcto —menos capacidad es genuinamente más simple— y en `market_regime_feature`,
`fundamental_momentum`, `feature_weighting_mode` o `snapshot_step_months` el ganador de empates no
coincide con el `recommended`, pero eso es la tensión normal entre punto de partida y desempate, no
un defecto.

### Alcance real

Afecta a decisiones frágiles: en el study 1 se resolvieron **4 de 19** decisiones por
`tie_simplicity` (una con ventaja de +0,00001, ruido puro), 1 en el study 2 y 2 en el study 3. En
concreto, el lag ganador **tanto del study 1 (30) como del study 3 (60) salió de esta regla**, no de
la evidencia: la ventaja pareada entre ambos nunca superó `TIE_TOLERANCE` (0,002). El TFM debe
declarar que la evidencia no distingue entre 30 y 60 días, en vez de presentar el lag final como un
hallazgo medido.

La cadena de tres studies es **v6** y el código vigente es **v7**; el manuscrito debe citar la
versión de catálogo junto al `study_id`.

## 2026-08-13 · Estrategia: estudios encadenados (el ganador de uno es el baseline del siguiente)

### Decisión

Encadenar tres Model Studies: el ganador de cada uno se marca como `baseline` de todas las
variables del siguiente. El TFM se redactará sobre el **último** de la cadena, y la progresión
entre los tres se contará como evidencia de que el procedimiento converge. Plan completo en
[docs/plan_estudios_encadenados.md](plan_estudios_encadenados.md).

### Motivo

La optimización es greedy secuencial, así que **el ganador depende del punto de partida**: cada
variable se evalúa sobre el incumbent acumulado, no sobre todas las combinaciones. Encadenar es
un ascenso por coordenadas — cada pasada completa es una iteración.

El study 1 (`study-20260812-163136-1b104667`, catálogo v6, ganador `run-6eaa47a0597b`) lo pedía:
su configuración ganadora se aparta del baseline del catálogo en **8 de 21 variables
predictivas**, y **4 de 19 decisiones se resolvieron por `tie_simplicity`** (empate estadístico:
`execution_lag_days`, `market_regime_feature`, `lgbm_learning_rate`, `lgbm_min_child_samples`).
Un empate significa que la ventaja no se distinguía del ruido *con el incumbent de ese momento*;
reevaluadas desde un baseline mejor pueden resolverse de otra forma.

### Cómo se implementa

No requiere código nuevo: `normalized_definition`
([module/studies/config.py:41-74](../module/studies/config.py)) ya admite `baseline` por variable
—exigiendo que sea uno de los `values` seleccionados— e `initial_values` (línea 173) siembra con
él la evaluación `predictive:baseline`. Encadenar es lanzar el study siguiente marcando como
baseline el valor ganador del anterior.

### Riesgo declarado

Encadenar pasadas multiplica las configuraciones probadas sobre los mismos datos y **agrava la
selección múltiple**. El Deflated Sharpe ya lo penaliza vía `n_trials` (74 en el study 1). El
capítulo de limitaciones debe declarar que la ganancia de Rank-IC entre pasadas y el riesgo de
sobreajuste crecen a la vez, y que la ventana reservada 2025-2026 —que no participa en ninguna
pasada— es la única defensa real.

### Hallazgos del study 1 que obligarán a reescribir el manuscrito

Medidos sobre el study 1; **habrá que recalcularlos sobre el study final**, no copiarlos:

- El perfil `defensive` bate a `balanced` (exceso 3,34 % vs 1,07 %; IR 0,46 vs 0,19), lo que
  contradice la quinta afirmación vertebradora del TFM.
- El agente `risk` por sí solo (Rank-IC 0,117) **supera al meta-agente** (0,094), que le asigna el
  tope máximo de peso (0,50).
- El contraste de carteras aleatorias no se supera en el escenario general (percentil 61,4); sí en
  el emparejado por riesgo (96,8). El Deflated Sharpe sigue sin superarse (0,844 < 0,95).
- Rank-IC casi idéntico al study anterior (0,1000 vs 0,1004) pero economía más débil (IR 0,189 vs
  0,269; exceso 1,07 % vs 1,62 %), con coeficiente de transferencia 0,178.

### Pendiente

El usuario lanza los studies 2 y 3. Después: rellenar la tabla de progresión y ejecutar la fase 2
(migración del manuscrito) sobre el study final.

## 2026-08-12 · `target_size` incorpora 5 posiciones al extremo concentrado de la rejilla

### Decisión

La rejilla de `target_size` pasa de (8, 12, 16, 25, 50) a **(5, 8, 12, 16, 25, 50)**. El baseline
sigue siendo 12: solo se amplía el barrido, no se cambia la configuración recomendada.

### Motivo

La rejilla se había ampliado por arriba (25 y 50) para medir cuánta señal recupera la amplitud, pero
nunca por abajo: 8 era el extremo concentrado y no permitía ver dónde empieza a dominar el riesgo
idiosincrático. Con 5 nombres cada posición pesa en torno al 20 %, así que el barrido cubre ahora el
rango completo entre «la cartera es un puñado de convicciones» y «la cartera cosecha el Rank-IC
medio», que es justo la curva que la ley fundamental predice.

`target_size` es `predictive=False` (etapa `portfolio`), así que este valor **no puede alterar el
ganador**: se ejecuta después de congelarlo y solo describe su comportamiento económico.

### Cambios

1. [module/studies/catalog.py](../module/studies/catalog.py): valor `5` en la rejilla, con su
   etiqueta («5 posiciones») y su descripción larga para la interfaz.
2. [docs/gestion_cartera.md](gestion_cartera.md): tabla de variables actualizada.

### Verificación

`pytest` 75/75. El catálogo público expone los seis valores con etiqueta y descripción. El hash del
catálogo cambia —lo fija el contenido de las rejillas—, pero `CATALOG_VERSION` se mantiene en 6: no
hay ruptura de comparabilidad predictiva, porque ninguna variable de selección se toca.

### Pendiente

El barrido de cartera de studies anteriores no contiene el punto de 5 posiciones; aparecerá en el
próximo run.

## 2026-08-12 · Robustez: se persisten las distribuciones nulas, no solo sus resúmenes

### Decisión

Los tres contrastes de robustez basados en simulación calculaban sus réplicas y las descartaban,
guardando únicamente estadísticos-resumen. Ahora se persisten las distribuciones completas en
`robustness.json`:

| Contraste | Función | Qué se guarda ahora |
|---|---|---|
| Permutación | `score_permutation` ([module/research/robustness.py](../module/research/robustness.py)) | `null_distribution`: los 9.999 estadísticos permutados |
| Bootstrap por bloques | `block_bootstrap_ci` ([module/evaluation/stats.py](../module/evaluation/stats.py)) | `replicates`: las 2.000 medias remuestreadas |
| Carteras aleatorias | `_simulate` ([module/research/robustness.py](../module/research/robustness.py)) | `null_distribution`: el CAGR de las 1.000 simulaciones, en el escenario general y en el emparejado por riesgo |

### Motivo

El manuscrito declaraba una limitación autoimpuesta (anexo de evidencia complementaria): al
conservar solo resúmenes, tres figuras del TFM representaban estadísticos y no histogramas de la
distribución nula, porque dibujar el histograma habría exigido *simular* una distribución
plausible —es decir, inventar datos—. El propio anexo concluía que «el coste de almacenamiento es
despreciable frente al valor probatorio». Con las réplicas persistidas esa limitación desaparece:
las figuras pueden mostrar dónde cae el valor observado dentro de la nube completa.

### Cambios

1. `module/evaluation/stats.py`: `block_bootstrap_ci` devuelve `replicates`.
   **`paired_difference_ci` las descarta a propósito**: alimenta la selección del Study y se
   evalúa una vez por candidato de cada variable, así que arrastrarlas multiplicaría el tamaño de
   `decisions.json` sin que ninguna figura las lea.
2. `module/research/robustness.py`: `score_permutation` acumula los estadísticos en vez de solo
   contar excedencias; `_simulate` devuelve las muestras; `bootstrap_and_eras` guarda las réplicas
   **una sola vez** (en `interval_95`), porque los intervalos al 90 % y al 95 % remuestrean la
   misma serie con la misma semilla y sus réplicas son idénticas.
3. `latex/assets/c_evidencia_complementaria.tex`: la sección «Limitación de trazabilidad» pasa a
   describir qué se guarda de cada contraste.
4. `latex/assets/09_resultados.tex` y `latex/assets/t09_limitaciones.tex`: se elimina la
   advertencia y la fila de la tabla de limitaciones que declaraban la limitación ya resuelta.

### Verificación

Los estadísticos-resumen **no cambian**: recalculados sobre el study
`study-20260803-201234-b4d7a8d8`, `interval_90` e `interval_95` reproducen bit a bit los valores
ya persistidos (media 0,100409; IC 95 % [0,033450; 0,169474]) y las exclusiones por era son
idénticas. Además el IC se recalcula exactamente desde las réplicas guardadas y el $p$-valor de
permutación se reproduce desde su distribución nula: el cambio es puramente aditivo. Coste en
disco ≈ 260 KB por study. `pytest` 75/75 y `verify_latex_assets.py` correctos.

### Pendiente

El study en curso queda obsoleto con este cambio: **el usuario reinicia el run**. Después, la fase
2 (migrar el manuscrito al study nuevo, con material nuevo: explicabilidad por acción, análisis
real de cartera, calibración de la señal, evolución de coeficientes e histogramas de distribución
nula) según el plan acordado.

## 2026-08-12 · Revisión a priori del baseline (`recommended`) de las 21 variables predictivas

### Decisión

Antes de lanzar ningún Model Study, se revisó variable por variable si el `recommended` actual
en [module/studies/catalog.py](../module/studies/catalog.py) seguía siendo el punto de partida
más defendible **por lógica de dominio únicamente**, sin usar ningún resultado empírico (no
existe todavía ningún study lanzado con este catálogo). Resultado: **los 21 baselines vigentes se
mantienen sin cambios** — cada uno resistió el análisis bajo dos criterios rectores: simplicidad/
regularización como default (la complejidad debe ganarse compitiendo, no asumirse) y prudencia
PIT/causal donde había ambigüedad.

Se aclaró explícitamente con el usuario que esto no se refiere a los 8 "perfiles de inversor"
(`module/evaluation/profiles.py`, `PROFILE_WEIGHTS`) — esos ya son pesos de agente elegidos a
priori y se aplican después de que el study elija un único ganador, no configuran el catálogo.

### Motivo

El usuario pidió fijar el mejor baseline por variable "en base a la lógica, antes de haber hecho
ningún study ni saber lo que mejor funcionaría", para tenerlo listo cuando decida iniciar el
próximo Model Study. El `recommended` de cada variable es el baseline con el que arranca la
optimización secuencial greedy (docs/metodologia.md §3): condiciona sobre qué incumbent se
evalúa cada variable siguiente, así que no es un detalle cosmético.

### Cambios

- `docs/metodologia.md`: nueva sección 4.5 con la razón de cada baseline, variable por variable.
- Ningún cambio en `module/studies/catalog.py` (no se justificó ningún cambio de valor).

### Pendiente

Ninguno: es una revisión, no abre trabajo nuevo. El baseline queda documentado para cuando se
lance el próximo Model Study.

## 2026-08-12 · Cartera: suelo de cobertura incondicional y nueva tolerancia de rebalanceo 0.40

### Decisión

Dos cambios pedidos por el usuario sobre `docs/gestion_cartera.md`, ambos en variables
`predictive=False` (no exigen re-ejecutar el study):

1. **Suelo de cobertura siempre vende.** El paso 1-bis (`below_coverage_percentile`) en
   `decide_orders` ([module/evaluation/portfolio.py](../module/evaluation/portfolio.py)) dejó de
   frenarse por el suelo de diversificación (`coverage_floor`). Antes, con `max_cash_weight > 0`,
   una posición bajo `coverage_percentile_floor` podía **no** venderse si hacerlo rompía el mínimo
   de plazas ocupadas. Ahora se vende siempre, sin excepción, respetando solo
   `minimum_holding_period`. El paso 4 (relleno obligatorio) sigue decidiendo, sin cambios, si la
   plaza liberada se recompra o queda en efectivo.
2. **Nueva opción de catálogo `rebalance_drift_tolerance = 0.40`** ([module/studies/catalog.py](../module/studies/catalog.py)),
   para atacar el 20 % de turnover atribuido a rebalanceo de pesos (§6 de
   `docs/gestion_cartera.md`). El valor "actual" del ganador congelado no cambia (sigue en 0.25);
   0.40 queda disponible para el barrido de escenarios.

### Motivo

El usuario, revisando `docs/gestion_cartera.md`, pidió que la venta por suelo de cobertura fuera
incondicional (vender siempre y decidir el destino del efectivo después, no antes) y que se
explorara reducir la rotación anual actuando sobre pesos/rebalanceo sin tocar variables
predictivas ni relanzar el study.

### Cambios

- `module/evaluation/portfolio.py`: eliminado el freno `if len(holders) <= coverage_floor: break`
  del bucle de venta por suelo de cobertura; docstring del módulo y del bucle actualizados.
- `module/studies/catalog.py`: añadido `0.40` a los valores de `rebalance_drift_tolerance`, con su
  etiqueta y descripción.
- `tests/test_economic_contract.py`: `test_coverage_floor_never_breaks_the_diversification_floor`
  reescrito como `test_coverage_floor_sells_every_position_below_it_unconditionally` para reflejar
  el nuevo comportamiento (las cuatro posiciones se venden, no solo hasta el suelo).
- `docs/gestion_cartera.md`: secciones 2, 3, 4, 6 y 7 actualizadas; la tabla de efecto medido en §4
  se marca como anterior al cambio, pendiente de remedir.

### Pendiente

Remedir el panel de prueba de §4 y el barrido de turnover de §6 con la nueva lógica; decidir si
`rebalance_drift_tolerance = 0.40` pasa a ser el valor "actual" una vez medido.

## 2026-08-05 · Manuscrito: unificación de capítulos, tablas y figuras en `latex/assets/`

### Decisión

Fusionar `latex/caps/`, `latex/figuras/` y `latex/tablas/` en una única carpeta plana
`latex/assets/`, sin subcarpetas: los 16 capítulos `.tex`, las 25 tablas `.tex` y las figuras `.png`
conviven sueltos ahí. Es una reorganización de rutas, no de contenido: ninguna cifra ni artefacto
cambia.

### Cambios

1. `main.tex` pasa a incluir capítulos con `\input{assets/...}`; dentro de cada capítulo, las
   referencias a tablas y figuras pierden el prefijo de subcarpeta (`\input{tXX_...}`,
   `\includegraphics{fXX_...}`).
2. `latex/scripts/export_study_assets.py`: `load_paths()` escribe figuras y tablas directamente en
   `latex/assets/`; el manifiesto filtra por prefijo (`f*.png`, `t*.tex`) para no listar los
   capítulos como si fueran tablas generadas.
3. `latex/scripts/verify_latex_assets.py` reescrito para la estructura plana: exige que
   `\includegraphics`/`\input` de tablas no lleven subcarpeta, y que `\input` de capítulos use el
   prefijo `assets/`.
4. `latex/plan_tfm.md` actualizado con la nueva estructura de carpetas.

## 2026-08-05 · Manuscrito: reordenación de capítulos para lectura lineal, sin cambio de evidencia

### Decisión

Reorganizar la prosa y el orden de los capítulos del manuscrito LaTeX para que una lectura lineal,
de principio a fin, no obligue a saltos ni a relecturas. Ningún artefacto, cifra, tabla ni figura
cambia; es una reorganización puramente editorial sobre el mismo estudio `study-20260803-201234-b4d7a8d8`.

### Motivo

El capítulo 8, «Proceso de desarrollo, defectos encontrados y decisiones», vivía después del
capítulo de Resultados, pero varios capítulos anteriores (4, 5, 6, 7) ya lo citaban o daban por
sabido su contenido —por ejemplo, la «salvaguarda de la curva de alfa» se usaba en el capítulo de
diseño experimental sin definirse, y sólo se explicaba en el capítulo 8, leído después—. Además
había tres duplicaciones de contenido casi textuales (la advertencia de reproducibilidad v5/v6
repetida tres veces; dos secciones del capítulo de Resultados que resumían lo ya dicho en el mismo
capítulo; el capítulo de Limitaciones repitiendo la narrativa causal de tres defectos ya contados en
el capítulo 8) y una duplicación interna en Estado del arte (dos secciones explicando la misma
fórmula IR≈IC·√amplitud·TC).

### Cambios

1. **División del capítulo de desarrollo en dos.** `05_desarrollo_metodo.tex` (historia I: estudio
   único, contratos, puerta de no inferioridad, ponderación por recencia) se coloca tras el capítulo
   de diseño metodológico. `08_desarrollo_cartera.tex` (historia II: rediseño de cartera, colapso
   isotónico, advertencia de reproducibilidad) se coloca tras diseño experimental. Cada mitad
   aparece justo antes de que su contenido se dé por sabido en el capítulo siguiente.
2. **Renumeración en cascada** de todos los ficheros de `latex/caps/` para que el prefijo numérico
   vuelva a documentar el orden real de lectura (01 a 12, más los tres anexos sin prefijo numérico).
3. **Tres duplicaciones resueltas** con el patrón fuente única + remisión breve: la advertencia de
   reproducibilidad queda íntegra sólo en `08_desarrollo_cartera.tex`; las dos secciones redundantes
   de Resultados se eliminan/recortan; «Limitaciones heredadas del desarrollo» se recorta a la
   consecuencia interpretativa, remitiendo al capítulo de desarrollo para la causa.
4. **Fusión de la duplicación interna en Estado del arte**: las dos secciones sobre Rank-IC,
   amplitud y transferencia se funden en una única sección.
5. **Referencias cruzadas reapuntadas**: el label `chap:desarrollo` se divide en
   `chap:desarrollo-metodo` y `chap:desarrollo-cartera`, y las siete citas existentes se reasignan al
   capítulo correcto. Se añaden labels `chap:estado-arte` y `chap:datos`, ausentes hasta ahora, para
   poder referenciarlos por macro en vez de número literal.
6. **Números de capítulo en texto plano corregidos** a macros `\ref`: el resumen de organización del
   capítulo de introducción y una referencia suelta en el capítulo de agentes.

## 2026-08-05 · Manuscrito: bibliografía manual, trazabilidad de robustez y ampliación a ~69 páginas

### Decisión

Ampliar el TFM de ~38 a ~69 páginas estimadas, eliminar `biblatex` del documento y cerrar el último
hueco de trazabilidad del pipeline de activos. Toda la evidencia sigue procediendo de un único
estudio, `study-20260803-201234-b4d7a8d8` (catálogo v5); no se introduce ningún otro.

### Motivo

Tres problemas distintos. El primero era bloqueante: la cadena `biblatex` + `biber` impedía compilar,
de modo que el manuscrito no podía ni revisarse. El segundo, de contenido: los capítulos explicaban
*qué* se decidió sin desarrollar *cómo* ni *por qué*, los anexos eran esbozos de seis líneas y la
bitácora —donde está registrado el proceso real de desarrollo— no aparecía en ningún sitio. El
tercero, de coherencia: el estudio persistía bastante más evidencia de la que el documento explotaba.

### Cambios

1. **Fuera `biblatex`.** Se eliminan el paquete, `\addbibresource` y `\printbibliography`; se
   conserva `csquotes` (lo usa babel). Las 17 llamadas de cita pasan a texto plano autor-año, y las
   cuatro `\textcite` de sujeto gramatical se reescriben como frase. Nuevo `caps/10_bibliografia.tex`
   con las doce referencias a mano. `referencias.bib` se conserva como registro, sin intervenir en la
   compilación. **Overleaf: XeLaTeX, dos pasadas, sin Biber.**
2. **Trazabilidad de robustez.** `draw_robustness()` recibía `robustness` y `attribution` y **no los
   leía**: `tests` y `verdict=[1,1,1,1,1,1,1,0]` eran literales, igual que las ocho filas de
   `t07_robustez.tex`. Eran los dos únicos activos no trazados a artefactos. Ahora
   `build_robustness_rows()` deriva los ocho contrastes y alimenta tabla y figura. **Ninguna cifra ni
   veredicto cambia** (verificado contra la versión anterior); sólo cambia su procedencia, y dos
   filas ganan precisión al dejar de estar redondeadas a mano.
3. **Activos nuevos**: 15 figuras y 16 tablas generadas desde artefactos ya existentes —matriz
   rank-IC por agente y era, pesos anuales del meta, escalera de decisiones, forest plot del
   bootstrap, barrido de cartera, descomposición del turnover por motivo, catálogo completo,
   cobertura anual, catálogo de features. `write_tables` se divide en cuatro funciones y se añade
   `longtable()` para las tablas que no caben en una página.
4. **Capítulo 8 nuevo, «Proceso de desarrollo, defectos encontrados y decisiones»** (2.500
   palabras), escrito desde esta bitácora: la puerta de no inferioridad que penalizaba a los
   superiores, las features declaradas que no se calculaban, el colapso de la isotónica y el
   artefacto media/mediana, el caso MAC, las dos decisiones tomadas contra la recomendación técnica y
   la advertencia de reproducibilidad v5/v6. Limitaciones pasa a capítulo 9 y conclusiones a 10.
5. **Resto de capítulos**: resumen/abstract, `\listoffigures`, `\listoftables`, `amsthm` para la
   proposición de no fuga, y ampliación de los capítulos 1 a 7, 9 y 10 más los tres anexos.

### Correcciones detectadas al escribir

- `f07_barrido_cartera` usaba `PercentFormatter(decimals=0)` sobre un rango de tres puntos
  porcentuales: todas las etiquetas del eje colapsaban en «2 %». Ahora un decimal.
- El anexo de reproducibilidad seguía indicando «ejecutar Biber», ya obsoleto.
- Se afirmaba que el producto cartesiano del catálogo superaba «diez millones» de combinaciones. Son
  1,9 × 10^14 (5,4 × 10^8 restringido a las predictivas).
- Al redactar el capítulo 7 se comprobó que el peor año es 2024, no 2020. Nota: conviven dos series
  próximas pero distintas —`annual_metrics.parquet` da alfa 2024 de −11,63 % y 2020 de −4,37 %,
  mientras `attribution.json.transfer.by_year` da exceso de −12,33 % y −4,01 %—. El capítulo 7 usa la
  primera para la tabla anual y la segunda para el gráfico de rotación/exceso, y lo declara.

### Asimetrías declaradas en el texto

- `feature_catalog.json` declara 68 features y `feature_diagnostics.parquet` contiene 73. La tabla
  hace unión externa y marca el origen en vez de descartar en silencio.
- `selection_status` vale `candidate` en las 73 filas: **no hay traza de qué se podó**, así que la
  tabla se presenta como «catálogo declarado y sus diagnósticos», no como seleccionadas/descartadas.
- Conviven dos ventanas de rank-IC por agente: 117 cohortes (selección: risk 0,1229, meta 0,1004) y
  123 (`robustness.json`: 0,1199 y 0,0949). Cada tabla declara cuál usa.
- **Las distribuciones nulas no se persisten** (permutación, bootstrap, carteras aleatorias): sólo
  estadísticos-resumen. No se dibujan histogramas porque exigiría simular datos; se representan los
  resúmenes reales y la limitación queda declarada en el anexo C.

### Resultado incómodo que el texto reporta

El barrido diagnóstico muestra que **13 de 23 variantes superan al ganador congelado** en exceso
geométrico. `minimum_holding_period → half_horizon` sube el exceso de 1,62 % a 2,58 %, el IR de
0,269 a 0,398 **y** baja el turnover de 3,59 a 2,28. El valor `none` no se midió: viene del
`recommended` del catálogo. No cambiarlo es correcto (regla 4: la cartera no se optimiza), pero es
una limitación real y se declara como tal en el capítulo 9.

### Validación

- `python latex/scripts/verify_latex_assets.py` correcto. Se amplía: ahora detecta comandos de
  bibliografía reaparecidos, `\ref` sin `\label` (recolectando también los de `tablas/`), capítulos
  inexistentes y activos generados sin insertar.
- 75 tests pasan, `node --check` sin avisos. `ruff` reporta un único error (`AGENT_NAMES` sin usar en
  `module/studies/catalog.py`) **preexistente**, verificado con el árbol limpio; no se toca por estar
  fuera del alcance.
- Sin compilador LaTeX local: se valida con un chequeo estructural (balance de entornos, llaves y
  `$`) y con el verificador. **La compilación real en Overleaf está pendiente de confirmación.**

## 2026-08-04 · Cartera: una sola variable de efectivo, corrección del reparto y suelo de cobertura

### 1. `cash_policy` se elimina; `max_cash_weight` gobierna sola el efectivo

`cash_policy` (`fully_invested` / `opportunity_cash`) y `max_cash_weight` eran dos grados de
libertad para una sola decisión. El propio catálogo ya lo reconocía: `settings_from_values` forzaba
el tope a 0 bajo `fully_invested` «para que no exista un grado de libertad inoperante». De las cinco
ramas que trataban la política como caso especial en `portfolio.py`, tres eran equivalentes a
`max_cash_weight = 0` y dos no estaban justificadas:

| Punto | `fully_invested` | `opportunity_cash` con tope 0 |
|---|---|---|
| Suelo de posiciones | `target_size` | `ceil(1,0 × target_size)` — igual |
| Relleno obligatorio | `target_size` | suelo = `target_size` — igual |
| Venta a efectivo | desactivada | activa, pero el suelo la corta — igual |
| Rotación: umbral al outsider | no lo exige | **sí lo exige — difería** |
| Fracción invertida | `holders / target_size` | `1,0` — **difería (ver 2)** |

Ahora `max_cash_weight = 0` **significa** «siempre invertido»: el suelo de diversificación se deriva
del tope y no hace falta una variable que diga lo mismo por otra vía. Se conserva el motivo
`fully_invested_fill` frente a `cash_floor_fill` en las órdenes para que la auditoría siga
distinguiendo por qué se compró.

### 2. Corrección: el reparto dejaba efectivo no declarado

Con 9 de 12 plazas cubiertas, `fully_invested` invertía el **75 %** y `opportunity_cash` con tope 0
invertía el **100 %**. La política llamada «siempre invertida» retenía más efectivo que la política
de efectivo con tope cero, y ese 25 % no lo declaraba ninguna variable, así que no respetaba tope
alguno.

La corrección distingue dos situaciones que la rama anterior confundía:

- **Universo escaso** (no hay con qué llenar la cartera): el capital se reparte entre las plazas
  realmente ocupadas y el tope decide cuánto se retiene. Aquí estaba el error.
- **Compra bloqueada** (`price_only_sell_only` en un snapshot sin fundamentales nuevos): el hueco
  sigue siendo efectivo transitorio y **no** se reparte. Repartirlo concentraría la cartera —una
  superviviente pasaría del 50 % al 100 %— justo cuando se ha decidido no actuar sin información
  nueva, que es lo contrario del propósito de la variable. Se detectó al reescribir
  `test_price_only_sell_only_allows_sale_but_blocks_replacement`, que falló con la primera versión
  de la corrección.

### 3. Nueva variable `coverage_percentile_floor` (0 / 60 / 80), baseline 0

Una posición que cae por debajo del percentil configurado se vende entera con el motivo
`below_coverage_percentile`, **una vez cumplido el mínimo de tenencia**. Es la generalización de
`missing_current_score`: percentil ausente sustituido por percentil demasiado bajo.

Esto **enmienda** la decisión registrada el 2026-07-30 («No se introduce un umbral p4/p5»). La
enmienda es deliberada y acotada: aquella frase defendía que los umbrales que compiten contra el
coste de operar deben ser económicos, y eso sigue intacto —la rotación sigue exigiendo cubrir coste
más margen—. El suelo de cobertura no se compara contra ningún coste: no decide si una operación es
rentable, sino si la acción sigue perteneciendo al universo invertible. Es un mandato, como el
mínimo de tenencia. El test `test_portfolio_thresholds_are_economic_not_percentiles` se estrecha en
vez de borrarse, conservando la doctrina y documentando por qué esta variable no la viola.

Diferencia con `missing_current_score`, que conviene tener presente: aquella **ignora** el mínimo de
tenencia (la posición ha perdido cobertura y ya no es evaluable), ésta **lo respeta** (la posición
sigue siendo scoreable y la exclusión es una preferencia declarada).

**Consecuencia declarada**: con `max_cash_weight = 0` el relleno obligatorio recompra la mejor
disponible en el mismo snapshot sin aplicar umbrales, así que la venta por cobertura fuerza una
rotación que el bucle económico habría rechazado y **aumenta** la rotación. Con tope de efectivo la
plaza puede quedarse vacía hasta el suelo de diversificación. Queda fijado por test.

### 4. Turnover: el barrido diagnóstico ya tenía la respuesta

Descomposición del turnover del ganador `study-20260803-201234-b4d7a8d8` (suma de |Δpeso| por
motivo, total 36,0 sobre 110 snapshots con órdenes): rotación **56 %**
(`net_edge_over_worst` + `displaced_by_net_edge`), `rebalance` 20 %, `initial_fill` 13 %,
`expected_alpha_below_exit` 10 %. La causa raíz es estructural: se re-decide **mensualmente** sobre
una señal a **12 meses** con `minimum_holding_period = "none"`.

El barrido de cartera ya medía las palancas sobre el ganador congelado:

| Cambio | Turnover | IR | Alfa geom. |
|---|---|---|---|
| *(ganador)* | 3,59 | 0,269 | 1,62 % |
| `minimum_holding_period` → `half_horizon` | **2,28** | **0,398** | **2,58 %** |
| `rotation_edge_bps` → 100 | 2,78 | 0,315 | 2,06 % |
| `price_only_sell_only` → true | 2,53 | 0,362 | 2,24 % |
| `minimum_holding_period` → `full_horizon` | 1,67 | −0,050 | −0,67 % |

`half_horizon` baja el turnover un 36 % y **a la vez** sube IR y alfa; `full_horizon` muestra que el
compromiso no es monótono. El valor `"none"` del ganador viene del `recommended` del catálogo, no de
una elección medida. **Se decide no cambiar ese baseline**: queda como evidencia disponible para el
informe, no como cambio de configuración.

### Validación

`python -m pytest -q` (75 pasan, 7 nuevos) y `node --check app/js/app.js`. Se capturaron 15
escenarios de `decide_orders` antes y después del refactor: 12 idénticos, 3 diferencias, todas
intencionadas (dos por la corrección del reparto en universo escaso, una por la etiqueta del motivo
de relleno).

**`CATALOG_VERSION` 5 → 6.** La unificación y la corrección cambian resultados, así que
`study-20260803-201234-b4d7a8d8` deja de ser reproducible con este código y la comparabilidad con
estudios anteriores queda rota. Habrá que re-ejecutar el study para tener un ganador coherente con
el catálogo v6.

## 2026-08-04 · Ruta científica completa: evidencia opcional de todos los runs

### Decisión

Se añade un conmutador junto a «Lanzar Study», «Ruta científica completa», que cuando se activa
hace que todos los runs del Study retengan su evidencia entera —cartera, órdenes, posiciones,
curva de capital, pesos del meta-agente, diagnósticos de Rank-IC y atribución— en
`runs_evidence/<clave_lógica>/`, y no solo el baseline y el ganador.

### Motivo

La regla 5 del proyecto (los descartados guardan solo resúmenes) es una restricción de disco, no
metodológica. Su efecto práctico era que, ante un candidato descartado, solo se podía ver su
Rank-IC agregado: no había forma de responder *por qué* quedó por detrás sin relanzar el Study
entero con otra configuración. Con la evidencia retenida, las vistas de rendimiento, aprendizaje,
cartera y acciones quedan disponibles para cualquier run.

### Qué no cambia

El plan experimental es idéntico con el conmutador activo o inactivo: mismo número de
evaluaciones, mismos candidatos, misma selección por Rank-IC en la ventana de selección y 2025-2026
igualmente fuera de toda decisión. El test `test_full_scientific_route_only_changes_storage_not_selection`
fija ese contrato comparando ambos preflight.

### Coste

El preflight declara por adelantado los runs que retendrán evidencia y el disco estimado (unos
60 MB por run), visible en el panel de presupuesto antes de lanzar. Como la evidencia debe
materializarse, esos runs se recalculan en vez de reutilizar el resumen en caché, así que la
ejecución es más lenta. Por eso es opcional y está desactivado por defecto.

### Validación

`python -m pytest -q` (68 pasan, dos nuevos) y `node --check app/js/app.js`. La resolución de
`source=run:<run_id>` toma el directorio del artefacto del run, nunca de la cadena recibida, y se
confina bajo el Study; hay test de escape de ruta.

## 2026-07-30 · Comparación de perfiles serializable en Parquet

### Corrección

El smoke de cinco tickers completó baseline, ganador y los ocho perfiles, pero PyArrow no pudo
persistir `profile_comparison.parquet`: la tabla recibía diccionarios anidados y uno vacío
(`confirmation`) no tiene un esquema Parquet válido. La comparación agregada guarda ahora solo las
métricas escalares que presenta el dashboard; cada perfil conserva su resumen completo en
`evidence/profiles/<perfil>/summary.json`. Es una corrección de persistencia, sin efecto científico.

## 2026-07-30 · Salida forzada por pérdida de score actual

### Diagnóstico y decisión

La cartera ya tomaba el percentil y el alfa del snapshot actual, no los de compra. En el baseline
de `study-20260730-115839-33de77e2`, ACN estaba en p4,96 con −9,01 % anual y MCD, fuera de cartera,
en p100 con +10,00 %: una ventaja de 1.901 pb, suficiente para una rotación. Sin embargo, MAC seguía
en cartera sin fila scoreable; dejó el S&P 500 el 2019-12-10. El bucle la elegía como peor posición,
no podía calcular su ventaja y detenía toda rotación, incluida la de ACN.

Toda posición sin `meta_rank` actual se vende ahora con el motivo `missing_current_score`, incluso si
no ha cumplido `minimum_holding_period`. No puede recomprarse en ese snapshot. La sustitución posterior
mantiene las reglas existentes: relleno bajo `fully_invested`, umbrales y suelo bajo `opportunity_cash`,
y efectivo transitorio bajo `price_only_sell_only`. No se introduce un umbral p4/p5: los percentiles
actuales se traducen a alfa y la rotación sigue exigiendo cubrir costes y margen.

### Verificación

Pruebas de contrato cubren la salida sin score con mínimo de tenencia, la rotación de una posición p4
antes bloqueada por otra sin score y la prioridad de `price_only_sell_only`.

## 2026-07-30 · Hora de Madrid en la consola de ejecución

### Corrección

Los eventos de cada Study continúan persistiendo su timestamp en UTC, para conservar una referencia
inequívoca y comparable. La línea visible en terminal se formatea ahora en `Europe/Madrid`, igual que
la Consola del dashboard. Al actualizar manualmente esa Consola se aplicaba por error el timestamp
UTC crudo; también queda corregido.

### Verificación

Una prueba de flujo fija una hora de Madrid y comprueba que `append_event` la emite en terminal,
manteniendo además la persistencia incremental de `events.jsonl`.

## 2026-07-30 · Recencia en el meta-agente y curva percentil→alfa con cascada de ventanas

### Diagnóstico que lo motiva

Partió de una observación sobre la cartera: posiciones con 81-90 meses de antigüedad y percentil
actual muy bajo (p4, p17, `null`) que nunca se vendían. La investigación encontró dos problemas
encadenados, ambos verificados sobre `study-20260730-083636-7a6bc807/evidence_baseline/`:

1. **El meta-agente no olvidaba el régimen viejo.** El rank-IC del meta acumula **7 trimestres
   consecutivos en negativo** (2023-12-30 → 2025-06-29, media −0,086), la racha más larga de la
   serie 2015-2025 —más que COVID, que fueron 4 trimestres con media −0,133—. Los cinco agentes se
   invirtieron a la vez (growth −0,065, momentum −0,090, quality −0,077 en el último año), patrón
   compatible con una rotación factorial de mercado ("caro" growth/quality → "barato" value) y no
   con un error de signo: se contrastó contra `rank_ic_diagnostics.parquet`, que calcula el propio
   pipeline, y coincide. El Ridge del meta usaba una ventana **rectangular**: una cohorte de hace
   cuatro años pesaba igual que la última.
2. **La calibración isotónica colapsaba.** En el snapshot 2026-06-29, `expected_excess_return` era
   **idéntico para los 504 tickers** (−307,68 pb). No era un fallo de implementación:
   `IsotonicRegression(increasing=True)` fuerza monotonía creciente, y cuando la relación real es
   decreciente la única curva creciente que minimiza el error es una constante. Con alfa plano
   `decide_orders` no puede discriminar y las posiciones quedan congeladas: `_advantage` devuelve
   `None` y aborta el bucle de rotación entero.

Una nota metodológica sobre el propio diagnóstico: un primer análisis por deciles usando la **media**
del retorno excedente sugirió que el ranking estaba invertido en todo el histórico. Era un artefacto
del estimador —la distribución tiene cola derecha extrema (media +3552 pb, mediana −146 pb)— y unos
pocos multibaggers en el decil bajo lo distorsionaban. Con mediana o media winsorizada la curva
histórica sí es creciente, coherente con el rank-IC positivo (+0,043). La inversión es real, pero
**solo en el régimen reciente**, no en todo el histórico.

### Decisión

1. **Nueva variable `meta_recency_weighting`** (`off`, `linear`, `exponential`, recomendado `off`),
   fase `meta` del catálogo. Replica el mecanismo ya validado de `recency_weighting` para los
   agentes base (`agents.py`), aplicándolo ahora al `sample_weight` del Ridge del meta y reutilizando
   `RECENCY_HALFLIFE_YEARS`. Depende de `meta_method` (no aplica con `equal`). Entra como eje del
   Model Study, no como cambio directo al ganador: tendrá que ganarse el sitio por Rank-IC OOS.

   Validación previa offline (réplica del stacker sobre los datos reales del estudio): en la racha
   adversa el rank-IC medio pasa de −0,0489 a **−0,0398** (`linear`) / −0,0410 (`exponential`), sin
   degradar el histórico completo (0,0532 → 0,0549 / 0,0551). Mejora modesta y consistente: no
   revierte la rotación de mercado —nada lo haría— pero acorta cuánto tarda el meta en dejar de
   confiar en un agente que ya no funciona. Se observó que sin clamp el stacker puede concentrarse
   en un solo agente (100 % `value` en el último snapshot), por lo que la variable está pensada para
   evaluarse junto a `stacked_rolling_bounded`, que limita cada agente al 10–50 %.

2. **La calibración isotónica se sustituye por una curva percentil → retorno real anualizado con
   cascada de ventanas**: `horizonte objetivo → era (16 trimestres) → todo el histórico →
   salvaguarda`. Se ajusta una recta por mínimos cuadrados, ponderando las cohortes recientes más
   que las antiguas, y se acepta la primera ventana con **pendiente creciente**. Todo el alfa pasa a
   expresarse **anualizado**, lo que unifica las unidades con los umbrales del catálogo (que ya eran
   anuales). En consecuencia, `_annual_to_horizon_bps` se sustituye por
   `_horizon_cost_to_annual_bps`: lo que se convierte ahora es el **coste** (que se paga una vez por
   operación) y no el umbral.

   Sobre la granularidad se descartaron dos extremos. Estimar la recta con 100 percentiles deja ~5
   acciones por punto (~20 observaciones en la ventana corta): la recta se ajustaría sobre ruido.
   Estimarla con 10 deciles es robusto pero grueso. Se adoptaron **20 ventiles** (~25 acciones por
   punto) para *estimar*, y —esto es lo que de verdad importa para la cartera— la recta se *evalúa*
   en el **rank continuo** de cada acción, no en su ventil: un p99 recibe estrictamente más alfa que
   un p88, y no hay saltos artificiales en las fronteras. En el estudio actual eso produce 504
   valores de alfa distintos donde la isotónica producía 1.

### Dos decisiones tomadas contra la recomendación técnica

Se dejan registradas con ambos lados del argumento, por honestidad metodológica:

- **Sustituir la isotónica sin dejarla como opción de catálogo.** Se recomendó introducir la cascada
  como una variable más (`alpha_calibration`: `isotonic` | `windowed_cascade`) para que el Model
  Study eligiera con evidencia, a coste casi nulo. *A favor de lo decidido:* el colapso de la
  isotónica es demostrable y deja la cartera inoperante, y mantener dos rutas de calibración
  complica el código sin que el autor prevea volver a la anterior. *En contra:* cambia el ganador
  sin pasar por la selección secuencial por Rank-IC, que es la regla 4 del proyecto, y renuncia a
  poder comparar ambas formulaciones con evidencia en la memoria del TFM. El autor asumió el riesgo
  y pidió eliminar el legacy.
- **Salvaguarda lineal fija de −10 %/+10 % hardcodeada.** *A favor:* garantiza que la cartera
  siempre tiene un alfa utilizable y nunca queda sin señal operativa. *En contra:* es un supuesto a
  priori, no aprendido de datos, del mismo tipo que el `REGIME_TILT` que este proyecto ya eliminó
  (ver 2026-07-25); y se activa **precisamente** cuando las tres ventanas coinciden en que el
  ranking no discrimina a favor, es decir, impone convicción justo cuando la evidencia disponible
  dice lo contrario. Se recomendó una curva plana (alfa neutro, cartera equiponderada) como
  alternativa que también evita quedarse sin señal sin imponer dirección. El autor prefirió la recta
  fija. Mitigación adoptada: cada fila registra en `alpha_curve_window` qué ventana se usó, de modo
  que el informe puede cuantificar en qué snapshots se operó sobre supuesto en vez de sobre
  evidencia. En el estudio actual, la salvaguarda se activa en 4384 de 20545 filas.

### Verificación

`pytest` (59 pruebas), `ruff` y `node --check` en verde. Dos pruebas se reescribieron porque
verificaban la semántica anterior: `test_annual_threshold_converts_geometrically_not_linearly` pasa a
ser `test_operating_cost_annualizes_geometrically_not_linearly` más una prueba explícita de que el
umbral anual se compara contra alfa anual sin reescalados.

`calibrated_alpha_path` **no tenía ninguna cobertura**, y durante el desarrollo un renombrado dejó
una llamada rota en esa ruta que la suite no detectó. Se añade `tests/test_alpha_curve_contract.py`
(6 pruebas) cubriendo lo que puede fallar en silencio: que cada acción reciba un alfa único y
monótono en su rank (no aplanado por ventil), que una relación decreciente caiga hasta la
salvaguarda, que una creciente se quede en la ventana más reactiva, que sin cohortes cerradas el
valor sea `NaN` y no `0.0`, que las tres ventanas se ajusten sobre evidencia distinta, y que la
anualización componga en vez de prorratear.

Se comprobó además que `meta_recency_weighting` en `off` reproduce exactamente los pesos previos.
Queda pendiente de un estudio real autorizado.

## 2026-07-28 · Umbrales anuales con conversión geométrica, mínimo de tenencia y venta sin reemplazo

### Decisión

Tres cambios adicionales a la gestión de cartera, catálogo v4 → **v5**, todos con el mismo objetivo
declarado: **estabilidad del modelo ya congelado, no más alfa**. Barrer variables de cartera nunca
elige el modelo predictivo (eso ya está fijado por Rank-IC); busca cómo aprovechar ese modelo de
forma sostenible en el tiempo, con menos rotación innecesaria y sin operar a ciegas cuando falta
información. `portfolio_comparison.parquet` sigue siendo diagnóstico puro: ninguna comparación entre
`cash_policy`, `minimum_holding_period` o `price_only_sell_only` elige una configuración por dar más
alfa; una opción con más rentabilidad pero más varianza entre semillas o más rotación no se prefiere
solo por eso.

1. **`exit_expected_alpha_bps` y `rotation_edge_bps` pasan a definirse en puntos básicos anuales**,
   convertidos geométricamente (compuesto, no lineal) al horizonte real del modelo antes de
   comparar: `umbral_horizonte = (1 + umbral_anual)^(horizonte_meses / 12) − 1`. Antes, el mismo
   valor de catálogo significaba magnitudes económicas distintas según `target_horizon_months` (250
   pb en un horizonte de 3 meses ≈ 10 %/año; en uno de 12 meses ≈ 2,5 %/año), rompiendo la
   comparabilidad entre configuraciones. El prorrateo lineal (dividir sin más) se descartó
   deliberadamente por ser el mismo atajo aritmético que ya se corrigió en CAGR e IR.
2. **Nueva variable `minimum_holding_period`** (`none`, `quarter_horizon`, `half_horizon`,
   `full_horizon`, recomendado `none`): bloquea toda venta —caída de alfa o rotación— mientras una
   posición no cumpla ese mínimo de meses, expresado como fracción del horizonte. `quarter_horizon`
   está pensado para el caso de un horizonte de 12 meses revisado trimestre a trimestre. Reutiliza
   `state.entry_dates`, ya existente; `months_held` se extrae de `backtest.py` a `portfolio.py` como
   función pública para eliminar la duplicación.
3. **Nueva variable `price_only_sell_only`** (booleana, recomendado `False`): en un snapshot que
   solo trae precio nuevo (sin fundamentales frescos), permite vender una posición cuyo alfa
   esperado ya no cumple, pero prohíbe comprar cualquier reemplazo —compra nueva, relleno
   obligatorio o rotación—, porque no hay información nueva que justifique elegir una acción
   distinta a la ya elegida con datos reales. Bajo `fully_invested`, esto abre una vía de efectivo
   transitorio (sin el tope ni el suelo de `opportunity_cash`, por ser un estado pasajero hasta el
   siguiente snapshot con datos reales) que antes no existía.

### Efecto en el backtest

El guard de coherencia de `backtest.py` (que antes trataba "órdenes sin cartera objetivo" siempre
como un bug) se simplificó: `decide_orders` ya conserva las posiciones previas cuando no hay scores
negociables, y un objetivo vacío en cualquier otro snapshot es una liquidación deliberada a efectivo
(el mecanismo del punto 3), no un error. Solo el primer snapshot sin nada que invertir sigue siendo
un problema real de datos.

### Validación

52 tests (7 nuevos: conversión geométrica, los cuatro valores de `minimum_holding_period`, y dos de
`price_only_sell_only`), ruff sin errores nuevos sobre el baseline preexistente, sintaxis JS. El
dashboard integra ambas variables nuevas sin cambios de código (el catálogo se renderiza de forma
dinámica); solo se actualizaron dos etiquetas fijas de métricas ("pb" → "pb/año").

## 2026-07-28 · Rediseño de la gestión de cartera: toda venta necesita un destino mejor

### Decisión

Reordenar la lógica de compras, ventas y rotaciones bajo un principio único —**una venta solo se
emite si el destino del dinero es mejor que la posición después de costes**— y subir el catálogo a
la versión 4 con `max_cash_weight` en `(0 %, 10 %, 25 %)`, por defecto 25 %. Rompe la
comparabilidad con los Studies de catálogo 3 (que no llegaron a ejecutarse).

### Defectos que motivan el cambio

Los tres tenían la misma raíz: las ventas se decidían sin mirar a dónde iba el dinero.

1. **`opportunity_cash` con todas las candidatas bajo el umbral** (el escenario para el que existe
   la política): `decide_orders` vendía la cartera entera y devolvía un objetivo vacío; el backtest
   interpretaba «vacío» como «mantener posiciones» y las resucitaba, pero ya había cobrado los
   costes de ventas que nunca ocurrieron. La cartera pagaba por replegarse sin replegarse.
2. **`fully_invested` con la curva calibrada bajo el umbral**: como el alfa esperado es monótono con
   el ranking, se vendían las 12 posiciones por umbral y el relleno obligatorio recompraba
   exactamente las mismas en el mismo snapshot: una ida y vuelta completa de la cartera para quedar
   igual. Contribuía al 877 % de rotación anual del diagnóstico.
3. **Concentración sin tope**: con una sola candidata admisible y tope de efectivo del 20 %, el
   80 % de la cartera acababa en una única acción, porque el suelo de inversión se repartía entre
   las admisibles que hubiera.

### Reglas resultantes

- **Rotación**: única vía de venta bajo `fully_invested`; exige ventaja superior al coste de ida y
  vuelta más `rotation_edge_bps` (sin cambios).
- **Venta a efectivo**: solo bajo `opportunity_cash`, con alfa bajo el umbral **y** respetando el
  suelo de diversificación `ceil((1 − max_cash_weight) · target_size)`, que garantiza a la vez el
  tope de efectivo y un mínimo de posiciones (12 plazas y tope 25 % → suelo de 9 → ninguna acción
  supera ~15 % del total).
- **Compra con histéresis**: entrar exige el umbral de salida más el coste de ida y vuelta de la
  propia operación; mantener exige solo el umbral. Elimina el churn de frontera.
- **Prudencia sin fundamentales**: el multiplicador ensancha ahora la banda entera (baja la salida,
  sube la entrada y la rotación).
- **Invariante nuevo**: con puntuaciones en la fecha, la cartera objetivo nunca queda vacía; el
  backtest lo verifica y falla ruidosamente si se viola.

Se documenta además una propiedad, no un defecto: con calibración en 20 ventiles y cartera
concentrada, el efectivo es casi binario y responde a la salud reciente de la señal; solo se vuelve
gradual con `target_size` 25 o 50.

### Correcciones menores del mismo día

- El nulo de carteras aleatorias aplicaba la guarda **mensual** de artefactos a retornos
  **anuales**, excluyendo del azar a ganadores legítimos de más del 100 % anual que el modelo sí
  puede cobrar: sesgaba el nulo a favor del modelo. Ahora usa la cota compuesta `(1+g)^12 − 1`.
- El Deflated Sharpe declara en su docstring la aproximación que hace (Sharpe de cartera contra
  dispersión de series de Rank-IC de candidatos) y que se lee como orden de magnitud del haircut.

### Validación

Suite completa (45 tests, incluidos cinco nuevos de este rediseño), ruff sin errores nuevos,
sintaxis JS y smoke dev end-to-end.

## 2026-07-28 · Pre-registro del protocolo de confirmación 2025–2026

### Decisión

Este bloque se escribe **antes** de ejecutar el Study con la regla de selección corregida, y fija
qué se medirá en la era reservada y cómo se leerá el resultado. La corrección de la puerta pareada
cambia el ganador con alta probabilidad, de modo que la confirmación solo es creíble si no podemos
elegir configuración sabiendo cómo se comporta en 2025–2026.

### Protocolo cerrado

1. **Qué se mide.** Rank-IC transversal por cohorte del `meta_rank` contra `forward_excess_return`
   en 2025–2026, y el alfa geométrico de la cartera en esa misma ventana.
2. **Cuándo.** Una sola vez, después de congelar `winner.json`. La evaluación vive fuera de todo
   bucle de decisión, en `module/research/attribution.py`, invocada por el runner tras escribir el
   ganador.
3. **Estadísticos.** Rank-IC medio, fracción de cohortes positivas, IC-IR, t de Newey-West con 12
   retardos y número efectivo de observaciones independientes.
4. **Cómo se lee.** Con horizonte de 12 meses y cadencia mensual, las cohortes cerradas disponibles
   son aproximadamente seis y comparten casi toda la etiqueta. El resultado se declara **evidencia
   direccional del signo**, no un contraste con potencia. Un Rank-IC medio positivo apoya que la
   ordenación aprendida se mantiene fuera de la ventana de selección; uno negativo o nulo se publica
   igualmente y obliga a matizar la conclusión principal.
5. **Compromiso.** El resultado se publica sea cual sea, sin repetir la evaluación con otra
   configuración ni ampliar la ventana a posteriori.

### Por qué

El entrenamiento es walk-forward con purga, así que no hay lookahead y ninguna cohorte está
contaminada. El sesgo que esto ataca es distinto: las 17 decisiones secuenciales se tomaron
comparando Rank-IC sobre las 117 cohortes de 2015–2024, y la cifra de portada procede de esa misma
serie. Es un estimador insesgado del Rank-IC de esa configuración en ese periodo, pero optimista
como estimador del Rank-IC futuro. La era reservada y el Deflated Sharpe atacan ese sesgo por dos
vías complementarias: la primera mide fuera de la muestra de selección, el segundo descuenta el
efecto de haber buscado.

## 2026-07-28 · Corrección de validez, alfa neto y limpieza

### Decisión

Corregir los defectos que invalidaban parte de la evidencia, sustituir los umbrales de cartera por
umbrales económicos y eliminar el código sin consumidor.

### Correcciones de validez y su efecto

| Defecto | Efecto medido |
|---|---|
| La puerta de no inferioridad penalizaba a los candidatos **superiores**: cuanto mayor la diferencia, más ancho el intervalo | `feature_preset` pasa de `core` (Rank-IC 0,0730) a `all` (0,0958), con ventaja pareada +0,0216. Ningún retador era elegible pese a dominar |
| `market_regime_feature` se decidió sobre una ventaja de 0,00112, por debajo del ruido | Pasa de `True` a `False` por simplicidad |
| `meta_method` se decidió sobre una ventaja de 0,00033 | Se mantiene, pero ahora registrado como empate técnico, no como victoria |
| Emparejamiento por cadena de fecha con rejillas desplazadas por `execution_lag_days` | Se empareja por periodo mensual; sin bloque completo común el resultado se marca no aplicable en vez de devolver `ci_low = 0,0` |
| `evaluation_key` no incluía el hash del dataset | La misma clave aparecía con CAGR 0,1468 y 0,1692. Ahora la clave separa datasets |
| Seis factores de precio se inyectaban en el agente momentum fuera de todo condicional | La ablación `fundamental` ya no recibe información de precio y mide lo que declara |
| Dos definiciones incompatibles de *information ratio* bajo el mismo nombre | Una sola, anualizada |
| CAGR, drawdown y alfa mezclaban la era reservada con la de selección | Métricas segmentadas: selección, confirmación y curva completa |
| `geometric_excess_return` era una resta de CAGR | Cociente de acumulados |
| Una posición sin precio se marcaba plana | Convención de exclusión tipo CRSP (−30 %) y liquidación contra efectivo |
| `subsample=0.8` sin `subsample_freq`: el bagging nunca se activaba | `subsample_freq=1` |
| Nulo de carteras aleatorias con percentil 95 de CAGR del 107 % | Exige cobertura anual completa, aplica la guarda de datos y paga los mismos costes |

### Catálogo de presets: solo `core` y `all`

Se retiran `fundamental` y `technical`. El motivo no es de resultado sino de diseño: ninguno de los
dos alimenta a los cinco agentes —`fundamental` deja sin features a momentum y risk, y `technical` a
quality, value y growth—, de modo que un Study que los eligiera no estaría respondiendo «qué
información necesita cada agente» sino «qué pasa si amputo parte de la arquitectura». Ambos presets
supervivientes mantienen los cinco agentes activos: `core` con su bloque esencial y `all` con toda la
profundidad disponible de cada especialidad.

Este cambio interactúa con la corrección de la puerta pareada: con el catálogo reducido, el ganador
de `feature_preset` pasa de `core` a `all`.

### Cartera: de percentiles a puntos básicos

`min_hold_percentile` y `rotation_edge_percentiles` desaparecen. Los sustituyen
`exit_expected_alpha_bps` y `rotation_edge_bps`, y una rotación solo se autoriza si la ventaja de
alfa esperado supera `2·(comisión + slippage) + margen`. `sizing_mode` pasa de `score_linear`
—anclado a un percentil arbitrario— a `alpha_proportional`. `CATALOG_VERSION` sube a 3 y se asume la
ruptura de comparabilidad con los Studies anteriores.

Se añaden `cash_policy` y `max_cash_weight` en la etapa **diagnóstica**: el efectivo no altera el
Rank-IC y por tanto no puede elegir modelo. Se ejecutan ambas políticas al final y se comparan en
`portfolio_comparison.parquet`. El efectivo se remunera al 0 %.

`target_size` se amplía a (8, 12, 16, 25, 50) para medir cuánta señal recupera la amplitud: por la
ley fundamental, un IC de 0,074 sobre ~250 valores implica un IR teórico en torno a 1,1 frente al
~0,18 realizado, y una cartera de 12 nombres con 877 % de rotación destruía cerca del 85 % de la
señal.

### Estabilidad

Cada agente entrena cinco réplicas que solo difieren en la semilla y promedia los scores. Motivo: el
Rank-IC variaba ±0,001 entre semillas pero el exceso sobre SPY cambiaba de signo (semilla 7: −0,51
pp; semilla 42: +3,11 pp). `robustness.json` publica ahora el rango de alfa entre semillas.

### Evidencia nueva

`module/research/attribution.py`: regresión sobre réplicas de factores con Newey-West, Rank-IC
neutralizado, Deflated Sharpe, baselines deterministas, cobertura del universo por año y coeficiente
de transferencia. Da consumidor a `signal_calibration.parquet`, `signal_health.parquet`,
`top_minus_bottom` y `module/data/baselines.py`, que se calculaban y no leía nadie.

### Eliminado

`_temporal_permutation_importance` (rama inalcanzable), `meta_equal_shrinkage` (siempre 0,0),
`meta_ic_lookback_quarters` (duplicaba `meta_history_quarters`), `ensure_directories`,
`agents_code_version`, `backtest_code_version`, `PortfolioState.cash`, `price_guard`,
`stock_sleeve_return`, `accounting_error`, las columnas `commission`/`slippage` duplicadas, el
fallback a `top_decile_spread`, el endpoint `GET /api/studies/{id}/runs` y la duplicación de
`AGENT_NAMES`. `module/data/ingest/` se conserva con entrada explícita `python main.py ingest`.


## 2026-07-25 · Limpieza integral: código muerto, legacy y features huérfanas

### Decisión

Auditoría función a función de todo el repositorio para eliminar código sin consumidor,
duplicación de la misma verdad en varios sitios y modos legacy con sesgo hardcodeado, en línea
con las reglas de `AGENTS.md`.

### Hallazgo científico principal

`module/modeling/catalog.py::FEATURE_CATALOG` declaraba los bloques `momentum_core` y
`momentum_trend` (factores `mom_acceleration`, `mom_reversal_1m`, `ma_price_vs_sma6`,
`ma_price_vs_sma12`, `ma_distance_to_high12`), presentes en todos los `feature_preset` reales y
barridos en `recommended_definition()`. Pero esas columnas solo se calculaban si
`settings.price_momentum_multi`/`moving_averages` eran `True`, y esas dos variables nunca
existieron en el catálogo cerrado de `module/studies/catalog.py`: no eran alcanzables desde
ningún Study real. El smoke de 5 tickers documentado en `docs/informe_resultados.md`
(`study-20260725-132255-c49da9ff`) se ejecutó **sin** estas columnas de momentum multi-horizonte.

### Cambios

- `price_momentum_multi`/`moving_averages` dejan de ser flags: sus artefactos
  (`add_price_momentum_multi`, `add_moving_averages`) se calculan siempre, igual que ya ocurría
  con `add_market_risk_liquidity`. `features_code_version` y `agents_fit_code_version` suben
  para invalidar cachés y materializaciones previas sin estas columnas.
- Eliminados `regime_extended`/`quality_growth_derived` y sus artefactos
  (`add_regime_extended`, `add_quality_growth_derived`): a diferencia de momentum, sus factores
  nunca estuvieron declarados en `FEATURE_CATALOG` — código huérfano de punta a punta.
- Eliminados los modos `meta_type="regime"`/`"rank_ic"` (sesgo `REGIME_TILT` hardcodeado a
  mano, nunca aprendido de datos) y `meta_history_mode="expanding"`/`"exponential"` (el runner
  siempre forzaba `"rolling"`). Solo quedan `"equal"` y `"stacked_oos"`, los dos únicos modos
  que el catálogo cerrado puede producir.
- Simplificada la infraestructura de ensemble multi-familia en `module/modeling/agents.py`
  (nunca se ejecutaba con más de una familia; el catálogo obliga a exactamente una).
- Eliminados módulos y funciones sin ningún consumidor: `module/studies/budget.py`,
  `module/evaluation/robustness.py`, `settings_payload`, `discard_summary_cache`,
  `append_ledger`, `cache_usage`, `prune_prepared`/`pinned_dataset_hashes`/`prepared_usage`,
  cuatro funciones huérfanas de `signal_diagnostics.py` (`summarize_tail`, `era_summary`,
  `moving_block_bootstrap_delta`, `holm_adjust`), endpoint `GET /api/studies/{id}/runs`.
- Consolidada duplicación de la misma verdad: `PROFILE_NAMES` (antes definido dos veces),
  `SELECTION_ERAS` (antes triplicado literalmente), `_link_or_copy` (antes duplicada byte a
  byte en `cache.py` y `datasets.py`, ahora en `module/common/utils.py`).
- Añadido botón Cancelar en el dashboard junto a Pausar/Reanudar (el endpoint ya existía sin
  cliente de UI).
- `CLAUDE.md` actualizado: ya no describe la arquitectura Exploratory→Confirmatory eliminada
  el mismo día.

### Validación

- Suite completa: 15 tests superados, ruff y `node --check` sin avisos.
- Smoke dirigido: dataset dev reconstruido con `ensure_prepared`; las seis columnas de
  momentum multi-horizonte/medias móviles aparecen con 100 % de cobertura (antes ausentes).
- `build_agent_scores` verificado de extremo a extremo con `meta_type="equal"` y
  `meta_type="stacked_oos"` sobre el dataset dev tras la simplificación de `meta.py`.

## 2026-07-25 · Reconstrucción a Model Study único

### Decisión

Se elimina el protocolo Exploratory → hipótesis → Confirmatory. La unidad científica pasa a ser
un único Model Study automático. Solo las fases predictivas seleccionan mediante Rank-IC.

### Motivo

El flujo anterior mezclaba entidades, multiplicaba rutas y podía dejar Studies fantasma. El Study
iniciado el 24 de julio terminó un fit caro pero falló antes del ledger al consultar
`signal_health_lookback_quarters`, campo ya eliminado. El error solo vivía en memoria y el proceso
desapareció sin reconciliación.

### Cambios

- Catálogo v2 con variables predictivas y cartera informativa.
- Tres meta-agentes: equal, rolling free y rolling 10–50 %.
- Persistencia de Study, runs y eventos antes del cálculo.
- Worker hijo por Study, heartbeat, cancelación, interrupción y reanudación.
- API y dashboard reducidos a Inicio y Resultados.
- Cartera 100 % acciones; SPY solo benchmark.
- Robustez y ocho perfiles posteriores al ganador.
- Eliminación de Exploratory, hipótesis, Confirmatory y modelos promovidos.
- Corrección de la referencia al campo eliminado.

### Validación final

- Suite crítica: 15 tests superados.
- Ruff, compilación Python y sintaxis JavaScript superados.
- Auditoría UTF-8 sin secuencias de mojibake en fuentes.
- Smoke real corregido: `study-20260725-132255-c49da9ff`, estado `succeeded`.
- 27 runs físicos finalizados, 53 eventos persistentes y 872.775 bytes de evidencia.
- Reanudación del Study finalizado: cero runs añadidos y mismos identificadores.
- Worker finalizado y `worker_pid = null`.

### Incidencias descubiertas por los smokes

1. La primera ejecución falló al serializar valores de cartera de tipo texto y número en una
   columna Parquet. Se normalizaron ambos valores como JSON.
2. El primer smoke técnicamente exitoso produjo scores constantes: 50 observaciones mínimas por
   hoja impedían dividir árboles con 65 filas. No se aceptó como validación. El modo dev limita
   ahora el mínimo a 5; el smoke repetido produjo 23 cohortes y Rank-IC no degenerado.
3. La lista de Studies fallaba cuando un run aún tenía `result = null`. La consulta trata ahora
   correctamente los runs creados antes de calcular.
4. La concentración meta mezclaba la columna de cohortes realizadas con los pesos. Se sustituyó por
   HHI de pesos por fecha y turnover medio de media norma L1.
5. Se añadió vigilancia del PID padre: si termina abruptamente el dashboard, el worker se marca
   `interrupted` y se detiene por sí mismo.
6. El dashboard pasó de tablas aisladas a visualización analítica: porcentajes en escala humana,
   equity con ejes y leyenda, evolución multicolor de Rank-IC y pesos, perfiles por año y barras de
   robustez para semillas, placebos y agentes.
7. Los ejes de las curvas se calculan ahora por métrica. En particular, los pesos se limitan al
   intervalo válido 0–100 % y se ajustan al rango observado; cada punto ofrece fecha, serie y valor
   exacto al situar el cursor. La configuración de cada run se presenta en tarjetas temáticas en
   lugar de una tabla plana.
8. La navegación contextual vuelve bajo Resultados. Los gráficos de líneas usan cursor vertical y
   una leyenda flotante de todas las series en la fecha más cercana, sin puntos visibles. Performance
   usa años como marcas del eje X, divisores verticales secundarios y ticks enteros en equity.
   Portfolio y Stocks comparten snapshot; Portfolio integra las órdenes del día y Stocks permite
   consultar cartera, agentes, parámetros PIT, puntuaciones de factores y evolución temporal.

Las cifras del smoke sirven para validar el flujo, no como evidencia económica o científica del TFM.
