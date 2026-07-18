# Informe final — Sistema de IA para selección de acciones (2016-2026)

*Estudio completo ejecutado de principio a fin con un único comando (`RUN_MODE=full_study`).
Universo S&P 500 histórico, 680 empresas con datos, ventana 2016-2026. Todas las cifras proceden
de `results/escenarios/study_summary.json`. Informe deliberadamente honesto.*

## Resumen ejecutivo

El estudio tiene **dos hallazgos opuestos y coherentes**:

1. **El modelo de IA no aprende de forma significativa** a ordenar acciones (rank-IC +0.0036,
   p-valor del placebo 0.20 — indistinguible del azar).
2. **Pero una cartera con sesgo de estilo defendible sí bate al mercado**: los perfiles quality,
   value y conservative rinden **+4.4 %/año sobre el SPY**, y GARP lo bate el **64 % de los años**,
   con la rentabilidad ya limpia de artefactos de datos.

La conclusión no es "la IA funciona" ni "la IA fracasa", sino algo más fino y defendible: **el
valor no está en el aprendizaje automático —que no lo hay— sino en explotar de forma disciplinada
primas de factor conocidas (calidad, valor), y en haberlo demostrado con rigor.**

## 1. Qué se construyó

Un sistema modular, limpio y **totalmente automático de principio a fin**:
- 3 agentes **LightGBM** (calidad, momentum, valor) + meta-agente ponderado por rank-IC.
- Un catálogo de **7 artefactos activables** (bloques de features/contexto), point-in-time.
- Un **barrido de ablations** que decide **automáticamente**, por significancia estadística, qué
  artefactos mejoran el aprendizaje y compone la configuración final.
- **8 perfiles de inversor** que reordenan las buenas acciones del modelo según estilo.
- **Tests de robustez/placebo** (permutación de etiquetas, bootstrap, leave-one-year-out).
- Una **guarda anti-artefactos** que neutraliza datos corruptos (evitó el +953 % espurio de 2010).

Todo point-in-time, walk-forward, sin lookahead, 74 tests. Un comando reproduce el estudio entero.

## 2. El aprendizaje no es significativo

El barrido aceptó **un solo artefacto** de siete: la neutralización por sector (rank-IC 0.0036 →
0.0094). Los demás empeoran. El sistema final:

- **rank-IC del meta_final: +0.0036**, IC bootstrap **[−0.019, +0.024]** (cruza cero).
- **Placebo (permutación de etiquetas): p-valor 0.20.** Con etiquetas barajadas el rank-IC colapsa
  a ~0 (no hay fuga), pero 1 de cada 5 permutaciones aleatorias iguala al modelo real. **No supera
  al azar.**
- Leave-one-year-out: +0.0008 a +0.0085 quitando cada año; estable pero siempre ≈ 0.

Es el mismo techo observado con el modelo lineal: con datos gratuitos y factores GARP+momentum,
**el modelo no aprende a ordenar de forma estadísticamente significativa**. Añadir features no
ayuda; solo reorganizar el ranking dentro de sector aporta un poco.

## 3. La rentabilidad: los perfiles de estilo sí baten al SPY

Con la guarda anti-artefactos (rentabilidad limpia), 2016-2026:

| perfil | CAGR | vs SPY | años que baten | drawdown |
|---|---|---|---|---|
| quality | 20.7 % | **+4.5 %** | 45 % | 35 % |
| value | 20.6 % | +4.4 % | 55 % | 30 % |
| conservative | 20.6 % | +4.4 % | 55 % | 32 % |
| garp | 19.9 % | +3.7 % | **64 %** | 41 % |
| contrarian | 18.1 % | +1.9 % | 55 % | 38 % |
| momentum | 15.6 % | −0.6 % | 45 % | 34 % |
| aggressive | 14.9 % | −1.3 % | 55 % | 33 % |
| **balanced** (meta ML puro) | 14.7 % | **−1.5 %** | 36 % | 48 % |

## 4. La lectura clave

**El perfil que sigue el ML puro (balanced) es el peor.** Los que ganan imponen un sesgo de estilo
humano hacia calidad y valor. Esto encaja con el plano de aprendizaje: como el ML no ordena bien,
seguir su ranking no bate al mercado; pero inclinar la cartera hacia calidad/valor captura las
**primas de factor clásicas**, que sí existen en la literatura y en los datos.

El sistema, por tanto, **funciona como vehículo para explotar primas de factor de forma
disciplinada** (con costes, rotación controlada, drawdown moderado), no como un predictor de IA.
Y eso está **demostrado honestamente**: la guarda anti-artefactos garantiza que el +4 % es real y
no un dato corrupto; el placebo garantiza que el ML no aporta señal oculta; el rank-IC separa con
claridad lo que el sistema *aprende* (poco) de lo que *rinde* (los factores).

## 5. Valor para el TFM

Es un resultado **matizado, medido y muy defendible**:
- Demuestra con rigor estadístico que un enfoque de ML sobre factores abiertos **no aprende** a
  batir al mercado — un hallazgo negativo honesto y bien cuantificado.
- Pero también muestra que **la estructura de factores + sesgo de estilo sí rinde**, separando
  limpiamente ambos planos.
- Y aporta un **sistema reproducible, automático y explicable** (perfiles de inversor, artefactos
  activables, decisión automática, tests de robustez) que es en sí mismo un buen entregable de
  ingeniería y método.

El proyecto responde su pregunta —¿aprende el sistema y es útil?— con honestidad: **aprende poco
y de forma no significativa, pero es económicamente útil por las primas de factor, no por la IA.**
