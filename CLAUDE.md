# Guía del repositorio

## Propósito

Este TFM estudia aprendizaje automático financiero de forma reproducible. La evidencia principal es el Rank-IC fuera de muestra; la rentabilidad y los perfiles de cartera son comprobaciones económicas posteriores.

## Arquitectura actual

```text
download → dataset point-in-time → features/bloques → agentes → meta → backtest → study OOS
```

- `module/data/`: ingesta, universo histórico y panel PIT.
- `module/modeling/`: catálogo de factores, features, agentes y meta-agente.
- `module/evaluation/`: cartera, perfiles y robustez.
- `module/runs/`: ejecución, caché, manifiestos y studies.
- `module/ui/` y `app/`: Research Console e informes.

Hay cinco agentes (`quality`, `value`, `growth`, `momentum`, `risk`), tres familias de modelo y un catálogo declarativo de bloques. La fuente técnica y metodológica vigente es `docs/doc.md`.

## Reglas metodológicas

1. No introducir lookahead: datos y etiquetas tienen fechas de disponibilidad separadas.
2. Seleccionar el modelo solo con Rank-IC OOS hasta 2024; 2025–2026 es reserva final.
3. No elegir semillas ni costes favorables: son robustez y estrés, respectivamente.
4. Conservar resultados negativos, ablaciones y manifiestos para reproducibilidad.
5. Cualquier cambio de hipótesis, datos, etiquetas, modelos o cartera requiere instrucción explícita del usuario.
6. Actualizar siempre la documentación relativa a los cambios aplicados.

## Desarrollo

- Mantener UTF-8 y revisar acentos después de editar documentación.
- No tocar `.env`, secretos ni resultados históricos sin autorización.
- Preferir código simple, pruebas causales y configuraciones declarativas.
- Eliminar código realmente inactivo en vez de mantener compatibilidad ficticia.

## Verificación

```powershell
python -m pytest tests/ -q
python -m ruff check .
```

No ejecutar descargas ni un full study largo sin autorización explícita. Para operación y documentación de usuario, consultar `README.md` y `docs/doc.md`.
