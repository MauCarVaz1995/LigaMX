# ROADMAP.md — Plan de desarrollo MAU-STATISTICS

> Estado al **2026-04-04**. Actualizar al completar cada fase.

---

## Fases completadas ✅

### Fase 1 — Infraestructura de datos Liga MX
- ✅ Scraping de 38 torneos históricos (2010/11 → 2025/26)
- ✅ Stats de jugadores (P90, ratings FotMob, fotos)
- ✅ DataFrame maestro `jugadores_clausura2026.csv` (400 jugadores, 41 columnas)
- ✅ ELO histórico Liga MX (15 años, K=32, HOME_ADV=100, regresión 30%)

### Fase 2 — Sistema visual
- ✅ Paletas de color intercambiables (7 paletas con 10 claves cada una)
- ✅ Tipografía Bebas Neue integrada en matplotlib
- ✅ `config_visual.py` como módulo central reutilizable
- ✅ Layout estándar 1080×1080px para predicciones

### Fase 3 — Modelo de predicción Liga MX
- ✅ Modelo Poisson ponderado por torneo (mu=1.421, 608 partidos, 4 torneos)
- ✅ ELO + ventaja local (+100) como fuente de lambdas
- ✅ IC 95% bootstrap Monte Carlo (n=1000)
- ✅ Escudos de equipos en header (FotMob CDN, caché local)
- ✅ Tracker de predicciones con log CSV

### Fase 4 — Selecciones nacionales
- ✅ ELO histórico desde 1872 (~335 selecciones)
- ✅ K dinámico por importancia de torneo (20/25/35/60)
- ✅ Multiplicador margen de goles
- ✅ Scraping automático de resultados internacionales (`update_intl_results.py`)
- ✅ Actualización incremental del ELO (`update_elo_selecciones.py`)
- ✅ Banderas nacionales de flagcdn.com (caché local)

### Fase 5 — Dashboard interactivo
- ✅ Dash + Plotly + Bootstrap DARKLY
- ✅ 6 páginas: Home, Jornada, ELO, Simulación, Jugadores, Comparativo
- ✅ Modelo Poisson cargado al arrancar (~3s)

### Fase 6 — Documentación del proyecto
- ✅ Sistema `skills/` con 7 archivos markdown
- ✅ README.md completo

---

## En desarrollo 🔧

### `gen_predicciones_ligamx_20260404.py` → script canónico
- ✅ Diseño con heatmap degradado + bloques de %
- ✅ IC 95% bootstrap
- ✅ Escudos en header
- ⏳ Unificar con diseño de círculos de `11_modelo_prediccion.py`

---

## Próximos pasos ⏳ (por orden de impacto)

### Corto plazo (próximas sesiones)

1. **Registrar resultados reales** en `predicciones_log.csv`
   ```bash
   python3 scripts/04_predicciones_tracker.py resultado "Cruz Azul vs Pachuca" 2026-04-04 X X
   python3 scripts/04_predicciones_tracker.py reporte
   ```

2. **Script canónico único de predicción Liga MX**
   - Fusionar `11_modelo_prediccion.py` (círculos) y `gen_predicciones_ligamx_20260404.py` (bloques)
   - Un solo script con flag `--style [bloques|circulos]`
   - Soporte para jornada completa (`--jornada N`) y partido único (`--partido "A vs B"`)

3. **Resumen de jornada automatizado**
   - `12_resumen_jornada.py` necesita los 9 partidos hardcoded actualmente
   - Meta: leer fixtures de FotMob automáticamente

### Mediano plazo

4. **Dixon-Coles correction**
   - Corrige subestimación de resultados bajos (0-0, 1-0, 0-1, 1-1)
   - Estimación de ρ por MLE sobre el histórico (esperado: -0.13 a -0.08)
   - Ver `MODELO_METODOLOGIA.md` para fórmula completa

5. **FBref scraping**
   - xG, presión, pases progresivos — métricas que FotMob no ofrece
   - Pendiente de implementar para Liga MX

6. **Predicciones automáticas por jornada**
   - Script que lee fixtures de FotMob, genera todas las predicciones sin configuración manual
   - Deduplicación vs predicciones ya generadas

7. **Bot de publicación automática**
   - Twitter API v2 para subir imágenes y texto
   - Programar tweets para 2h antes de cada partido

### Largo plazo

8. **Modelo ensemble**
   - Combinar ELO + Poisson + Dixon-Coles + features adicionales (forma reciente, bajas)
   - Calibración de probabilidades (Platt scaling)

9. **Cobertura multi-liga**
   - Expansión a otras ligas CONCACAF (MLS, Guatemalteca)
   - Mismo sistema ELO y pipeline visual

10. **API pública**
    - FastAPI con endpoints de predicciones
    - Rate limiting + caché Redis

---

## Notas de arquitectura

- El proyecto es **monorepo Python puro** — no hay microservicios
- Todo dato se descarga a `data/raw/`, se procesa a `data/processed/`
- Los scripts no tienen dependencias circulares: pipeline lineal 01→02→03→04→05...
- `config_visual.py` es el único módulo compartido entre scripts de visualización
- El venv está en `.venv/` — activar con `source .venv/bin/activate`
