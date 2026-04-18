# PROYECTO_OVERVIEW.md — LEE ESTO PRIMERO

> Estado del proyecto al **2026-04-18**. Actualizar al cierre de cada sesión significativa.

---

## ¿Qué es esto?

**MAU-STATISTICS / @Miau_Stats_MX** — Motor de análisis estadístico de Liga MX y Selecciones Nacionales.
Produce infografías de élite para redes sociales, predicciones de partidos y un dashboard interactivo.
**Objetivo de monetización**: value betting en mercados de corners, BTTS y tarjetas en Liga MX.
Todo en Python puro sobre datos de FotMob.

Autor: **MauCarVaz1995** · GitHub: `MauCarVaz1995/LigaMX`

---

## Estado actual — 2026-04-13

### ✅ Funcionando
- Pipeline automatizado GitHub Actions — corre cada día 8am México sin intervención
  - Paso 1: Descarga resultados Liga MX de FotMob
  - Paso 2: Recalcula ELO Liga MX incremental
  - Paso 3: Descarga partidos internacionales
  - Paso 4: Recalcula ELO selecciones incremental
  - Paso 5: Actualiza tracker de predicciones con resultados reales
  - Paso 6: Genera imágenes de predicción de partidos del día
  - Paso 7: Hace commit y push automático al repo
- Modelo: ELO + Poisson + Dixon-Coles (rho=-0.13)
- Liga MX Clausura 2026: jornadas 1-14 completas (125 partidos)
- Internacionales: actualizados al 2026-04-13 (49,231 partidos)
- Tracker: 23 predicciones, 11/23 aciertos (47.8% baseline)
- Bebas Neue: cargada desde `assets/fonts/` en CI y local

### ❓ Preguntas frecuentes sobre el pipeline

**¿Dónde se guardan las imágenes?**
Las imágenes se generan en `output/charts/` dentro del repo de GitHub.
El Paso 7 hace commit y push automático — quedan guardadas en GitHub.
Para verlas/usarlas en tu PC: hacer `git pull` en tu terminal local.
NO se sincronizan automáticamente con tu desktop — necesitas hacer pull.

**¿Cómo sé que el job corrió bien?**
- GitHub → repo → Actions → checkmark verde = OK
- GitHub → repo → commits → busca "auto: daily update YYYY-MM-DD"
- O revisar `logs/daily_summary.json` en el repo

**¿Qué hace exactamente cada día?**
1. Descarga resultados de FotMob (Liga MX + internacionales)
2. Actualiza ELOs de todos los equipos afectados
3. Llena resultados reales en predicciones pasadas
4. Genera imágenes de predicción para partidos de hoy
5. Guarda todo en GitHub con commit automático

**¿Se mandan imágenes por correo?**
Aún NO — está en el roadmap. Se puede agregar con SendGrid o Gmail API.

**¿Se publican automáticamente en Twitter?**
Aún NO — falta configurar Twitter API keys como secrets en GitHub.

### ✅ Completado en esta sesión (2026-04-18)
- `generar_prediccion.py` — script canónico unificado (Liga MX + CCL + Intl, mismo diseño)
- `gen_postpartido.py` — ratings post-partido automáticos, guarda en `J{N}/postpartido/`
- `send_daily_email.py` — resumen diario con imágenes por sección a maucarvaz@gmail.com
- Pipeline completo: pasos 1-8 + email automático cada 8am México
- `skills/BETTING_MODEL.md` — arquitectura completa del modelo de value betting
- `scrape_match_events.py` — 126/126 partidos Clausura 2026 con corners, tarjetas, shots, xG ✅
  - μ corners = 9.32 | μ amarillas = 4.43 | μ xG_local = 1.48 | μ xG_visita = 1.16
  - Integrado en daily_pipeline.yml (paso diario no-crítico, `--days 4`)

### ⏳ Pendiente prioritario (actualizado 2026-04-18)
1. **Fase B1 BETTING** — `build_corners_dataset.py`: agregar 3+ temporadas históricas para calibrar λ
2. **Fase B1 BETTING** — `scrape_xg_fbref.py`: xG por partido y jugador desde FBref
3. **Fase B2 BETTING** — `modelo_corners.py` + `modelo_tarjetas.py` + `modelo_btts.py`
4. **Fase B3 BETTING** — `scrape_odds.py`: The Odds API wrapper para detectar value bets
5. CONCACAF Champions Cup — historial completo 2010→hoy (ELOs confiables)
6. Twitter API — publicación automática desde pipeline

---

## Sistemas disponibles (referencia rápida)

### Sistema de predicciones Liga MX ✅
- Script principal: `scripts/gen_predicciones_ligamx_20260404.py`
- Modelo: ELO histórico + Poisson + Dixon-Coles (rho=-0.13) + ventaja local (+100 ELO) ✅
- Output: 1080×1080px con heatmap degradado, círculos de % y IC 95% bootstrap
- Paletas Liga MX: `medianoche_neon`, `oceano_esmeralda`, `rojo_fuego`
- Escudos: descargados de FotMob CDN, cacheados en `data/raw/logos/ligamx/`
- Firma MAU-STATISTICS en footer con línea roja de borde
- Carpeta de salida estándar: `output/charts/predicciones/LigaMX_Clausura_2026/`

### Sistema ELO Selecciones ✅
- ELO histórico desde 1872, ~335 selecciones
- Scripts: `18_prediccion_selecciones.py`, `update_elo_selecciones.py`
- Último estado guardado: `data/processed/elos_selecciones_20260412.json` ✅
- K dinámico (20/25/35/60), multiplicador de margen de goles, sin regresión anual
- Genera 3 imágenes: ranking top 20, últimos 5 partidos, heatmap predicción
- Dixon-Coles implementado en `poisson_probs` (rho=-0.13) ✅

### Descarga de resultados internacionales ✅
- Script: `scripts/update_intl_results.py`
- Fuente: FotMob API `api/data/matches?date=YYYYMMDD`, filtro `ccode == "INT"`
- CSV maestro: `data/raw/internacional/results.csv` (49,231 partidos, actualizado al 2026-04-13) ✅

### Tracker de predicciones ✅
- Log: `data/processed/predicciones_log.csv`
- Script: `scripts/04_predicciones_tracker.py`
- Columnas: fecha, partido, ELOs, probabilidades, marcador probable, paleta, resultado real, acierto
- **23 predicciones evaluadas, 47.8% acierto baseline** ✅

### Dashboard interactivo ✅
- `scripts/dashboard/app.py` → http://localhost:8050
- 6 páginas: Home, Jornada, ELO, Simulación, Jugadores, Comparativo
- Stack: Dash + Plotly + Bootstrap DARKLY

### Infografías post-partido ✅
- Script: `scripts/05_viz_player_performance.py`
- Input: JSON de FotMob scrapeado con `03_get_match_stats_fotmob.py`
- Output: ratings por equipo (1080×1500) + team stats (1080×1080)

### Modelo Poisson Liga MX ✅
- Script: `scripts/11_modelo_prediccion.py`
- Círculos de % + división visual local/visitante + heatmap 7×7
- 4 torneos ponderados, mu=1.421, 608 partidos
- ELO histórico en `data/processed/elo_historico.csv`

### Pizza charts P90 ✅
- Script: `scripts/05_radar_p90.py`
- 100+ pizzas generadas para todo el plantel del Clausura 2026

---

## Datos disponibles

| Dataset | Ruta | Actualización | Confianza |
|---|---|---|---|
| ELO Liga MX | `data/processed/elo_historico.csv` | Daily (Actions) | ✅ Alta — 15 años, 38 torneos |
| ELO Selecciones | `data/processed/elos_selecciones_*.json` | Daily (Actions) | ✅ Alta — 49k partidos desde 1872 |
| Resultados internacionales | `data/raw/internacional/results.csv` | Daily (Actions) | ✅ Alta |
| Histórico Liga MX | `data/raw/historico/` | Daily (Actions) | ✅ Alta — 38 JSONs |
| Fixtures CCL 2025-26 | `data/raw/fotmob/ccl/ccl_fixtures_*.json` | Manual por ahora | ⚠️ Media — solo torneo actual |
| ELO CCL | calculado en memoria | Manual | ❌ Baja — falta historial 2010-2025 |
| Jugadores Clausura 2026 | `data/processed/jugadores_clausura2026.csv` | Manual | ⚠️ Media |
| Logos Liga MX | `data/raw/logos/ligamx/` | Manual | ✅ Alta |
| Logos CCL | `data/raw/logos/ccl/` | Manual | ✅ Alta — 22 equipos |
| Banderas | `data/raw/flags/` | Manual | ✅ Alta |

---

## Arquitectura de datos — Decisiones clave

### Regla fundamental: infraestructura antes que modelo
> Cada fuente de datos debe tener un GitHub Actions job que la actualice automáticamente ANTES de construir análisis sobre ella. Datos manuales = datos incompletos = modelo incorrecto.

### CCL — Deuda técnica documentada
El ELO de la CCL se calculó con solo 35 partidos (torneo 2025-26). **No es confiable** para equipos con pocas apariciones.

**Plan de remediación (pendiente):**
1. `scripts/build_ccl_historical.py` — scraper que recorre fechas históricas en FotMob para league id=915924. Estima ~500 fechas desde 2010. Persiste en `data/raw/fotmob/ccl/ccl_historical_{season}.json`.
2. Integrar en `gen_predicciones_ccl.py`: usar ELO Liga MX como punto de partida para equipos mexicanos (ya tienen ELO calibrado de 15 años).
3. MLS base ELOs manuales: LAFC ~1580, Seattle ~1560, LA Galaxy ~1530, Nashville ~1510, Cincinnati ~1490.
4. Una vez con historial completo: recalibrar todos los ELOs.

### FotMob API — Limitaciones conocidas
- Endpoint: `fotmob.com/api/data/matches?date=YYYYMMDD` — solo por fecha, no por competición
- Para historial de una liga específica: necesitas conocer las fechas de los partidos
- Rate limit: ~1 req/seg sin bloqueos. Con 0.5s de sleep entre requests es estable.
- League IDs relevantes: Liga MX=230, CCL=915924, Selecciones ccode=INT

### GitHub Actions — Jobs activos y pendientes

| Workflow | Trigger | Estado |
|---|---|---|
| `daily_pipeline.yml` | Diario 14:00 UTC | ✅ Activo |
| CCL en pipeline diario (Paso 8) | Al agregar a daily_pipeline | ⏳ Pendiente |
| `build_ccl_historical.yml` | Manual, una sola vez | ⏳ Pendiente |

---

## Próximos pasos (ver ROADMAP.md para detalle)

1. **Integrar CCL en pipeline diario** — Paso 8 no-crítico: `update_ccl_fixtures.py` → `gen_predicciones_ccl.py`
2. **Historial CCL** — `build_ccl_historical.py` + job manual de backfill
3. **Twitter API** — secrets en GitHub → publicación automática
4. **xG FBref** — Capa 3 del modelo

---

## Convenciones del proyecto

| Tema | Regla |
|---|---|
| Carpetas predicciones | `output/charts/predicciones/{Liga}_{Torneo}_{Año}/` |
| Paletas Liga MX | Solo: `medianoche_neon`, `oceano_esmeralda`, `rojo_fuego` |
| Paletas Selecciones | Todas las 7 disponibles, sin repetir consecutivas |
| Tamaño predicción | 1080×1080px @ 150 DPI (7.2×7.2 in) |
| Tamaño ratings | 1080×1500px @ 150 DPI |
| Firma | `MAU-STATISTICS` en footer, Bebas Neue 14pt, `pal['accent']`, alineado derecha |
| ELO Liga MX ventaja local | +100 puntos al cálculo de lambda (no al ELO almacenado) |
| ELO Selecciones ventaja local | HOME_ADV=100 en cálculo de We, solo si `neutral=False` |
| Escudos Liga MX | FotMob CDN `logo/teamlogo/{id}.png`, cache en `data/raw/logos/ligamx/` |
| Banderas internacionales | flagcdn.com, cache en `data/raw/flags/` |
| Normalización porcentajes | Siempre normalizar: `p_home/total`, `p_draw/total`, `p_away=1-ph-pd` |
| IC predicciones | Bootstrap n=1000 simulaciones Poisson, percentiles 2.5/97.5 |

---

## Cuentas / Publicación

- Twitter/X: **@Miau_Stats_MX** (pendiente de definir estrategia completa, ver `TWITTER_STRATEGY.md`)
- GitHub: `github.com/MauCarVaz1995/LigaMX`
