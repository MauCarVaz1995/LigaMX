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

## Estado actual — 2026-04-18

### ✅ Funcionando
- Pipeline automatizado GitHub Actions — corre cada día 8am México sin intervención
  - Paso 1: `00_daily_pipeline.py` — Liga MX + ELO + internacionales + ELO selecciones + tracker + predicciones
  - Paso 2: `scrape_match_events.py --days 4` — corners/tarjetas/xG (incremental)
  - Paso 3: `bots/retrain_bot.py` — re-calibra modelos si hay ≥5 partidos nuevos o ≥7 días
  - Paso 4: `bots/daily_betting_bot.py --days 3` — análisis value bets próximos 3 días
  - Paso 5: `scripts/gen_postpartido.py --days 2` — infografías post-partido
  - Paso 6: `scripts/update_ccl_fixtures.py` + `generar_prediccion.py --competition ccl`
  - Paso 7: git commit + push (incluye `output/reports/`)
  - Paso 8: `bots/audit_bot.py` — auditoría completa + depuración imágenes obsoletas
  - Paso 9: `scripts/send_daily_email.py` — email diario con imágenes relevantes + audit + betting
- Modelos científicos: Dixon-Coles MLE + Rue-Salvesen time decay + "feeling" Liga MX
- Liga MX corners/tarjetas: 794 partidos en 5 temporadas (`match_events.csv`)
  - μ corners = 9.32 | μ amarillas = 4.43 | μ xG_local = 1.48 | μ xG_visita = 1.16
  - Brier Score Over 8.5 corners = 0.127 (baseline naïve = 0.25, +49% mejora)
- Tracker: 23 predicciones, 47.8% baseline
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
- O revisar `output/reports/audit_{fecha}.json` en el repo

**¿Qué hace exactamente cada día?**
1. Descarga resultados de FotMob (Liga MX + internacionales)
2. Actualiza ELOs de todos los equipos afectados
3. Llena resultados reales en predicciones pasadas
4. Scrape corners/tarjetas/xG de partidos recientes
5. Re-calibra modelos si hay datos nuevos suficientes
6. Genera análisis de value bets para próximos 3 días
7. Genera infografías post-partido de jornadas recientes
8. Hace commit y push de todo
9. Audita integridad + depura imágenes obsoletas
10. Manda email con resumen + imágenes relevantes

**¿Se mandan imágenes por correo?**
SÍ — `send_daily_email.py` corre diario. Filtra solo imágenes relevantes (predicciones futuras, post-partido reciente). Auditoría con semáforo visual incluida.

**¿Se publican automáticamente en Twitter?**
Aún NO — falta configurar Twitter API keys como secrets en GitHub.

### ✅ Completado en sesiones 2026-04-18
- `scripts/liga_mx_knowledge.py` — "feeling" layer: rivalidades, altitud, árbitros, fases
- `scripts/modelo_corners.py` — Dixon-Coles MLE + Rue-Salvesen time decay + feeling
- `scripts/modelo_tarjetas.py` — Poisson calibrado sobre card_rate + rivalidades + árbitro
- `scripts/modelo_btts.py` — Dixon-Coles ρ=-0.13 + lambdas desde ELO
- `scripts/calcular_ev.py` — EV unificado CLI para los 3 modelos + cuotas manuales
- `bots/daily_betting_bot.py` — análisis autónomo, reporte HTML dark-themed + JSON
- `bots/retrain_bot.py` — re-entrena automáticamente cuando hay datos suficientes
- `bots/audit_bot.py` (expandido) — 6 secciones + GAP CHECK + auto-depuración imágenes
- `scripts/send_daily_email.py` (reescrito) — filtrado estricto de relevancia por fecha/jornada
- `skills/VISION_GLOBAL.md` — estrategia global de portafolio (Liga MX → LATAM → Europa)
- `scrape_match_events.py --seasons 5` — 794 partidos multi-temporada

### ⏳ Pendiente prioritario (actualizado 2026-04-18)
1. **Fix ELO naming** — 5 equipos sin ELO: "CF America", "Atletico de San Luis", "Mazatlan FC", "FC Juarez", "Queretaro FC" (normalización de nombres)
2. **Run retrain inicial** — `python bots/retrain_bot.py --force` para generar `retrain_log.json`
3. **Fase B3 BETTING** — `scrape_odds.py`: The Odds API wrapper (500 req/mes gratis)
4. **Historical odds backtest** — `football-data.co.uk` free CSVs para validar CLV vs Pinnacle
5. **CCL historial completo** — 2010→2025 (~500 fechas en FotMob league_id=915924)
6. **Twitter API** — publicación automática desde pipeline

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

### Sistema de Value Betting ✅
- Dataset: `data/processed/match_events.csv` — 794 partidos, 5 temporadas Liga MX
- Modelo corners: `scripts/modelo_corners.py` — Dixon-Coles MLE + Rue-Salvesen + Liga MX Knowledge
  - Brier Score Over 8.5 = 0.127 (baseline naïve 0.25, +49%)
  - home_adv MLE = 1.157x (vs hardcoded 1.07 anterior)
- Modelo tarjetas: `scripts/modelo_tarjetas.py` — Poisson + rivalidades + factor árbitro
- Modelo BTTS: `scripts/modelo_btts.py` — Dixon-Coles ρ=-0.13 + lambdas desde ELO
- EV unificado: `scripts/calcular_ev.py` — CLI con cuotas manuales, tags VALUE/borderline/❌
- Bot diario: `bots/daily_betting_bot.py` — analiza partidos 3 días, output HTML+JSON
- Bot reentrenamiento: `bots/retrain_bot.py` — re-calibra si ≥5 nuevos o ≥7 días
- Bot auditoría: `bots/audit_bot.py` — 6 secciones, GAP CHECK, auto-elimina imágenes obsoletas
- Knowledge base: `scripts/liga_mx_knowledge.py` — rivalidades, altitud, árbitros, fases, estadios
- Modelos guardados: `data/processed/corners_model.json`, `tarjetas_model.json`
- Reportes: `output/reports/betting_{fecha}.html` + `audit_{fecha}.json`
- Gate producción: 200 predicciones | Brier < 0.22 | CLV > 0% | ROI > 3%

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
| Match events (corners/xG) | `data/processed/match_events.csv` | Daily (Actions, `--days 4`) | ✅ Alta — 794 partidos, 5 temporadas |
| Modelos betting | `data/processed/corners_model.json`, `tarjetas_model.json` | Auto retrain_bot | ✅ Alta — MLE Dixon-Coles |
| Retrain log | `data/processed/retrain_log.json` | Auto retrain_bot | ✅ Alta |
| Reportes betting | `output/reports/betting_{fecha}.html` | Daily (Actions) | ✅ Alta |
| Reportes auditoría | `output/reports/audit_{fecha}.json` | Daily (Actions) | ✅ Alta |
| Fixtures CCL 2025-26 | `data/raw/fotmob/ccl/ccl_fixtures_*.json` | Daily (Actions paso 6) | ⚠️ Media — solo torneo actual |
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
