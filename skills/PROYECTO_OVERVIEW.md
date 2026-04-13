# PROYECTO_OVERVIEW.md — LEE ESTO PRIMERO

> Estado del proyecto al **2026-04-12**. Actualizar al cierre de cada sesión significativa.

---

## ¿Qué es esto?

**MAU-STATISTICS / @Miau_Stats_MX** — Motor de análisis estadístico de Liga MX y Selecciones Nacionales.
Produce infografías de élite para redes sociales, predicciones de partidos y un dashboard interactivo.
Todo en Python puro sobre datos de FotMob.

Autor: **MauCarVaz1995** · GitHub: `MauCarVaz1995/LigaMX`

---

## Estado actual — qué funciona HOY

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
- CSV maestro: `data/raw/internacional/results.csv` (49,231 partidos, actualizado al 2026-04-12) ✅

### Tracker de predicciones ✅
- Log: `data/processed/predicciones_log.csv`
- Script: `scripts/04_predicciones_tracker.py`
- Columnas: fecha, partido, ELOs, probabilidades, marcador probable, paleta, resultado real, acierto
- **22 predicciones evaluadas, 50% acierto baseline** ✅

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

| Dataset | Ruta | Contenido |
|---|---|---|
| ELO Liga MX | `data/processed/elo_historico.csv` | ELO por equipo y fecha, 38 torneos |
| ELO Selecciones | `data/processed/elos_selecciones_20260412.json` | ~335 selecciones, actualizado al 12-abr-26 |
| Resultados internacionales | `data/raw/internacional/results.csv` | 49,231 partidos desde 1872 |
| Jugadores Clausura 2026 | `data/processed/jugadores_clausura2026.csv` | 41 columnas, stats P90 |
| Histórico Liga MX | `data/raw/historico/` | 38 JSONs, 2010/11→2025/26 |
| Predicciones | `data/processed/predicciones_log.csv` | Log de todas las predicciones hechas |
| Logos Liga MX | `data/raw/logos/ligamx/` | 10 escudos PNG 192×192 de FotMob |
| Banderas | `data/raw/flags/` | PNG de flagcdn.com, cacheadas |

---

## Próximos 3 pasos más importantes

### 1. CCL fixtures y predicciones
Identificar el league ID de la CONCACAF Champions Cup en FotMob, descargar fixtures, agregar como competición con K=35 en el modelo. Generar predicciones de semifinales/finales.

### 2. Twitter API bot
Validar credenciales en `20_twitter_bot.py`. Script de jornada completa: genera imágenes + publica hilo automático. GitHub Actions cron para días de jornada Liga MX.

### 3. xG FBref — Capa 3
Scraper xG FBref para Liga MX. Forma reciente ponderada como variable adicional. Scraper cuotas para tracker de value bets.

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
