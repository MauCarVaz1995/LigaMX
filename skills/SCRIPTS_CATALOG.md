# SCRIPTS_CATALOG.md — Catálogo completo de scripts

> 27 scripts en `scripts/` + 2 en `scripts/dashboard/`. Actualizado al 2026-04-04.

---

## Leyenda de estado
- ✅ **Funcionando** — probado, output correcto
- 🔧 **En desarrollo** — funciona parcialmente o necesita ajustes
- 📦 **Estable/legado** — funciona, no se toca
- ⏳ **Pendiente** — descrito pero no implementado

---

## Infraestructura visual

### `config_visual.py`
| | |
|---|---|
| **Propósito** | Paletas de color, tipografía Bebas Neue, utilidades de imagen, banderas nacionales |
| **Inputs** | Ninguno (módulo de configuración) |
| **Outputs** | Funciones y constantes importables |
| **Exports clave** | `PALETAS`, `get_paleta()`, `bebas()`, `hex_rgb()`, `hex_rgba()`, `get_escudo()`, `make_h_gradient()` |
| **Estado** | ✅ Funcionando |
| **Notas** | `get_escudo()` sirve banderas de países (no escudos de Liga MX). Para Liga MX usar `get_escudo_ligamx()` definida en los scripts de predicción |

---

## Pipeline de datos — Liga MX

### `01_get_equipos.py`
| | |
|---|---|
| **Propósito** | Descarga los 18 equipos del torneo activo de FotMob |
| **Inputs** | FotMob API (league_id=230) |
| **Outputs** | `data/raw/equipos_clausura2026.json` |
| **Estado** | ✅ Funcionando |

### `02_get_jugadores.py`
| | |
|---|---|
| **Propósito** | Stats básicas de jugadores por equipo |
| **Inputs** | `equipos_clausura2026.json`, FotMob API |
| **Outputs** | `data/raw/jugadores/{id}_{equipo}.json` (18 archivos) |
| **Estado** | ✅ Funcionando |

### `02b_get_stats_jugadores.py`
| | |
|---|---|
| **Propósito** | Stats detalladas por jugador individual (per90, ratings) |
| **Inputs** | JSONs de jugadores, FotMob `players/{id}` |
| **Outputs** | `data/raw/stats_detalladas/{id}.json` |
| **Estado** | ✅ Funcionando |

### `02c_get_stats_liga.py`
| | |
|---|---|
| **Propósito** | Stats vía `data.fotmob.com` — complementa `02b` |
| **Inputs** | FotMob data API |
| **Outputs** | Stats adicionales de liga |
| **Estado** | ✅ Funcionando |

### `04_consolidar_dataframe.py`
| | |
|---|---|
| **Propósito** | Consolida todos los JSONs de jugadores en un DataFrame maestro |
| **Inputs** | `data/raw/jugadores/`, `data/raw/stats_detalladas/` |
| **Outputs** | `data/processed/jugadores_clausura2026.csv`, `.pkl` |
| **Estado** | ✅ Funcionando |

### `10_descargar_historico.py`
| | |
|---|---|
| **Propósito** | Descarga 38 torneos históricos (Apertura/Clausura 2010/11→2025/26) |
| **Inputs** | FotMob leagues API, league_id=230 |
| **Outputs** | `data/raw/historico/historico_{año}_-_{torneo}.json` |
| **Dependencias** | Ninguna |
| **Estado** | ✅ Funcionando |
| **Notas** | Flag `--force` para re-descargar todo. Los años usan guión (`2025-2026`), no slash |

---

## Visualización — Jugadores

### `05_radar_p90.py`
| | |
|---|---|
| **Propósito** | Pizza chart P90 individual, percentiles vs posición |
| **Inputs** | `jugadores_clausura2026.csv` |
| **Outputs** | `output/charts/pizza_{nombre}_{equipo}.png` |
| **Dependencias** | `config_visual.py`, `mplsoccer` |
| **Estado** | ✅ Funcionando |
| **Notas** | 8 métricas por posición, normalización vs ≥300 min misma posición |

### `05_viz_player_performance.py`
| | |
|---|---|
| **Propósito** | Infografías post-partido: ratings por equipo + team stats comparativa |
| **Inputs** | JSON de partido FotMob (`03_get_match_stats_fotmob.py`) |
| **Outputs** | `output/charts/partidos/{COD}_ratings_{equipo}.png`, `_team_stats.png` |
| **Dependencias** | `config_visual.py`, `03_get_match_stats_fotmob.py` |
| **Estado** | ✅ Funcionando |
| **Notas** | Foto del jugador destacado descargada de FotMob. Tamaño ratings: 1080×1500 |

### `07_ranking_posicion.py`
| | |
|---|---|
| **Propósito** | Top 10 jugadores por posición en barras visuales |
| **Inputs** | `jugadores_clausura2026.csv` |
| **Outputs** | `output/charts/ranking_{posicion}.png` |
| **Estado** | ✅ Funcionando |

### `08_comparativo_1v1.py`
| | |
|---|---|
| **Propósito** | Infografía cara a cara entre dos jugadores, barras espejo |
| **Inputs** | `jugadores_clausura2026.csv`, IDs de jugadores |
| **Outputs** | `output/charts/comparativo_{j1}_vs_{j2}.png` |
| **Estado** | ✅ Funcionando |

### `03_radar_jugador.py`
| | |
|---|---|
| **Propósito** | Radar chart alternativo (versión anterior de pizza) |
| **Inputs** | `jugadores_clausura2026.csv` |
| **Outputs** | `output/charts/radar_p90_{nombre}.png` |
| **Estado** | 📦 Estable/legado |

---

## Visualización — Scraping de partidos

### `03_get_match_stats_fotmob.py`
| | |
|---|---|
| **Propósito** | Scraping completo de un partido individual vía `__NEXT_DATA__` |
| **Inputs** | Nombre local, visitante, fecha |
| **Outputs** | `data/raw/fotmob/{Local}_{Visit}_{fecha}.json` |
| **Estado** | ✅ Funcionando |
| **Notas** | Evita Cloudflare extrayendo JSON embebido. Uso: `python 03... Mexico Portugal 2026-03-29` |

---

## Modelos y predicciones — Liga MX

### `11_modelo_prediccion.py` ⭐
| | |
|---|---|
| **Propósito** | Modelo Poisson ponderado por torneo. Diseño con **círculos de %** + heatmap 7×7 |
| **Inputs** | `data/raw/historico/` (38 JSONs) |
| **Outputs** | `output/charts/prediccion_{local}_vs_{visitante}.png` |
| **Dependencias** | `config_visual.py`, `scipy.stats.poisson` |
| **Estado** | ✅ Funcionando |
| **Notas** | Usar este para Liga MX cuando se quieran círculos. mu=1.421, 608 partidos, 4 torneos ponderados. HOME_ADV=1.15× multiplicador lambda |

### `12_modelo_elo.py`
| | |
|---|---|
| **Propósito** | ELO histórico Liga MX: evolución 15 años + ranking actual |
| **Inputs** | `data/raw/historico/` |
| **Outputs** | `data/processed/elo_historico.csv`, `elo_evolucion.png`, `elo_ranking.png` |
| **Estado** | ✅ Funcionando |
| **Notas** | K=32, HOME_ADV=100, REGRESSION=30% entre torneos |

### `gen_predicciones_ligamx_20260404.py` ⭐
| | |
|---|---|
| **Propósito** | Predicciones Liga MX con diseño actual: heatmap degradado + bloques de % + IC 95% + escudos |
| **Inputs** | `elo_historico.csv`, `data/raw/logos/ligamx/` |
| **Outputs** | `output/charts/predicciones/LigaMX_Clausura_2026/pred_{Local}_{Visit}.png` |
| **Dependencias** | `config_visual.py`, `scipy`, `PIL` |
| **Estado** | ✅ Funcionando |
| **Notas** | Paletas restringidas: `medianoche_neon`, `oceano_esmeralda`, `rojo_fuego`. `--only N` para validar un solo partido. IC 95% bootstrap n=1000 |

### `15_prediccion_elo_poisson.py`
| | |
|---|---|
| **Propósito** | Predicciones comparativas ELO vs Poisson puro (versión jornada 13) |
| **Inputs** | `elo_historico.csv`, `data/raw/historico/` |
| **Outputs** | `output/charts/predicciones_j13_comparativo.png` |
| **Estado** | 📦 Estable/legado |

### `14_simulacion_montecarlo.py`
| | |
|---|---|
| **Propósito** | Monte Carlo 10,000 simulaciones del torneo completo |
| **Inputs** | Tabla actual + partidos restantes (hardcoded por jornada) |
| **Outputs** | `output/charts/montecarlo_clausura2026.png` |
| **Estado** | ✅ Funcionando |

### `14_paletas_montecarlo.py`
| | |
|---|---|
| **Propósito** | Genera el heatmap Monte Carlo en las 6 paletas distintas |
| **Inputs** | Mismos que `14_simulacion_montecarlo.py` |
| **Outputs** | `output/charts/paletas/montecarlo_*.png` |
| **Estado** | ✅ Funcionando |

### `04_predicciones_tracker.py`
| | |
|---|---|
| **Propósito** | Registra predicciones y resultados reales en CSV, genera reporte de desempeño |
| **Inputs** | `data/processed/predicciones_log.csv` |
| **Outputs** | Append al CSV, reporte en consola |
| **Estado** | ✅ Funcionando |
| **Notas** | Columnas: fecha_prediccion, partido, ELOs, probs, marcador, paleta, resultado_real, acierto |

---

## Selecciones nacionales

### `18_prediccion_selecciones.py`
| | |
|---|---|
| **Propósito** | ELO selecciones + ranking top 20 + últimos 5 + predicción ELO+Poisson |
| **Inputs** | `data/raw/internacional/results.csv` |
| **Outputs** | `selecciones_ranking_elo.png`, `selecciones_ultimos5.png`, `selecciones_prediccion.png` |
| **Estado** | ✅ Funcionando |
| **Notas** | K dinámico 20/25/35/60 por importancia. EXTRA_MEXICO para partidos manuales |

### `19_predicciones_hoy.py`
| | |
|---|---|
| **Propósito** | Genera predicciones del día para selecciones (varios partidos) |
| **Inputs** | ELOs hardcoded o desde JSON, lista de partidos del día |
| **Outputs** | `output/charts/predicciones_hoy/prediccion_{COD}.png` |
| **Estado** | ✅ Funcionando |

### `update_intl_results.py`
| | |
|---|---|
| **Propósito** | Descarga resultados internacionales de FotMob y los agrega a results.csv |
| **Inputs** | FotMob API `api/data/matches?date=`, `results.csv` |
| **Outputs** | `data/raw/internacional/results.csv` actualizado, `data/raw/fotmob/intl_{fecha}.json` |
| **Estado** | ✅ Funcionando |
| **Notas** | Filtra por `ccode == "INT"`. Deduplicación por (fecha, home, away) |

### `update_elo_selecciones.py`
| | |
|---|---|
| **Propósito** | Aplica nuevos partidos al ELO de selecciones de forma incremental |
| **Inputs** | `elos_selecciones_{fecha}.json`, `results.csv` (partidos nuevos) |
| **Outputs** | `data/processed/elos_selecciones_{nueva_fecha}.json`, tabla de cambios |
| **Estado** | ✅ Funcionando |
| **Notas** | Filtra U21, femenino, club. Normaliza nombres (Czechia→Czech Republic, Turkiye→Turkey, etc.) |

### `gen_predicciones_20260331.py`
| | |
|---|---|
| **Propósito** | Predicciones de partidos FIFA del 31-mar al 01-abr 2026 |
| **Inputs** | `fixtures_20260330_20260404.json`, `elos_selecciones_20260329.json` |
| **Outputs** | `output/charts/predicciones/pred_{COD}.png` |
| **Estado** | 📦 Estable/legado (fecha específica) |

### `gen_predicciones_intl_3img.py`
| | |
|---|---|
| **Propósito** | Genera 3 imágenes por partido: heatmap + ranking ELO + últimos 5 |
| **Inputs** | `elos_selecciones_*.json`, `results.csv` |
| **Outputs** | 3 PNG por partido en `output/charts/predicciones/` |
| **Estado** | ✅ Funcionando |

---

## Jornada / Post-jornada

### `12_resumen_jornada.py`
| | |
|---|---|
| **Propósito** | Infografía con todos los partidos de una jornada y sus probabilidades |
| **Inputs** | `elo_historico.csv`, partidos configurados |
| **Outputs** | `output/charts/jornada{N}/resumen_jornada{N}.png` |
| **Estado** | ✅ Funcionando |

### `13_resumen_postjornada.py`
| | |
|---|---|
| **Propósito** | Predicción vs resultado real, visual post-jornada |
| **Inputs** | Predicciones previas, resultados reales |
| **Outputs** | `output/charts/resumen_postjornada{N}.png` |
| **Estado** | ✅ Funcionando |

---

## Dashboard

### `dashboard/app.py`
| | |
|---|---|
| **Propósito** | Dashboard interactivo Dash con 6 páginas |
| **Inputs** | `elo_historico.csv`, `jugadores_clausura2026.csv`, `historico/` |
| **Outputs** | Servidor web http://localhost:8050 |
| **Estado** | ✅ Funcionando |
| **Notas** | Carga modelo Poisson + Monte Carlo al arrancar (~3s). Tema DARKLY |

### `dashboard/pages/home.py`
| | |
|---|---|
| **Propósito** | Página HOME del dashboard |
| **Estado** | ✅ Funcionando |
