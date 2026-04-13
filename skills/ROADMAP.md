# ROADMAP.md — Plan de desarrollo MAU-STATISTICS

> Estado al **2026-04-13**. Actualizar al completar cada fase.

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
- ✅ Dixon-Coles implementado en los 3 scripts de predicción (rho=-0.13)
- ✅ Jornadas 13 y 14 Liga MX Clausura 2026 actualizadas (125/153 terminados)
- ✅ Tracker con 22 predicciones evaluadas, 50% acierto baseline

### Fase 4 — Selecciones nacionales
- ✅ ELO histórico desde 1872 (~335 selecciones)
- ✅ K dinámico por importancia de torneo (20/25/35/60)
- ✅ Multiplicador margen de goles
- ✅ Scraping automático de resultados internacionales (`update_intl_results.py`)
- ✅ Actualización incremental del ELO (`update_elo_selecciones.py`)
- ✅ Banderas nacionales de flagcdn.com (caché local)
- ✅ 68 partidos internacionales 2-12 abril agregados (total: 49,231)
- ✅ ELO selecciones actualizado al 12-abril (`elos_selecciones_20260412.json`, México 1892.6)

### Fase 5 — Dashboard interactivo
- ✅ Dash + Plotly + Bootstrap DARKLY
- ✅ 6 páginas: Home, Jornada, ELO, Simulación, Jugadores, Comparativo
- ✅ Modelo Poisson cargado al arrancar (~3s)

### Fase 6 — Documentación del proyecto
- ✅ Sistema `skills/` con 7 archivos markdown
- ✅ README.md completo

---

## Principio rector — Infraestructura primero

> **Regla de oro**: Antes de construir un modelo o análisis, construir el job que recolecta y persiste los datos en GitHub de forma automática. Un modelo que corre sobre datos incompletos produce resultados incorrectos. La infraestructura de datos es la prioridad absoluta.

**Checklist antes de cualquier análisis nuevo:**
1. ¿Existe un GitHub Actions job que descarga y persiste los datos automáticamente?
2. ¿El historial es completo (no solo el torneo actual)?
3. ¿Los datos sobreviven si Claude no está disponible por semanas?
4. Solo después de responder SÍ a los tres: construir modelos encima.

**Lección CCL 2025-26**: Se construyó modelo ELO con solo 35 partidos del torneo actual. Los ELOs resultantes no son confiables porque no hay historial. Fix pendiente: descargar historial completo de Concacaf CL desde 2010 (~500 fechas en FotMob).

---

## Completado hoy ✅ — 2026-04-13
- Pipeline GitHub Actions funcionando y confirmado verde
- Dixon-Coles implementado en `11_modelo_prediccion.py`, `15_prediccion_elo_poisson.py`, `18_prediccion_selecciones.py`
- Bebas Neue centralizada en `config_visual.py` — `FontProperties(fname=...)` directo, sin `addfont()`
- YAML syntax fix en workflow (heredoc para scripts Python inline) + `.gitattributes`
- CONCACAF Champions Cup bootstrap: CCL id=915924, 22 logos, 35 fixtures 2025-26, 4 imágenes cuartos vuelta
- Scripts: `scrape_ccl_logos.py`, `update_ccl_fixtures.py`, `gen_predicciones_ccl.py`

---

## Próximos pasos priorizados (actualizado 2026-04-13)

### 🔴 Prioridad 0 — Infraestructura de datos (PRIMERO, siempre)
> Construir el job de GitHub Actions ANTES que el modelo. Sin datos completos en GitHub, no hay análisis confiable.

#### CCL — Historial completo (BLOCKER para ELOs confiables)
- [ ] **Job GitHub Actions `ccl_pipeline.yml`** — ya integrado en `daily_pipeline.yml` (Paso 8 pendiente de agregar)
- [ ] **Scraper histórico CCL**: descargar Concacaf Champions League desde 2010 (~500 fechas FotMob, league id=915924)
  - Script: `scripts/build_ccl_historical.py` — consulta fechas de partidos CCL históricos por año
  - Output: `data/raw/fotmob/ccl/ccl_historical_YYYYMMDD.json` por temporada
  - ELO arranca en 1500 por equipo y converge con ~15 años de datos
- [ ] **Liga MX ELO como base para equipos mexicanos** en CCL
  - Fix en `gen_predicciones_ccl.py`: si el equipo existe en `elo_historico.csv`, usar ese ELO como punto de partida en vez de 1500
  - Equipos beneficiados: Tigres, Toluca, América, Cruz Azul, Monterrey, etc.
- [ ] **MLS base ELOs**: asignar ELOs iniciales razonables para equipos MLS basados en rendimiento histórico en CCL
  - LAFC ~1600, Seattle ~1560, Galaxy ~1530, Nashville ~1510, etc.

#### Otros datos faltantes en GitHub Actions
- [ ] **Clausura 2026 completo** — el pipeline diario ya corre, verificar que no se pierdan partidos
- [ ] **Histórico internacional** — `results.csv` se actualiza diario ✅ ya funciona
- [ ] **Apertura/Clausura histórico** — 38 torneos ya descargados ✅ completo

### 🟡 Prioridad 1 — Esta semana
- [ ] Activar write permissions: GitHub repo → Settings → Actions → General → Read and write permissions
- [ ] Integrar CCL en `daily_pipeline.yml` como Paso 8 no-crítico: `update_ccl_fixtures.py` → `gen_predicciones_ccl.py`
- [ ] Git pull automático en desktop para sincronizar imágenes generadas

### 🟡 Prioridad 2 — Próximas 2 semanas
- [ ] Twitter API: secrets en GitHub → pipeline publica imágenes automáticamente
- [ ] Notificación diaria por correo: resumen + imágenes del día
- [ ] xG FBref: scraper para Capa 3 del modelo

### 🟢 Prioridad 3 — Mes siguiente
- [ ] Value betting: scraper de cuotas + comparador vs modelo
- [ ] Kelly fraccionado: tamaño óptimo de apuesta
- [ ] 200 predicciones en tracker para validar modelo antes de apostar

### 🟢 Prioridad 4 — Largo plazo
- [ ] Templates Figma para infografías de calidad profesional
- [ ] Engagement @Miau_Stats_MX: tracker público, contenido semanal recurrente

---

## Filosofía del proyecto
- **Infraestructura primero**: job de GitHub Actions que persiste datos ANTES de construir modelos
- Minimizar tokens de Claude: automatizar todo lo repetitivo, Claude solo para diseño y análisis
- Skills actualizados en cada sesión — Claude SIEMPRE lee skills antes de trabajar
- Modelo que mejora solo con más datos, no con más intervención manual
- Engagement orgánico basado en credibilidad del modelo vs resultados reales

---

## GitHub Actions Jobs — Estado

| Workflow | Schedule | Estado | Hace |
|---|---|---|---|
| `daily_pipeline.yml` | 8am México diario | ✅ activo | Liga MX + intl + ELO + tracker + predicciones |
| `ccl_pipeline.yml` | pendiente de crear | ⏳ | CCL fixtures + logos + predicciones |
| `historical_backfill.yml` | manual (una vez) | ⏳ | Descargar historial CCL 2010→2025 |

---

## Notas de arquitectura

- El proyecto es **monorepo Python puro** — no hay microservicios
- Todo dato se descarga a `data/raw/`, se procesa a `data/processed/`
- Los scripts no tienen dependencias circulares: pipeline lineal 01→02→03→04→05...
- `config_visual.py` es el único módulo compartido entre scripts de visualización
- El venv está en `.venv/` — activar con `source .venv/bin/activate`
- **Regla de naming para workflows**: `{competición}_pipeline.yml`, pasos no-críticos con `|| true`
