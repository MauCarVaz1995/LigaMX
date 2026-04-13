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

## Completado hoy ✅ — 2026-04-13
- Pipeline GitHub Actions funcionando y confirmado verde
- Dixon-Coles implementado en `11_modelo_prediccion.py`, `15_prediccion_elo_poisson.py`, `18_prediccion_selecciones.py`
- Bebas Neue centralizada en `config_visual.py` — `FontProperties(fname=...)` directo, sin `addfont()`
- try/except defensivo en 15 scripts
- Jornadas 13 y 14 Liga MX actualizadas (125 partidos terminados)
- YAML syntax fix en workflow (heredoc para scripts Python inline)
- `.gitattributes` — TTF/imágenes marcados como binary

---

## Próximos pasos priorizados (actualizado 2026-04-13)

### Prioridad 1 — Esta semana
- [ ] Activar write permissions: GitHub repo → Settings → Actions → General → Read and write permissions
- [ ] CONCACAF Champions Cup: buscar league ID en FotMob, agregar como paso opcional en pipeline
- [ ] Git pull automático en desktop: configurar script o cron local para sincronizar imágenes

### Prioridad 1.5 — CONCACAF Champions Cup (detalle)
- [ ] Identificar fixture completo de la CCL en FotMob (league ID a buscar)
- [ ] Descargar resultados históricos de CCL para alimentar ELO
- [ ] Agregar CCL como competición en el modelo (K=35 — continental)
- [ ] Generar predicciones de semifinales/finales cuando aplique
- [ ] Contenido específico para @Miau_Stats_MX sobre CCL

### Prioridad 2 — Próximas 2 semanas
- [ ] Twitter API: agregar `TWITTER_API_KEY` etc como secrets en GitHub → Settings → Secrets, modificar pipeline Paso 6 para publicar automáticamente
- [ ] Notificación por correo: agregar paso en pipeline que mande resumen diario + imágenes a maucarvaz@gmail.com
- [ ] xG FBref: scraper para Capa 3 del modelo

### Prioridad 3 — Mes siguiente
- [ ] Value betting: scraper de cuotas + comparador vs modelo
- [ ] Kelly fraccionado: calcular tamaño óptimo de apuesta
- [ ] 200 predicciones en tracker para validar modelo antes de apostar dinero real

### Prioridad 4 — Infografías con Figma (cuando tengamos templates)
- [ ] Diseñar templates en Figma para cada tipo de imagen
- [ ] Script Python que llene templates via Figma API o exportación
- [ ] Reemplazar matplotlib por imágenes Figma renderizadas
- Objetivo: imágenes de calidad profesional sin depender de matplotlib

### Prioridad 5 — Engagement @Miau_Stats_MX
- [ ] Comentar en cuentas grandes con datos (ESPN, TUDN, Record)
- [ ] Publicar tracker público de aciertos del modelo
- [ ] Contenido semanal recurrente: ranking ELO lunes, predicciones viernes
- [ ] Encuestas vinculadas a predicciones del modelo

---

## Filosofía del proyecto
- Minimizar tokens de Claude: automatizar todo lo repetitivo
- Skills actualizados en cada sesión para recuperar contexto
- Modelo que mejora solo con más datos, no con más intervención manual
- Engagement orgánico basado en credibilidad del modelo vs resultados reales

---

## Notas de arquitectura

- El proyecto es **monorepo Python puro** — no hay microservicios
- Todo dato se descarga a `data/raw/`, se procesa a `data/processed/`
- Los scripts no tienen dependencias circulares: pipeline lineal 01→02→03→04→05...
- `config_visual.py` es el único módulo compartido entre scripts de visualización
- El venv está en `.venv/` — activar con `source .venv/bin/activate`
