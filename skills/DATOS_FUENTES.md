# DATOS_FUENTES.md — Fuentes de datos y estructura

---

## FotMob — Fuente principal

Liga MX `league_id = 230` en FotMob.

### Endpoints utilizados

| Endpoint | Datos | Script |
|---|---|---|
| `fotmob.com/api/data/matches?date=YYYYMMDD` | Todos los partidos del día (cualquier liga) | `update_intl_results.py` |
| `fotmob.com/es/leagues/230/overview/liga-mx` | Página torneo activo | `01_get_equipos.py` |
| `fotmob.com/es/leagues/230/matches/liga-mx?season=N` | Partidos por torneo | `10_descargar_historico.py` |
| `fotmob.com/api/table?leagueId=230&seasonId=N` | Tabla de posiciones | `10_descargar_historico.py` |
| `data.fotmob.com/leagues?id=230&...` | Stats de jugadores del torneo | `02c_get_stats_liga.py` |
| `fotmob.com/players/{id}/stats` | Stats individuales | `02b_get_stats_jugadores.py` |
| `fotmob.com/match/{id}/matchfacts/{slug}` | Stats completas de partido (via `__NEXT_DATA__`) | `03_get_match_stats_fotmob.py` |
| `images.fotmob.com/image_resources/playerimages/{id}.png` | Foto de jugador | `05_viz_player_performance.py` |
| `images.fotmob.com/image_resources/logo/teamlogo/{id}.png` | Escudo de equipo | `gen_predicciones_ligamx_20260404.py` |

### Notas de scraping
- FotMob usa Cloudflare. Los endpoints de API (`/api/...`, `data.fotmob.com`) funcionan directamente con headers apropiados
- La página de partido (`/match/{id}/matchfacts/`) requiere extraer `__NEXT_DATA__` del HTML
- Sufijo `_xh` en logos da **403 Forbidden** — usar URL sin sufijo (192×192px)
- Rate limit: respetar ~1.2s entre requests

### Headers mínimos necesarios
```python
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.fotmob.com/",
}
```

---

## Estructura de datos raw

### `data/raw/historico/` — 38 JSONs Liga MX
```
historico_2010-2011_-_apertura.json
historico_2010-2011_-_clausura.json
...
historico_2025-2026_-_clausura.json
```

**IMPORTANTE**: Los años usan guión (`2025-2026`), NO slash. Al cargar en modelo Poisson:
```python
tkey = f"{parts[0].replace('-','/').replace('_','/')} - ..."
```

Estructura de cada JSON:
```json
{
  "torneo": "2025/2026 - Clausura",
  "season_id": 27048,
  "tabla": [...],
  "partidos": [
    {
      "id": 4712345,
      "fecha": "2026-01-18",
      "jornada": "1",
      "local": "Pachuca",
      "visitante": "Toluca",
      "goles_local": 2,
      "goles_visit": 1,
      "score": "2 - 1",
      "terminado": true
    }
  ]
}
```

### `data/raw/fotmob/` — JSONs de partidos individuales
```
Mexico_Portugal_2026-03-29.json       ← scraping completo via __NEXT_DATA__
Colombia_France_2026-03-29.json
resultados_20260329.json              ← resultados del día (formato propio)
fixtures_20260330_20260404.json       ← fixtures de la ventana FIFA
intl_20260330.json                    ← respuesta cruda FotMob /api/data/matches
intl_20260331.json
intl_20260401.json
```

### `data/raw/internacional/results.csv` — Histórico de selecciones
- **49,080+ filas** desde 1872
- Fuente original: https://github.com/martj42/international_results
- Columnas: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`
- Actualizado al **2026-04-01** con `update_intl_results.py`
- Nombres de equipos del dataset (no de FotMob): "Mexico" (sin tilde), "Czech Republic", "Turkey", "United States", etc.

### `data/raw/logos/ligamx/` — Escudos Liga MX
```
América.png, Atlas.png, Cruz Azul.png, León.png,
Monterrey.png, Pachuca.png, Querétaro.png, Santos.png,
San Luis.png, Toluca.png
```
- Tamaño: 192×192px, modo PIL `P` (palette)
- Fuente: FotMob CDN (descargados el 2026-04-04)

### `data/raw/flags/` — Banderas nacionales
```
mx.png, us.png, pt.png, ...  (ISO 3166-1 alpha-2)
gb-eng.png, gb-wls.png, gb-sct.png  (subdivisiones UK)
xk.png  (Kosovo)
```
- Tamaño: 80×60px
- Fuente: flagcdn.com

---

## Datos procesados

### `data/processed/elo_historico.csv`
```
fecha, equipo, elo, torneo
2026-03-22, Toluca, 1709.04, 2025/2026 - Clausura
2026-03-22, Cruz Azul, 1691.77, 2025/2026 - Clausura
...
```
Un registro por equipo por partido jugado. El ELO de cada fila es el estado DESPUÉS del partido.

### `data/processed/elos_selecciones_20260401.json`
Diccionario plano `{nombre_selección: elo_float}` con ~335 entradas.
Actualizar nombre del archivo al generar nuevo estado.

### `data/processed/jugadores_clausura2026.csv`
DataFrame maestro con ~400 jugadores, 41 columnas incluyendo stats P90 por posición.
Columnas clave: `nombre`, `equipo`, `posicion`, `minutos`, `fotmob_id`, + todas las stats P90.

### `data/processed/predicciones_log.csv`
Log append-only de todas las predicciones generadas. Ver `MODELO_METODOLOGIA.md` para columnas.

---

## FBref — xG y stats avanzados (pendiente implementar)

FBref (StatsBomb) tiene xG, presión, pases progresivos. Crítico para modelo de betting.

### URLs relevantes Liga MX
```
https://fbref.com/en/comps/31/schedule/Liga-MX-Scores-and-Fixtures
https://fbref.com/en/comps/31/shooting/Liga-MX-Stats  ← xG por equipo
https://fbref.com/en/comps/31/Liga-MX-Stats           ← stats generales
```
- No tiene API pública, requiere scraping HTML con `BeautifulSoup`
- Rate limit estricto: ~3-5s entre requests o bloquea IP
- Datos disponibles desde ~2017/18 para Liga MX

### Script pendiente: `scrape_xg_fbref.py`
```
Output: data/processed/xg_equipos_clausura2026.csv
        data/processed/xg_jugadores_clausura2026.csv
Columnas: partido, fecha, equipo, xG, xGA, shots, shots_on_target
```

---

## The Odds API — Cuotas para value betting

**URL**: `https://api.the-odds-api.com/v4/sports/soccer_mexico_ligamx/odds/`
**Free tier**: 500 requests/mes, suficiente para uso diario en Liga MX
**Documentación**: the-odds-api.com

### Endpoints clave
```
GET /v4/sports/                          ← ligas disponibles
GET /v4/sports/{sport}/odds/             ← cuotas actuales
    ?apiKey=KEY&regions=eu&markets=h2h,totals,btts
GET /v4/sports/{sport}/scores/           ← resultados
GET /v4/sports/{sport}/historical/odds/  ← histórico (plan de pago)
```

### Script pendiente: `scrape_odds.py`
```python
MARKETS = ['h2h', 'totals', 'btts', 'asian_handicap', 'player_props']
# Output: data/processed/odds_ligamx_{fecha}.json
# Columnas: partido, mercado, casa, cuota, cuota_implicita, EV_modelo
```

---

## football-data.co.uk — Cuotas históricas gratuitas

URL: `https://www.football-data.co.uk/mexico.php`
Formato: CSV con columnas de cuotas de múltiples casas (B365, Max, Avg, etc.)
Cubre: Liga MX desde ~2012
Uso: backtesting histórico de value betting sin costo

### Script pendiente: `scrape_historical_odds.py`
```
Output: data/processed/historical_odds_ligamx.csv
```

---

## Banderas / Escudos externos

| Servicio | URL | Uso |
|---|---|---|
| flagcdn.com | `https://flagcdn.com/80x60/{iso}.png` | Banderas 80×60px para selecciones |
| FotMob CDN | `https://images.fotmob.com/image_resources/logo/teamlogo/{id}.png` | Escudos Liga MX 192×192px |
| FotMob CDN | `https://images.fotmob.com/image_resources/playerimages/{id}.png` | Fotos de jugadores |
