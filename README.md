# ⚽ LigaMX Stats

> Motor de análisis estadístico de la **Liga MX** construido íntegramente sobre datos de [FotMob](https://www.fotmob.com).
> Scraping, limpieza, feature engineering per-90, visualizaciones de élite y un **modelo de Poisson** para predicción de marcadores — todo en Python puro.

---

## 🖼️ Galería

### Rankings Top 10 por posición — Clausura 2026

| Delanteros | Mediocampistas |
|:---:|:---:|
| ![Ranking Delantero](output/charts/ranking_delantero.png) | ![Ranking Mediocampista](output/charts/ranking_mediocampista.png) |

| Defensas | Porteros |
|:---:|:---:|
| ![Ranking Defensa](output/charts/ranking_defensa.png) | ![Ranking Portero](output/charts/ranking_portero.png) |

---

### Pizza Chart P90 (perfil individual)

Métricas per-90 normalizadas contra todos los jugadores de la misma posición.
Fondo semitransparente con foto del jugador integrada.

![Pizza Paulinho](output/charts/pizza_paulinho_toluca.png)

---

### Comparativo 1 vs 1

Infografía cara a cara con barras espejo coloreadas por equipo, rating FotMob y conteo de victorias por métrica.

![1v1 Paulinho vs Ángel Sepúlveda](output/charts/comparativo_paulinho_vs_angel_sepulveda.png)

---

### Predicción de marcador — Modelo de Poisson

Heatmap de probabilidades de cada marcador posible (0-0 a 5-5), calculado con los últimos **4 torneos ponderados**.

![Predicción Pachuca vs Toluca](output/charts/prediccion_pachuca_vs_toluca.png)

---

### Resumen de Jornada

Infografía de todos los partidos de una jornada: barras de probabilidad por equipo, marcador más probable y escudos de equipo.

![Resumen Jornada 13](output/charts/jornada13/resumen_jornada13.png)

---

## 🗂️ Estructura del proyecto

```
LigaMX_Stats/
│
├── data/
│   ├── raw/
│   │   ├── equipos_clausura2026.json      # IDs y nombres de los 18 equipos
│   │   ├── historico/                     # 32 JSONs — torneos 2010/11 → 2025/26
│   │   ├── jugadores/                     # Stats básicas por equipo (JSON x equipo)
│   │   └── images/
│   │       ├── players/                   # Fotos de jugadores (cache local)
│   │       └── teams/                     # Escudos de equipos (cache local)
│   └── processed/
│       ├── jugadores_clausura2026.csv     # DataFrame maestro — stats limpias + P90
│       └── jugadores_clausura2026.pkl     # Misma data en formato pickle
│
├── scripts/
│   ├── 01_get_equipos.py                  # Paso 1 — Equipos del torneo activo
│   ├── 02_get_jugadores.py                # Paso 2 — Stats básicas de jugadores
│   ├── 02b_get_stats_jugadores.py         # Paso 2b — Stats detalladas por jugador
│   ├── 02c_get_stats_liga.py              # Paso 2c — Stats vía data.fotmob.com
│   ├── 03_radar_jugador.py                # Radar clásico con mplsoccer
│   ├── 04_consolidar_dataframe.py         # Consolida todos los JSONs en CSV/PKL
│   ├── 05_radar_p90.py                    # Pizza chart P90 estilo Statiskicks
│   ├── 07_ranking_posicion.py             # Top 10 por posición con foto y escudo
│   ├── 08_comparativo_1v1.py              # Infografía comparativa cara a cara
│   ├── 10_descargar_historico.py          # Histórico completo (32 torneos)
│   ├── 11_modelo_prediccion.py            # Modelo Poisson + heatmap de probabilidades
│   └── 12_resumen_jornada.py              # Resumen visual de todos los partidos de jornada
│
├── output/
│   └── charts/                            # PNGs generados
│       ├── ranking_*.png                  # Rankings por posición
│       ├── pizza_*.png                    # Pizza charts individuales
│       ├── radar_p90_*.png                # Radares P90
│       ├── comparativo_*_vs_*.png         # Comparativos 1v1
│       ├── prediccion_*_vs_*.png          # Heatmaps de predicción
│       └── jornada13/                     # Predicciones individuales + resumen
│
└── notebooks/                             # Exploración interactiva (Jupyter)
```

---

## 🛠️ Scripts — Guía de uso

### Pipeline de datos (ejecutar en orden la primera vez)

```bash
# 1. Obtener equipos del torneo activo
.venv/bin/python scripts/01_get_equipos.py
# → data/raw/equipos_clausura2026.json

# 2. Descargar stats de jugadores por equipo
.venv/bin/python scripts/02_get_jugadores.py
# → data/raw/jugadores/{id}_{equipo}.json  (18 archivos)

# 3. (Opcional) Stats detalladas por jugador individual
.venv/bin/python scripts/02b_get_stats_jugadores.py

# 4. Consolidar todo en un DataFrame maestro
.venv/bin/python scripts/04_consolidar_dataframe.py
# → data/processed/jugadores_clausura2026.csv
```

---

### `05_radar_p90.py` — Pizza chart individual

Genera el perfil visual de un jugador con métricas per-90 normalizadas contra su posición.
Fondo con foto semitransparente, escudo del equipo y Bebas Neue como tipografía.

```bash
# Por defecto genera el pizza chart de Paulinho (Toluca)
.venv/bin/python scripts/05_radar_p90.py

# Especificando jugador por nombre (parcial, case-insensitive)
.venv/bin/python scripts/05_radar_p90.py --nombre "Sepúlveda"
# → output/charts/pizza_angel_sepulveda_chivas.png
```

Métricas incluidas según posición:
- **Delanteros / Mediocampistas**: Goles, xG, Tiros, Tiros a Puerta, Asistencias, xA, Chances Creadas, Pases Precisos, Duelos Ganados, Regates
- **Defensas**: Intercepciones, Despejes, Duelos Aéreos, Entradas, Pases Precisos
- **Porteros**: Paradas, Goles Concedidos, Porterías en Cero, Distribución

---

### `07_ranking_posicion.py` — Top 10 por posición

Ranking visual de los mejores jugadores por posición, con Índice Compuesto basado en percentiles.

- Requiere ≥ 300 minutos jugados
- Barras con gradiente rojo → verde
- Score Total como círculo de color (verde top 3 / amarillo top 7 / naranja resto)
- Foto circular y escudo del equipo por jugador

```bash
# Todas las posiciones (genera 4 imágenes)
.venv/bin/python scripts/07_ranking_posicion.py

# Una posición específica
.venv/bin/python scripts/07_ranking_posicion.py --posicion Delantero
# Opciones: Delantero | Mediocampista | Defensa | Portero
# → output/charts/ranking_delantero.png
```

---

### `08_comparativo_1v1.py` — Comparativo cara a cara

Infografía de dos jugadores con barras espejo coloreadas por equipo, rating FotMob y contador de victorias por métrica.

**Diseño:**
- Header bicolor con foto circular, escudo inline y datos del jugador
- Círculo de rating (verde ≥7.5 / naranja ≥7.0 / rojo <7.0)
- 10 métricas P90 enfrentadas — ganador en color del equipo, perdedor en gris
- Banner inferior con conteo de victorias

```bash
# Por defecto: Paulinho vs Ángel Sepúlveda
.venv/bin/python scripts/08_comparativo_1v1.py

# Con IDs de FotMob específicos
.venv/bin/python scripts/08_comparativo_1v1.py 361377 215428
# → output/charts/comparativo_paulinho_vs_angel_sepulveda.png
```

> Los IDs de jugador se obtienen de la URL de FotMob: `fotmob.com/players/361377/paulinho`

---

### `10_descargar_historico.py` — Histórico completo Liga MX

Descarga **32 torneos** (Apertura/Clausura 2010/11 → 2025/26) con resultados partido a partido y tabla de posiciones.

```bash
# Descarga todos los torneos (salta los ya existentes)
.venv/bin/python scripts/10_descargar_historico.py

# Forzar re-descarga del torneo actual
.venv/bin/python scripts/10_descargar_historico.py --force
# → data/raw/historico/historico_{año}_-_{torneo}.json  (32 archivos)
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
      "jornada": 1,
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

---

### `11_modelo_prediccion.py` — Modelo de Poisson

Predice la distribución de probabilidad de marcadores para cualquier partido de Liga MX.

#### Cómo funciona

El modelo usa la distribución de **Poisson** para estimar los goles esperados de cada equipo:

```
λ_local     = ataque_local(A)     × defensa_visitante(B) × μ_home
λ_visitante = ataque_visitante(B) × defensa_local(A)     × μ_away
```

Los factores se calculan con los últimos **4 torneos ponderados**:

| Torneo | Peso |
|--------|------|
| Clausura 2026 | **4** |
| Apertura 2025 | **3** |
| Clausura 2025 | **2** |
| Apertura 2024 | **1** |

Se calcula la **matriz de probabilidades 6×6** (marcadores 0-0 a 5-5) normalizada al 100%:

```
P(local=i, visitante=j) = Poisson(i; λ_local) × Poisson(j; λ_visitante)
```

```bash
.venv/bin/python scripts/11_modelo_prediccion.py "Pachuca" "Toluca"
# → output/charts/prediccion_pachuca_vs_toluca.png
```

**Salida de ejemplo (Clausura 2026, Jornada 12):**
```
P(Victoria Pachuca) = 23.7%
P(Empate)           = 29.9%
P(Victoria Toluca)  = 46.4%
Score más probable:  0-1  (16.5%)
```

---

### `12_resumen_jornada.py` — Resumen visual de jornada

Genera una infografía vertical con todos los partidos de una jornada:
barras de probabilidad coloreadas por equipo, marcador más probable y escudos.

```bash
# Editar la lista PARTIDOS dentro del script con los duelos de la jornada
.venv/bin/python scripts/12_resumen_jornada.py
# → output/charts/jornada13/resumen_jornada13.png
```

---

## 🚀 Setup

```bash
# 1. Clonar repositorio
git clone git@github.com:MauCarVaz1995/LigaMX.git
cd LigaMX

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib Pillow requests scipy numpy mplsoccer

# 3. (Recomendado) Instalar fuente Bebas Neue para las visualizaciones
mkdir -p ~/.fonts
curl -sL "https://github.com/dharmatype/Bebas-Neue/raw/master/fonts/ttf/BebasNeue-Regular.ttf" \
     -o ~/.fonts/BebasNeue.ttf
fc-cache -fv
```

---

## 📊 Datos — Fuentes

Todos los datos provienen de **FotMob** mediante scraping del JSON embebido `__NEXT_DATA__` o de la API no oficial `data.fotmob.com`. No se usa ninguna API oficial ni de terceros pagos.

| Endpoint | Contenido |
|----------|-----------|
| `fotmob.com/leagues/230/...` | Página del torneo Clausura 2026 |
| `fotmob.com/leagues/230/matches?season=...` | Resultados por torneo (histórico) |
| `data.fotmob.com/leagues?id=230&...` | Stats de jugadores del torneo activo |
| `fotmob.com/players/{id}/...` | Stats detalladas por jugador |
| `images.fotmob.com/image_resources/playerimages/{id}.png` | Fotos de jugadores |
| `images.fotmob.com/image_resources/logo/teamlogo/{id}.png` | Escudos de equipos |

> Liga MX = `league_id: 230` en FotMob.

---

## 🧱 Stack

| Librería | Uso |
|----------|-----|
| `pandas` | Manipulación, limpieza y feature engineering |
| `matplotlib` | Todas las visualizaciones estáticas |
| `Pillow` (PIL) | Imágenes circulares, escudos, composición RGBA |
| `requests` / `urllib` | Scraping HTTP |
| `scipy` | Distribución de Poisson para el modelo |
| `numpy` | Álgebra lineal, matrices de probabilidad |
| `mplsoccer` | Radar chart base (script 03) |

---

## 🗺️ Roadmap

- [x] Scraping de equipos y jugadores (Clausura 2026)
- [x] DataFrame maestro con stats P90 por posición
- [x] Pizza chart P90 individual (Statiskicks style)
- [x] Ranking visual Top 10 por posición
- [x] Comparativo 1v1 con barras espejo
- [x] Histórico completo 15 años (32 torneos, ~6 000 partidos)
- [x] Modelo de Poisson para predicción de marcadores
- [x] Resumen visual de jornada completa
- [ ] Dashboard interactivo (Streamlit / Dash)
- [ ] Predicciones automáticas para la jornada vigente
- [ ] Análisis de rachas y forma reciente
- [ ] Modelo ELO por equipo
- [ ] Comparativo de temporadas (rendimiento histórico por equipo)

---

<p align="center">
  <strong>MAU-STATISTICS</strong> · Datos: FotMob · Liga MX 2010 – 2026
</p>
