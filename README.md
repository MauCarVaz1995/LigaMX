# ⚽ LigaMX Stats

> Motor de análisis estadístico de la **Liga MX** construido íntegramente sobre datos de [FotMob](https://www.fotmob.com).
> Scraping, limpieza, métricas per-90, visualizaciones de élite, modelo de Poisson para predicción de marcadores y sistema de rating ELO histórico — todo en Python puro.

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

### Pizza Chart P90 — Perfil individual

Métricas per-90 normalizadas por percentil contra todos los jugadores de la misma posición.
Fondo semitransparente con foto del jugador integrada.

![Pizza Paulinho](output/charts/pizza_paulinho_toluca.png)

---

### Comparativo 1 vs 1

Infografía cara a cara con barras espejo coloreadas por equipo, rating FotMob y conteo de victorias por métrica.

![1v1 Paulinho vs Ángel Sepúlveda](output/charts/comparativo_paulinho_vs_angel_sepulveda.png)

---

### Predicción de marcador — Modelo de Poisson

Heatmap de probabilidades de cada marcador posible (0-0 a 5-5), calculado con los últimos 4 torneos ponderados.

![Predicción Pachuca vs Toluca](output/charts/prediccion_pachuca_vs_toluca.png)

---

### Resumen de Jornada

Infografía de todos los partidos de una jornada: barras de probabilidad por equipo, marcador más probable y escudos.

![Resumen Jornada 13](output/charts/jornada13/resumen_jornada13.png)

---

### Sistema ELO — Evolución histórica

15 años de evolución del rating ELO de los 18 equipos actuales (Apertura 2010/11 → Clausura 2026). Suavizado con rolling window de 8 semanas.

![ELO Evolución](output/charts/elo_evolucion.png)

---

### Sistema ELO — Ranking actual

Tabla de los 18 equipos ordenados por rating ELO al cierre del torneo más reciente, con barras de gradiente y delta respecto a la media (1500).

![ELO Ranking](output/charts/elo_ranking.png)

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
│   │   ├── stats_detalladas/              # Stats granulares por jugador (JSON x jugador)
│   │   └── images/
│   │       ├── players/                   # Fotos de jugadores (caché local)
│   │       └── teams/                     # Escudos de equipos (caché local)
│   └── processed/
│       ├── jugadores_clausura2026.csv     # DataFrame maestro — stats limpias + P90
│       ├── jugadores_clausura2026.pkl     # Misma data en formato pickle
│       └── elo_historico.csv             # Historial completo de ratings ELO
│
├── scripts/
│   ├── config_visual.py                   # Paleta y utilidades visuales centralizadas
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
│   ├── 12_resumen_jornada.py              # Resumen visual de todos los partidos de jornada
│   └── 12_modelo_elo.py                   # Sistema ELO histórico + evolución + ranking
│
├── output/
│   └── charts/                            # PNGs generados (150 DPI)
│       ├── ranking_*.png                  # Rankings por posición
│       ├── pizza_*.png                    # Pizza charts individuales
│       ├── comparativo_*_vs_*.png         # Comparativos 1v1
│       ├── prediccion_*_vs_*.png          # Heatmaps de predicción
│       ├── elo_evolucion.png              # Evolución ELO histórica
│       ├── elo_ranking.png                # Ranking ELO actual
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

# 2. Descargar stats básicas de jugadores por equipo
.venv/bin/python scripts/02_get_jugadores.py
# → data/raw/jugadores/{id}_{equipo}.json  (18 archivos)

# 3. Descargar stats detalladas por jugador individual
.venv/bin/python scripts/02b_get_stats_jugadores.py
# → data/raw/stats_detalladas/{id}.json

# 4. Consolidar todo en un DataFrame maestro
.venv/bin/python scripts/04_consolidar_dataframe.py
# → data/processed/jugadores_clausura2026.csv
```

---

### `05_radar_p90.py` — Pizza chart individual

Genera el perfil visual de un jugador con métricas per-90 normalizadas contra su posición. Fondo con foto semitransparente, escudo del equipo y Bebas Neue como tipografía.

```bash
# Por defecto genera el pizza chart de Paulinho (Toluca)
.venv/bin/python scripts/05_radar_p90.py

# Especificando jugador por nombre (parcial, case-insensitive)
.venv/bin/python scripts/05_radar_p90.py --nombre "Sepúlveda"
# → output/charts/pizza_angel_sepulveda_chivas.png

# Generar 3 ejemplos variados
.venv/bin/python scripts/05_radar_p90.py --todos
```

**Métricas por posición:**

| Posición | Métricas (8 radios) |
|---|---|
| **Delantero / CAM** | Goles P90, xG P90, Tiros P90, Asistencias P90, xA P90, Grandes Chances P90, Duelos Ganados P90, Penales Ganados P90 |
| **Mediocampista** | Pases Precisos P90, Pases Largos P90, Chances Creadas P90, Asistencias P90, xA P90, Recuperaciones Campo Rival P90, Intercepciones P90, Duelos Ganados P90 |
| **Defensa** | Intercepciones P90, Despejes P90, Recuperaciones Campo Rival P90, Entradas P90, Tiros Bloqueados P90, Pases Precisos P90, Pases Largos P90, Faltas Cometidas P90* |
| **Portero** | Paradas P90, % Paradas P90, Goles Evitados P90, Goles Recibidos P90*, Porterías en Cero P90, Pases Precisos P90, Pases Largos P90, Despejes P90 |

> `*` Métrica invertida: menor valor → mejor percentil.

**Metodología de normalización:**
Cada valor se transforma a percentil comparado únicamente contra jugadores de la misma posición con ≥ 300 minutos jugados en el torneo. El percentil 100 es el mejor de su grupo posicional.

---

### `07_ranking_posicion.py` — Top 10 por posición

Ranking visual de los mejores jugadores por posición, con Índice Compuesto calculado sobre percentiles.

```bash
# Todas las posiciones (genera 4 imágenes)
.venv/bin/python scripts/07_ranking_posicion.py

# Una posición específica
.venv/bin/python scripts/07_ranking_posicion.py --posicion Delantero
# Opciones: Delantero | Mediocampista | Defensa | Portero
# → output/charts/ranking_delantero.png
```

**Diseño:**
- Requisito mínimo: **≥ 300 minutos jugados**
- Barras de cada métrica con gradiente de color rojo→verde según percentil
- Círculo de Score Total: verde (top 3) / naranja-amarillo (top 4-7) / naranja (resto)
- Foto circular del jugador + escudo del equipo por fila
- Fondo con gradiente oscuro y separador rojo MAU-STATISTICS

**Métricas del Índice Compuesto por posición:**

| Posición | Métricas (5 barras) |
|---|---|
| **Delantero** | Goles P90, xG P90, Tiros P90, Asistencias P90, xA P90 |
| **Mediocampista** | Pases Precisos P90, Pases Largos P90, Asistencias P90, Recuperaciones Campo Rival P90, Duelos Ganados P90 |
| **Defensa** | Intercepciones P90, Duelos Ganados P90, Recuperaciones Campo Rival P90, Pases Precisos P90, Despejes P90 |
| **Portero** | Paradas P90, Porterías en Cero P90, Goles Concedidos P90 (invertido) |

El **Score Total** es el promedio simple de los percentiles de cada métrica, escalado a 0-100.

---

### `08_comparativo_1v1.py` — Comparativo cara a cara

Infografía de dos jugadores con barras espejo coloreadas por equipo, rating FotMob y contador de victorias por métrica.

```bash
# Por defecto: Paulinho vs Ángel Sepúlveda
.venv/bin/python scripts/08_comparativo_1v1.py

# Con IDs de FotMob específicos
.venv/bin/python scripts/08_comparativo_1v1.py 361377 215428
# → output/charts/comparativo_paulinho_vs_angel_sepulveda.png
```

> Los IDs de jugador se obtienen de la URL de FotMob: `fotmob.com/players/361377/paulinho`

**Diseño:**
- Header bicolor: fondo del color del equipo con gradiente diagonal, foto circular del jugador y escudo inline
- Círculo de rating FotMob coloreado por equipo (valor numérico centrado con `anchor='mm'`)
- 10 métricas P90 enfrentadas en barras espejo desde el centro
- Ganador de cada barra en el color del equipo; perdedor en `#252525` con borde `#444444`
- Banner inferior con conteo de victorias por métrica y barra proporcional

**Métricas comparadas (10):**

| # | Métrica | Descripción |
|---|---|---|
| 1 | Goles P90 | Goles marcados por 90 minutos |
| 2 | xG P90 | Expected Goals por 90 minutos |
| 3 | Tiros P90 | Total de tiros por 90 minutos |
| 4 | Tiros a Puerta P90 | Tiros entre los tres palos por 90 minutos |
| 5 | Asistencias P90 | Asistencias de gol por 90 minutos |
| 6 | xA P90 | Expected Assists por 90 minutos |
| 7 | Chances Creadas P90 | Pases que generan oportunidad de gol por 90 min |
| 8 | Pases Precisos P90 | Pases completados por 90 minutos |
| 9 | Duelos Ganados P90 | Duelos en tierra ganados por 90 minutos |
| 10 | Regates Exitosos P90 | Regates completados por 90 minutos |

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

**Metodología — cómo funciona:**

El modelo usa la distribución de **Poisson** para estimar los goles esperados λ de cada equipo:

```
λ_local     = ataque_local(A)_casa    × defensa_visitante(B)_fuera × μ_home
λ_visitante = ataque_visitante(B)_fuera × defensa_local(A)_casa    × μ_away
```

Los factores se calculan con los **últimos 4 torneos ponderados**:

| Torneo | Peso |
|---|---|
| Clausura 2026 | **4** |
| Apertura 2025 | **3** |
| Clausura 2025 | **2** |
| Apertura 2024 | **1** |

**Parámetros del modelo:**

- `μ_home` / `μ_away`: promedio ponderado de goles como local y visitante en la liga
- `ataque[equipo].home`: ratio goles marcados como local ÷ `μ_home` (normalizado por partidos jugados)
- `ataque[equipo].away`: ratio goles marcados como visitante ÷ `μ_away`
- `defensa[equipo].home`: ratio goles concedidos como local ÷ `μ_away`
- `defensa[equipo].away`: ratio goles concedidos como visitante ÷ `μ_home`

Si un equipo no tiene historial en algún rol, se usa factor 1.0 (media de la liga).

La **matriz de probabilidades 6×6** (marcadores 0-0 a 5-5) se calcula como:
```
P(local=i, visitante=j) = Poisson(i; λ_local) × Poisson(j; λ_visitante)
```
y se normaliza al 100% dividiendo entre la suma total de la matriz truncada.

---

### `12_resumen_jornada.py` — Resumen visual de jornada

Genera una infografía vertical con todos los partidos de una jornada: barras de probabilidad coloreadas por equipo, marcador más probable y escudos.

```bash
# Editar la lista PARTIDOS dentro del script con los duelos de la jornada
.venv/bin/python scripts/12_resumen_jornada.py
# → output/charts/jornada13/resumen_jornada13.png
```

Cada partido muestra:
- Escudos de ambos equipos
- Tres barras de probabilidad coloreadas: victoria local, empate, victoria visitante
- Porcentaje numérico de cada resultado
- Marcador más probable con su probabilidad

---

### `12_modelo_elo.py` — Sistema de rating ELO histórico

Procesa los **32 torneos históricos** (~5,250 partidos, 2010/11 → 2025/26) y genera dos visualizaciones: evolución temporal y ranking actual.

```bash
.venv/bin/python scripts/12_modelo_elo.py
# → data/processed/elo_historico.csv
# → output/charts/elo_evolucion.png
# → output/charts/elo_ranking.png
```

**Metodología ELO:**

El sistema ELO asigna a cada equipo un rating numérico que sube al ganar y baja al perder. Los parámetros usados:

| Parámetro | Valor | Descripción |
|---|---|---|
| `ELO_BASE` | 1500 | Rating inicial de todo equipo |
| `K` | 32 | Factor de actualización máximo por partido |
| `HOME_ADV` | 100 | Puntos extra para el local al calcular probabilidad esperada |
| `SCALE` | 400 | Escala logística (equivalente Elo estándar) |
| `REGRESSION` | 30% | Regresión a la media al inicio de cada torneo nuevo |

**Fórmula de probabilidad esperada:**
```
E_local = 1 / (1 + 10^((ELO_visitante - ELO_local - HOME_ADV) / 400))
E_visitante = 1 - E_local
```

**Actualización por partido:**
```
ΔElo = K × GoalMarginMultiplier × (resultado_real - resultado_esperado)
```

Donde `resultado_real` es 1 (victoria), 0.5 (empate) o 0 (derrota).

**Multiplicador por margen de goles** (basado en metodología 538/Club Elo):
```
GoalMarginMultiplier = 1.0 + ln(|goles_local - goles_visitante| + 1) × 0.5
```
Esto penaliza las goleadas pero con rendimientos decrecientes para evitar que un 5-0 domine el cálculo.

**Regresión entre torneos:**
Al inicio de cada torneo, todo equipo regresa un 30% hacia la media de 1500:
```
ELO_nuevo = ELO_actual + 0.30 × (1500 - ELO_actual)
```
Esto evita que un equipo acumule ventaja indefinida y refleja la naturaleza de cada torneo como competencia semi-independiente.

**Visualización de evolución:**
- Datos resampleados semanalmente (`.resample('W').last().ffill()`)
- Suavizado con rolling window de 8 semanas centrado (`center=True`)
- Solo se grafican los 18 equipos del Clausura 2026
- Línea de referencia blanca punteada en ELO=1500 (media)
- Anotación de pausa COVID-19 (marzo 2020)
- Etiquetas al final de cada línea con separación mínima de 22 puntos ELO

**Visualización de ranking:**
- Barras con gradiente horizontal color de equipo → oscuro
- Delta respecto a base 1500: verde si positivo, rojo si negativo
- Fondo especial `#1a1f2e` para los tres primeros lugares
- Escudo de equipo al inicio de cada fila

---

## 📐 `config_visual.py` — Paleta de identidad MAU-STATISTICS

Módulo centralizado importado por todos los scripts de visualización. Define la paleta de colores, tipografía y utilidades de color.

```python
from config_visual import PALETTE, bebas, hex_rgba, hex_rgb
```

**Paleta base:**

| Token | Hex | Uso |
|---|---|---|
| `bg_main` | `#0d1117` | Fondo principal |
| `bg_secondary` | `#161b22` | Filas alternas / header |
| `bg_card` | `#0f151e` | Tarjetas / paneles |
| `text_primary` | `#ffffff` | Texto principal |
| `text_secondary` | `#8b949e` | Texto secundario / "Fuente: FotMob" |
| `accent` | `#D5001C` | Rojo MAU-STATISTICS — líneas, títulos |
| `positive` | `#2ea043` | Delta positivo / verde |
| `negative` | `#f85149` | Delta negativo / rojo claro |
| `border` | `#30363d` | Bordes |
| `divider` | `#21262d` | Divisores internos |
| `grid` | `#1e2530` | Líneas de cuadrícula |
| `bar_track` | `#181c24` | Track (fondo de barra vacía) |
| `bar_loser` | `#252525` | Barra del jugador perdedor en 1v1 |
| `bar_loser_border` | `#444444` | Borde de barra perdedora |

**Funciones de utilidad:**

```python
bebas(size: float) -> dict       # kwargs de Bebas Neue para matplotlib
hex_rgb(hex: str) -> tuple       # '#RRGGBB' → (R, G, B) int 0-255
hex_rgba(hex: str, a=1.0)        # '#RRGGBB' → (r, g, b, a) float 0-1
darken(hex: str, factor=0.55)    # Oscurece un color RGB
make_h_gradient(hex, w=256)      # Array RGBA (1, w, 4) con gradiente horizontal
```

**Convención de footer en todos los gráficos:**
- Esquina inferior derecha: `MAU-STATISTICS` en Bebas Neue 20pt, rojo `#D5001C`
- Esquina inferior izquierda: `Fuente: FotMob` en 10pt, gris `#8b949e`

---

## 📊 Métricas P90 disponibles en el DataFrame

El archivo `data/processed/jugadores_clausura2026.csv` contiene las siguientes columnas per-90 (por 90 minutos jugados):

**Tiro / Gol:**
`goles_p90`, `xG_p90`, `xG_np_p90`, `xGOT_p90`, `tiros_p90`, `tiros_a_puerta_p90`, `cabezazos_p90`

**Pase / Creación:**
`asistencias_p90`, `xA_p90`, `chances_creadas_p90`, `grandes_chances_p90`, `pases_precisos_p90`, `precision_pases_p90`, `pases_largos_p90`, `precision_pases_largos_p90`, `centros_precisos_p90`, `precision_centros_p90`

**Posesión / Duelos:**
`duelos_ganados_p90`, `duelos_ganados_pct_p90`, `regates_p90`, `duelos_tierra_ganados_p90`, `duelos_aereos_ganados_p90`, `duelos_aereos_ganados_pct_p90`, `toques_p90`, `toques_area_rival_p90`, `faltas_recibidas_p90`, `perdidas_balon_p90`

**Defensa:**
`intercepciones_p90`, `despejes_p90`, `tiros_bloqueados_p90`, `recuperaciones_p90`, `faltas_cometidas_p90`, `regateado_p90`, `entradas_p90`, `recuperaciones_campo_rival_p90`

**Disciplina:**
`tarjetas_amarillas_p90`, `tarjetas_rojas_p90`

**Portería:**
`paradas_p90`, `porcentaje_paradas_p90`, `goles_recibidos_p90`, `goles_evitados_p90`, `salidas_p90`, `balones_al_aire_p90`, `porterias_cero_p90`, `penales_atajados_p90`, `pct_penales_atajados_p90`, `dist_pases_precisos_p90`

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

# 3. (Recomendado) Instalar fuente Bebas Neue
mkdir -p ~/.fonts
curl -sL "https://github.com/dharmatype/Bebas-Neue/raw/master/fonts/ttf/BebasNeue-Regular.ttf" \
     -o ~/.fonts/BebasNeue.ttf
fc-cache -fv
```

---

## 📡 Fuentes de datos

Todos los datos provienen de **FotMob** mediante scraping del JSON embebido `__NEXT_DATA__` o de la API no oficial `data.fotmob.com`. No se usa ninguna API oficial ni de terceros pagos.

| Endpoint | Contenido |
|---|---|
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
|---|---|
| `pandas` | Manipulación, limpieza y feature engineering |
| `matplotlib` | Todas las visualizaciones estáticas |
| `Pillow` (PIL) | Imágenes circulares, escudos, composición RGBA, gradientes |
| `requests` / `urllib` | Scraping HTTP |
| `scipy` | Distribución de Poisson para el modelo de predicción |
| `numpy` | Álgebra lineal, matrices de probabilidad, gradientes |
| `mplsoccer` | Radar chart base (script 03) |

---

## 🗺️ Roadmap

- [x] Scraping de equipos y jugadores (Clausura 2026)
- [x] DataFrame maestro con stats P90 por posición
- [x] Pizza chart P90 individual (Statiskicks style)
- [x] Ranking visual Top 10 por posición
- [x] Comparativo 1v1 con barras espejo
- [x] Histórico completo 15 años (32 torneos, ~5,250 partidos)
- [x] Modelo de Poisson para predicción de marcadores
- [x] Resumen visual de jornada completa
- [x] Sistema de rating ELO histórico (evolución + ranking)
- [x] Paleta centralizada MAU-STATISTICS (`config_visual.py`)
- [ ] Dashboard interactivo (Streamlit / Dash)
- [ ] Predicciones automáticas para la jornada vigente
- [ ] Análisis de rachas y forma reciente por equipo
- [ ] Comparativo de temporadas (rendimiento histórico por equipo)
- [ ] Integración ELO como feature en el modelo de Poisson

---

<p align="center">
  <strong>MAU-STATISTICS</strong> · Datos: FotMob · Liga MX 2010 – 2026
</p>
