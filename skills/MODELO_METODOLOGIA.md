# MODELO_METODOLOGIA.md — Metodología de modelos estadísticos

---

## 1. Sistema ELO — Liga MX

**Script**: `12_modelo_elo.py`
**Data**: `data/processed/elo_historico.csv`

### Parámetros
```python
ELO_BASE  = 1500   # Rating inicial de todo equipo nuevo
K         = 32     # Factor de actualización máximo
HOME_ADV  = 100    # Puntos de ventaja local (en cálculo de E, no en ELO)
SCALE     = 400    # Escala logística estándar
REGRESSION = 0.30  # Regresión a la media entre torneos
```

### Fórmula completa
```
E_local = 1 / (1 + 10^((ELO_visita - ELO_local - HOME_ADV) / 400))

S_local = 1.0  si local gana
          0.5  si empate
          0.0  si visita gana

GoalMult = 1.0 + ln(|GL - GV| + 1) × 0.5

ΔELO_local  = K × GoalMult × (S_local - E_local)
ΔELO_visita = K × GoalMult × ((1 - S_local) - (1 - E_local))
```

### Regresión entre torneos
Al inicio de cada nuevo torneo:
```
ELO_nuevo = ELO_actual + 0.30 × (1500 - ELO_actual)
```
Equipos sobre 1500 bajan, equipos bajo 1500 suben.

### ELO actual Clausura 2026 (al 2026-03-23)
| Equipo | ELO |
|---|---|
| Toluca | 1709 |
| Cruz Azul | 1692 |
| Chivas | 1660 |
| Pumas | 1590 |
| Tigres | 1553 |
| Pachuca | 1542 |
| América | 1541 |
| Monterrey | 1519 |
| FC Juárez | 1487 |
| Atlas | 1461 |
| Tijuana | 1460 |
| Necaxa | 1439 |
| Querétaro | 1411 |
| Mazatlán | 1399 |
| León | 1390 |
| Puebla | 1390 |
| San Luis | 1383 |
| Santos | 1377 |

---

## 2. Sistema ELO — Selecciones Nacionales

**Script**: `18_prediccion_selecciones.py`, `update_elo_selecciones.py`
**Data**: `data/processed/elos_selecciones_20260401.json`

### Diferencias con Liga MX ELO
- Sin regresión entre torneos
- K dinámico por tipo de torneo (metodología eloratings.net)
- HOME_ADV=100 solo si `neutral=False`

### K dinámico por torneo
```python
def k_base(tournament: str) -> int:
    # Copa del Mundo (rondas finales, no clasificatorias)
    if 'world cup' and not qualifier: return 60
    # Torneos continentales principales (Copa América, Gold Cup, Euro, etc.)
    if torneo_continental_main and not qualifier: return 35
    # Clasificatorias y UEFA Nations League
    if qualifier or 'uefa nations league': return 25
    # Amistosos y otros
    return 20
```

### Multiplicador de margen de goles
```python
def goal_mult(gf, ga):
    diff = abs(gf - ga)
    if diff <= 1:   return 1.0
    elif diff == 2: return 1.5
    else:           return 1.75 + (diff - 3) * 0.04
```

K efectivo = `k_base(torneo) × goal_mult(gf, ga)`

### Top 10 ELO Selecciones (al 2026-04-01)
| # | País | ELO |
|---|---|---|
| 1 | Argentina | 2125.3 |
| 2 | España | 2103.4 |
| 3 | Francia | 2092.9 |
| 4 | Brasil | 2041.0 |
| 5 | Colombia | 2014.0 |
| 6 | Inglaterra | 2011.8 |
| 7 | Portugal | 1995.2 |
| 8 | Países Bajos | 1989.6 |
| 9 | Croacia | 1946.7 |
| 10 | Ecuador | 1944.2 |

México: **1888.3** | Bélgica: 1906.4

### Normalización de nombres (FotMob → ELO dict)
```python
NAME_MAP = {
    "Czechia":             "Czech Republic",
    "Turkiye":             "Turkey",
    "USA":                 "United States",
    "Ireland":             "Republic of Ireland",
    "Curacao":             "Curaçao",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "Chinese Taipei":      "Taiwan",
}
```

---

## 3. Modelo de Poisson — Liga MX

**Script**: `11_modelo_prediccion.py`

### Parámetros del modelo
```python
TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4,
    '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2,
    '2024/2025 - Apertura': 1,
}
HOME_ADV_LAMBDA = 1.15   # multiplicador λ para local
mu = 1.421               # promedio ponderado goles/partido
# 608 partidos cargados
```

### Fórmula de lambdas
```
λ_local     = att[local]  × defe[visita] × μ × 1.15
λ_visitante = att[visita] × defe[local]  × μ

att[equipo] = (goles marcados ponderados / partidos ponderados) / μ
defe[equipo] = (goles recibidos ponderados / partidos ponderados) / μ
```

### Matriz de probabilidades
```
P(local=i, visita=j) = Poisson(i; λ_local) × Poisson(j; λ_visitante)
```
Matriz 7×7 (0 a 6 goles).

### Probabilidades de resultado
```
P(local gana)  = Σ P(i,j) para i > j  (triángulo inferior)
P(empate)      = Σ P(i,i) (diagonal)
P(visita gana) = Σ P(i,j) para j > i  (triángulo superior)
```

**Bug crítico corregido**: los JSONs históricos usan guión en el año (`2025-2026`) pero el diccionario de pesos usaba slash (`2025/2026`). Fix en `load_model()`:
```python
tkey = f"{parts[0].replace('-','/').replace('_','/')} - ..."
```

---

## 4. Modelo ELO+Poisson — Predicciones del día

**Scripts**: `19_predicciones_hoy.py`, `gen_predicciones_ligamx_20260404.py`

### Para selecciones (neutral)
```python
AVG_GOALS = 1.35
elo_mean  = promedio de todos los ELOs del día
λ_equipo  = AVG_GOALS × (elo_equipo / elo_mean)
# Sin ventaja de local (partidos neutrales)
```

### Para Liga MX (con localía)
```python
AVG_GOALS    = 1.35
HOME_ADV_ELO = 100   # se suma al ELO local SOLO para λ, no al ELO almacenado
elo_mean     = promedio ELOs de todos los equipos jugando ese día
λ_local      = AVG_GOALS × (elo_local + 100) / elo_mean
λ_visitante  = AVG_GOALS × elo_visitante / elo_mean
```

### Normalización obligatoria
```python
_total  = p_home + p_draw + p_away
p_home  = p_home / _total
p_draw  = p_draw / _total
p_away  = 1.0 - p_home - p_draw   # el último se calcula por diferencia
```

---

## 5. Intervalos de Confianza — Bootstrap Monte Carlo

**Implementado en**: `gen_predicciones_ligamx_20260404.py`

### Metodología
```python
def bootstrap_ci(lam_h, lam_a, n_sim=1000, ci=0.95):
    # 1. Simular 1000 partidos Poisson
    gh = np.random.poisson(lam_h, n_sim)
    ga = np.random.poisson(lam_a, n_sim)
    outcomes = where(gh > ga, 0, where(gh < ga, 2, 1))  # 0=local, 1=emp, 2=vis

    # 2. Bootstrap: 1000 remuestreos de los 1000 resultados
    for _ in range(1000):
        sample = choice(outcomes, size=n_sim, replace=True)
        registrar(proporción home, draw, away)

    # 3. Percentiles 2.5 y 97.5
    CI_low  = percentile(bootstrap_props, 2.5)
    CI_high = percentile(bootstrap_props, 97.5)
```

Formato de display: `IC 95%: [X.X% — X.X%]` en 9pt, alpha=0.6

---

## 6. Dixon-Coles — ✅ IMPLEMENTADO (2026-04-04)

El modelo de Poisson independiente subestima la frecuencia de resultados bajos (0-0, 1-0, 0-1, 1-1). Dixon-Coles (1997) corrige esto con un factor ρ.

### Corrección Dixon-Coles
```
τ(i, j, λ, μ, ρ) =
  1 - λ·μ·ρ   si i=0, j=0
  1 + λ·ρ     si i=0, j=1
  1 + μ·ρ     si i=1, j=0
  1 - ρ       si i=1, j=1
  1           en cualquier otro caso

P_DC(i, j) = τ(i, j, λ, μ, ρ) × Poisson(i; λ) × Poisson(j; μ)
```

### Implementación actual
- `rho = -0.13` (valor estándar académico para fútbol)
- Función `dixon_coles_correction(home_goals, away_goals, lambda_home, lambda_away, rho)`
- Integrada en `gen_predicciones_ligamx_20260404.py`, `11_modelo_prediccion.py`, `15_prediccion_elo_poisson.py`
- La matriz se renormaliza tras aplicar DC: `probs /= probs.sum()`

### Efecto medido — Cruz Azul vs Pachuca (λh=1.59, λa=1.37)
| Resultado | Poisson Puro | Dixon-Coles | Δ |
|---|---|---|---|
| Local gana | 42.51% | 41.03% | −1.48% |
| Empate | 24.56% | 27.52% | **+2.97%** |
| Visita gana | 32.93% | 31.45% | −1.48% |
| 0-0 | 5.25% | 6.74% | +1.48% |
| 1-1 | 11.40% | 12.89% | +1.48% |
| 1-0 | 8.35% | 6.86% | −1.48% |
| 0-1 | 7.18% | 5.70% | −1.48% |

### Pendiente
- Estimar ρ óptimo por MLE sobre el histórico de 38 torneos (esperado: −0.13 a −0.08)
- Comparar Brier score con ρ=−0.13 vs ρ_MLE cuando haya suficientes predicciones registradas

---

## 7. Tracker de predicciones

**Script**: `04_predicciones_tracker.py`
**Data**: `data/processed/predicciones_log.csv`

### Columnas del CSV
```
fecha_prediccion, partido, equipo_local, equipo_visitante,
elo_local, elo_visitante, prob_local, prob_empate, prob_visitante,
ganador_predicho, marcador_mas_probable, lambda_local, lambda_visitante,
paleta_usada, fecha_partido, resultado_real, goles_local_real,
goles_visitante_real, acierto_ganador, error_marcador
```

### Registro de resultado real
```bash
python3 scripts/04_predicciones_tracker.py resultado "Monterrey vs San Luis" 2026-04-04 X X
python3 scripts/04_predicciones_tracker.py reporte
```

### Métricas de desempeño
- `acierto_ganador`: 1 si el ganador predicho coincide con el real
- `error_marcador`: distancia Manhattan `|goles_pred_local - real_local| + |...|`

---

## 8. Arquitectura objetivo del modelo

### Capas del modelo final

```
CAPA 1 — ELO dinámico (✅ implementado)
  - K dinámico por tipo de partido (20/25/35/60)
  - Factor de localía +100 ELO
  - Margen de goles como multiplicador
  - Metodología eloratings.net

CAPA 2 — Poisson + Dixon-Coles (🔄 en implementación)
  - rho = -0.13 como valor inicial
  - Corrección en marcadores bajos (0-0, 0-1, 1-0, 1-1)
  - Reemplaza Poisson puro en todos los scripts

CAPA 3 — Variables contextuales (⏳ pendiente)
  Variables de alto impacto ya identificadas:
  - Forma reciente ponderada (últimos 5 partidos, peso exponencial)
  - xG por equipo por partido (fuente: FBref scraping)
  - Días de descanso entre partidos
  - Ausencias clave (variable binaria: titular top disponible)
  - Head-to-head reciente (últimos 3 encuentros directos)
  - Valor de mercado convocatoria (fuente: Transfermarkt)

CAPA 4 — XGBoost calibrado (⏳ pendiente)
  - Entrena sobre el error residual del modelo base
  - Requiere mínimo 3 temporadas de datos con todas las variables
  - Valida con Brier score y log-loss

CAPA 5 — Value betting + Kelly fraccionado (⏳ pendiente)
  - Compara p_modelo vs p_implícita de casas de apuestas
  - Umbral de valor: diferencia > 5%
  - Kelly fraccionado al 25% para reducir varianza
  - Límite por apuesta: máximo 3% del bankroll
  - Máximo 3 apuestas por jornada
  - Requiere 200 predicciones históricas para validar antes de dinero real
```

### Métricas de evaluación del modelo
- % acierto en ganador (baseline: ~50% en Liga MX)
- Brier score (menor = mejor calibración)
- Log-loss
- ROI hipotético con Kelly 25%
- Calibración: cuando modelo dice 60%, ¿ocurre 60%?
