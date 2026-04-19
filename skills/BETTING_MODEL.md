# BETTING_MODEL.md — Modelo de Value Betting MAU-STATISTICS

> Creado: 2026-04-18. Objetivo: monetización mediante apuestas con ventaja estadística comprobada.

---

## Filosofía central

**NO apostamos por apostar. Apostamos cuando el modelo dice que la cuota tiene valor.**

```
Valor esperado (EV) = (prob_modelo × cuota_decimal) - 1

Apostar SOLO si EV > 0.05   ← mínimo 5% de ventaja sobre la casa
```

Si el modelo dice 60% y la cuota implica 50% → EV = 0.60×2.00−1 = 0.20 → apostar.
Si el modelo dice 55% y la cuota implica 53% → EV = 0.02 → NO apostar (ruido).

---

## Mercados objetivo (por orden de prioridad)

| Mercado | Eficiencia de casas | Por qué atacarlo |
|---|---|---|
| **Corners totales / línea** | Media | Pocas casas tienen modelo robusto de corners |
| **BTTS** (ambos anotan) | Media | Depende de xG, no de resultado — menos volumen de apuestas |
| **Tarjetas totales** | Media | Muy poco histórico en modelos de casas para Liga MX |
| **Handicap asiático** | Media-Alta | Liga MX menos cubierta que EPL, más ineficiencias |
| **Gol de jugador** | Alta | Requiere xG por jugador + minutaje esperado |
| **1X2 resultado** | Muy Alta | Pinnacle tiene 3,000 variables — NO es nuestro mercado principal |

---

## Arquitectura del modelo (dos capas)

### Capa 1: Modelo estadístico (Poisson MLE)
Usado en `modelo_corners.py`, `modelo_tarjetas.py`, `modelo_btts.py`

### Capa 2: Modelo ML (LightGBM calibrado) — PRINCIPAL
Usado en `modelo_ml.py`. Reemplaza Poisson en todos los mercados excepto BTTS.

```
Features (23 variables):
  - ELO diff, ELO local, ELO visita, win_expected
  - Rolling 5 partidos: corners_for, corners_against, cards_avg,
    goals_scored, goals_allowed, xg_scored, win_rate (local + visita)
  - sum_corners_rate, sum_cards_rate
  - altitude_local, altitude_visita, altitude_diff
  - phase_pressure (0=J1, 1=Liguilla)
  - h2h_n (número de antecedentes, no el promedio — evita leakage)

Calibración: Isotónica temporal (split 80/20 cronológico, no random k-fold)
h2h_corners: ajuste Bayesiano POSTERIOR (30% peso si h2h_n >= 3)
Monotonicity: P(Over 8.5) >= P(Over 9.5) >= P(Over 10.5)

Métricas actuales (686 partidos, 137 holdout):
  corners_over_8.5:  Brier=0.215  skill=+12.4%  ✅ USAR
  cards_over_4.5:    Brier=0.206  skill=+13.7%  ✅ USAR
  cards_over_3.5:    Brier=0.138  skill=+12.7%  ✅ USAR
  corners_over_9.5:  Brier=0.249  skill=-1.5%   ⚠️ marginal
  btts:              Brier=0.313  skill=-29.6%   ❌ NO usar ML (usar Poisson)
  resultado_1X2:     LogLoss=1.56 vs naive 1.10  ❌ no hay edge aún

SHAP top features (corners_over_9.5):
  1. v_corners_against (defensiva visitante vulnerable)
  2. altitude_diff (altitud local da ventaja de corners)
  3. sum_cards_rate (partidos tensos → menos corners)

## Corners (modelo_corners.py — Poisson base)
```
Modelo: Poisson MLE con time decay  [Dixon-Coles 1997 + Rue-Salvesen 2000]

  λ_local  = exp(μ + att[local]  + def[visita] + home_adv[local])
  λ_visita = exp(μ + att[visita] + def[local])

  Constraint sum-to-zero: Σ att_i = 0, Σ def_i = 0  (identificabilidad)
  Pesos por recencia: w_t = max(0.15, exp(-0.003 × días))  [Rue-Salvesen]
  Fitting: L-BFGS-B sobre NLL ponderada

  Capa feeling (liga_mx_knowledge.py):
  - home_adv calibrado por estadio (Azteca: ×1.10, altura Toluca: ×1.05)
  - rivalry_bonus aditivo (Clásico Regio: +0.8 total corners)
  - altitude_penalty visitante (Toluca 2680m: ×0.88 λ_visita)
  - phase_mult (liguilla: ×1.05)

  Validación holdout 20 partidos:
  Brier Over 8.5 = 0.127  (baseline naïve = 0.25) → +49% mejor ✅
  MAE total corners = 2.31

Over/Under línea X: convolución de dos Poisson independientes (max_k=50)
```

### BTTS (Both Teams To Score)
```
Modelo: Poisson independiente por equipo
  P(local anota)   = 1 - Poisson(0; λ_local)   = 1 - e^(-λ_local)
  P(visita anota)  = 1 - Poisson(0; λ_visita)  = 1 - e^(-λ_visita)
  P(BTTS = Sí)     = P(local anota) × P(visita anota)

λ viene del modelo ELO+Poisson actual — ya tenemos esto.
```

### Tarjetas
```
Modelo: Poisson sobre card_rate ponderado + factores contextuales

  λ_tarjetas = card_rate[local] + card_rate[visita]
             + rivalry_bonus     (Clásico Nacional: +1.2)
             + referee_factor    (árbitros severos: ×1.25)
             × phase_mult        (liguilla semis: ×1.30)

  card_rate = media histórica de tarjetas ponderadas (amarilla=1, roja=2) por equipo
  Fitting: media simple (MLE colapsa a media con Poisson sin covariables)

  Feeling:
  - 7 árbitros Liga MX calibrados (0.90 a 1.25)
  - 4 rivalidades con bonus de cards (Clásico Nacional: +1.2)
  - Fase del torneo: liguilla cuartos ×1.20, semis ×1.30, final ×1.35

Over/Under X tarjetas: CDF Poisson(X, λ_tarjetas)
```

### Gol de jugador específico
```
Modelo: Poisson por jugador
  λ_gol_jugador = (xG_p90[jugador] × min_esperados/90)
                × factor_rival_def
                × factor_local

P(jugador anota) = 1 - e^(-λ_gol_jugador)
```

---

## Validación antes de apostar dinero real

### Métrica clave: Closing Line Value (CLV)
> Si consistentemente nuestras probabilidades son mejores que la cuota de cierre de Pinnacle, el modelo tiene edge real.

```
CLV = (nuestra_cuota_implícita / cuota_cierre_pinnacle) - 1

Si CLV promedio > 0 en 200+ predicciones → modelo válido para apuestas reales
Si CLV ≈ 0 → modelo sin ventaja, no apostar
```

### Checklist mínimo antes de dinero real
- [ ] 200 predicciones en tracker con resultado real
- [ ] Brier score < 0.22 (baseline naive ≈ 0.25)
- [ ] CLV positivo en backtesting de cuotas históricas
- [ ] ROI hipotético > 3% con Kelly 25% en últimas 100 predicciones
- [ ] Al menos 50 predicciones en el mercado específico (no mezclar corners con 1X2)

### Estado actual del tracker (2026-04-19)
- 23 predicciones evaluadas (solo 1X2 — corners/tarjetas no logueadas aún)
- Acierto: 47.8% → baseline histórico Liga MX ~47% (la predicción siempre es el favorito)
- Brier 1X2: 0.613 | Skill: −11.2% (muestra n=23, insuficiente — necesita n≥30)
- **IMPORTANTE**: n=23 no es estadísticamente concluyente. Esperar n≥50 para conclusiones.
- Calibración base del modelo (para equipos promedio ELO=1500): local=49.1%, empate=24.3%, visita=26.6% — **bien calibrado vs histórico Liga MX (46.7%/25.2%/28.1%)**

### Análisis de datos disponibles (2026-04-19)
```
match_events.csv: 802 partidos (694 con datos completos, 108 sin eventos)
  Datos faltantes: Apertura 2024 (29), Clausura 2025 (29), Clausura 2024 (25), Apertura 2025 (25)
  → Scrape de eventos históricos prioritario para completar dataset

xG de FotMob: r=0.035 con goles reales → datos INUTILIZABLES para BTTS
  → Usar FBref/StatsBomb xG para mercados de goles (r esperado ≈ 0.35+)
  → Mientras tanto: shots como proxy (r=0.621 con corners, r≈0.30 con goles)

Mejor predictor de corners: shots_local+visitante (r=0.303/0.268)
```

---

## Gaps de datos para mejorar modelos

| Dato faltante | Impacto | Fuente |
|---|---|---|
| **xG de calidad** (StatsBomb/FBref) | Alto para BTTS y 1X2 | FBref.com — Liga MX coverage desde 2022 |
| **Asignación de árbitros por partido** | Alto para tarjetas | FotMob API / Soccerway |
| **Cuotas históricas** (Pinnacle) | Alto para CLV backtesting | football-data.co.uk (parcial) |
| **Eventos faltantes 2024-2025** | Medio para ML training | Re-scrape FotMob (108 partidos) |
| **Lesiones / disponibilidad** | Medio para 1X2 | Transfermarkt / FotMob API |
| **Datos CCL históricos 2010-2025** | Bajo para CCL | RSSSF / Wikipedia scrape |

---

## Sistema de retroalimentación — discovery_bot.py

`bots/discovery_bot.py` corre diariamente y analiza (7 secciones):

1. **Calibración corners/tarjetas** — ¿modelo naïve sobreestima/subestima vs datos reales?
2. **Calibración predicciones 1X2** — Brier, ECE, bias por resultado (local/empate/visita)
3. **Outliers por equipo** — equipos con comportamiento atípico (z-score > 1.5σ)
4. **Deriva temporal** — ¿hay drift en corners/tarjetas vs historial?
5. **Correlaciones** — ¿qué mercados son independientes entre sí?
6. **Patrones alta presión** — partidos candidatos para portafolio múltiple
7. **Recomendaciones** — accionables con prioridad alta/media/baja

Output: `output/reports/discovery/discovery_YYYY-MM-DD.json` + `.html`
También `output/reports/discovery_latest.html` para el email diario.

**Hallazgos actuales (2026-04-19):**
- Drift corners +1.3 (reciente=10.6 vs hist=9.3) → retrain urgente ✅ detectado
- Querétaro FC (+5.5c/j), Mazatlán (+5.4c/j), Puebla (+5.3c/j): top generadores
- Puebla y Querétaro también top en corners concedidos → mercado double-sided
- Corners y tarjetas son independientes (r=-0.12) → portafolio mixto válido
- Correlación shots vs corners: r=0.621 → mejor feature para corners ML
- xG FotMob inutilizable para BTTS (r=0.035 con goles reales)

---

## Bankroll management — Kelly fraccionado

```python
def kelly_bet(prob_modelo: float, cuota_decimal: float, bankroll: float,
              fraccion: float = 0.25) -> float:
    """
    Kelly fraccionado. fraccion=0.25 reduce varianza al 25% del Kelly óptimo.
    """
    b = cuota_decimal - 1      # ganancia neta por unidad apostada
    p = prob_modelo
    q = 1 - p
    k_full = (b * p - q) / b   # Kelly completo
    if k_full <= 0:
        return 0.0              # sin valor esperado positivo
    bet = bankroll * k_full * fraccion
    return min(bet, bankroll * 0.03)  # máximo 3% del bankroll por apuesta
```

### Reglas de gestión
- **Bankroll inicial recomendado**: separado del dinero personal, solo lo que se puede perder
- **Máximo por apuesta**: 3% del bankroll
- **Máximo por jornada**: 3 apuestas activas simultáneas
- **Stop-loss**: si el bankroll baja 30%, parar y re-evaluar modelo
- **Registro obligatorio**: cada apuesta en tracker con fecha, mercado, cuota, resultado

---

## Fuentes de cuotas

| Fuente | Tipo | Uso |
|---|---|---|
| **Pinnacle** | Casa más eficiente del mundo | Benchmark de CLV — si les ganas, tienes edge real |
| **The Odds API** | Agrega 40+ casas | `api.the-odds-api.com` — free tier 500 req/mes |
| **football-data.co.uk** | Cuotas históricas gratis | Backtesting — Liga MX parcial |
| **Betfair Exchange** | Mercado de pares | Mejores cuotas si tienes volumen |
| **Caliente / Bet365** | Casas con Liga MX | Comparar vs modelo para encontrar discrepancias |

### Script pendiente: `scrape_odds.py`
```
Input:  partido (local, visita, fecha), mercado
Output: dict con cuotas de todas las casas disponibles
        + cuota implícita de Pinnacle (sin margen)
        + EV calculado vs modelo propio
```

---

## Datos necesarios por mercado (estado actual)

| Dato | Mercado | Fuente | Estado |
|---|---|---|---|
| Goles por partido | 1X2, BTTS | FotMob historico ✅ | Listo |
| xG por partido | 1X2, BTTS, jugador | FBref/StatsBomb | ❌ Pendiente |
| Corners por partido | Corners | FotMob matchfacts | ✅ `match_events.csv` 126 partidos |
| Tarjetas por partido | Tarjetas | FotMob matchfacts | ✅ `match_events.csv` 126 partidos |
| xG por jugador | Gol jugador | FBref | ❌ Pendiente |
| Minutaje por jugador | Gol jugador | FotMob stats | ⚠️ Parcial |
| Cuotas históricas | Backtesting | football-data.co.uk | ❌ Pendiente |
| Cuotas en vivo | Value betting | The Odds API | ❌ Pendiente |

---

## Roadmap betting — Fases

### Fase B1 — Datos granulares de partido ✅ (casi completo)

- ✅ `scrape_match_events.py` — 126 partidos Clausura 2026 con corners, tarjetas, shots, xG
  - Output: `data/processed/match_events.csv`
  - En proceso: `--all` para extender a 5+ torneos (~980 partidos)
  - En pipeline diario: `--days 4` actualiza automáticamente
- ✅ Parámetros calibrados: μ_corners=9.32, μ_amarillas=4.43, μ_xG_local=1.48
- ⏳ `scrape_xg_fbref.py` — xG por jugador desde FBref (pendiente)

### Fase B2 — Modelos por mercado ✅ (implementados, pendiente re-calibrar con dataset completo)

- ✅ `modelo_corners.py` — Poisson bivariado con att/def por equipo
  - `corners_model.json` con factores att/def de todos los equipos
  - Top atacantes-corner Clausura 2026: Toluca(1.26) Cruz Azul(1.18) Chivas(1.13)
- ✅ `modelo_btts.py` — Dixon-Coles rho=-0.13, P(BTTS) + P(Over 1.5/2.5)
- ✅ `modelo_tarjetas.py` — Poisson sobre card_rate + factor rivalidad
  - `tarjetas_model.json` con card_rate por equipo
- ✅ `calcular_ev.py` — EV unificado de los 3 modelos: `python calcular_ev.py --local X --visita Y --corners-over 1.90 --tarjetas-over 1.85 --btts-si 1.80`
- ⏳ `modelo_jugador_gol.py` — requiere xG por jugador de FBref

### Fase B3 — Odds scraper + EV calculator
- ⏳ `scrape_odds.py` — The Odds API wrapper (free tier: 500 req/mes)
- ⏳ Integrar cuotas automáticas en `calcular_ev.py` (hoy es manual)
- ⏳ Sección "🎯 Value bets detectadas hoy" en email diario

### Fase B4 — Tracker unificado
- Extender `predicciones_log.csv` con: mercado, cuota_apostada, casa, resultado_mercado, P&L
- Dashboard Dash con ROI por mercado, CLV histórico, bankroll curve

### Fase B5 — Producción
- Solo cuando CLV > 0 en 200+ predicciones por mercado
- Apuestas reales con Kelly 25%, máximo 3% bankroll

---

## Lección aprendida de CCL
> Construir la infraestructura de datos ANTES del modelo. Con 35 partidos el ELO CCL no es confiable. Para betting, necesitamos 3+ temporadas de corners/tarjetas antes de calibrar λ.
