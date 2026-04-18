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

## Arquitectura del modelo por mercado

### Corners
```
Modelo: Poisson bivariado sobre corner_rate
  λ_corners_local   = att_corner[local]  × defe_corner[visita] × μ_corners
  λ_corners_visita  = att_corner[visita] × defe_corner[local]  × μ_corners

μ_corners Liga MX = 9.32 por partido (calibrado en 126 partidos Clausura 2026)

Over/Under línea X: P(total > X) = 1 - CDF_Poisson(X, λ_total)
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
Modelo: Poisson simple sobre card_rate histórico
  λ_tarjetas = card_rate[local] + card_rate[visita]
             + factor_rivalidad[local vs visita]
             + factor_árbitro (pendiente)

Over/Under X tarjetas: directo de CDF Poisson
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

### Estado actual del tracker
- 23 predicciones evaluadas, solo 1X2
- Acierto: 47.8% → por debajo del baseline — modelo 1X2 NO está listo para apostar
- Corners/BTTS/tarjetas: 0 predicciones — falta construir

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
