# VISION_GLOBAL.md — Portafolio Global de Value Betting
> Versión 1.0 — 2026-04-18. Actualizar al completar cada fase.

---

## La visión

**Construir el sistema de value betting estadístico más completo para mercados hispanohablantes.**

No apostamos para apostar. Apostamos cuando el modelo tiene ventaja demostrable.
La ventaja viene de tres fuentes:
1. **Modelos más precisos** que los algoritmos internos de las casas en ligas poco cubiertas
2. **Diversificación geográfica** — pérdidas en una liga se compensan con ganancias en otras
3. **Disciplina de bankroll** — Kelly fraccionado + stop-loss previenen la ruina

El objetivo no es "ganar siempre en un partido". Es tener **EV esperado positivo** en un portafolio
de 10-20 apuestas semanales distribuidas en múltiples ligas y mercados.

---

## Las ligas objetivo (por prioridad)

### Tier 1 — Cubrir primero (menos eficientes, mejor edge)

| Liga | Por qué | Mercados prioritarios | Datos |
|------|---------|----------------------|-------|
| **Liga MX** | Mercado local, datos FotMob ✅, casas menos sofisticadas | Corners, tarjetas, BTTS | ✅ Listo |
| **CONCACAF Champions Cup** | Equipos mexicanos + MLS, poca cobertura de casas | Resultado, BTTS | ⚠️ Parcial |
| **Liga Argentex (Argentina)** | Modelo ELO transferible, mercado activo en Latinoamérica | Corners, goles | ⏳ |
| **Brasileirão** | Liga más grande LATAM, muchos partidos, menos analizada | Corners, BTTS | ⏳ |
| **MLS** | Equipos presentes en CONCACAF, datos FotMob abundantes | BTTS, Over goles | ⏳ |

### Tier 2 — Agregar cuando Tier 1 esté validado

| Liga | Por qué |
|------|---------|
| **Libertadores** | Partidos de alta volatilidad — buenas cuotas en favoritos claros |
| **Champions League** | Muy eficiente en 1X2, pero hay edge en corners/tarjetas de grupos |
| **Europa League** | Menos analizada que UCL — más ineficiencias |
| **Premier League** | La más eficiente del mundo, pero volumen altísimo de datos |
| **La Liga** | Modelos Poisson bien calibrados con datos históricos disponibles |
| **Bundesliga** | Partidos con muchos goles — Over goles y BTTS |
| **Serie A** | Históricamente menos goles — Under y BTTS No |
| **Ligue 1** | Mercado relativamente ineficiente comparado con EPL |

### Tier 3 — Largo plazo

| Liga | Por qué |
|------|---------|
| Copa MX / Copa Argentina | Muchos equipos débiles vs fuertes — corners masivos |
| Copa Sudamericana | Similar a Libertadores pero más ineficiente |
| Nations League CONCACAF | Partidos con ELOs muy dispares — resultados predecibles |

---

## Concepto de "apuestas fáciles" por jornada

Las casas cometen errores sistemáticos en ciertos patrones. Estos son los más detectables:

### 1. Dominancia absoluta (ELO gap > 300)
```
Cuando un equipo tiene 300+ puntos ELO sobre el rival:
  P(victoria) ≈ 85%+
  Mercado: Draw No Bet (elimina el empate)
  Cuota esperada: 1.20-1.35
  EV típico si la casa lo paga a 1.25: modelo = 85%, EV = 85%×1.25−1 = +6.25%
```

### 2. Corners seguros — partidos de alta presión ofensiva
```
Cuando ambos equipos están en los top-4 de corner_rate de su liga:
  P(Over 8.5 corners) ≈ 75%+
  La mayoría de las casas usan línea genérica de 9.5 sin calibrar por equipo
  → Edge sistemático detectado en Liga MX (Clausura 2026: Toluca att=+0.24)
```

### 3. Tarjetas en rivalidades + árbitro severo
```
Clásico + árbitro con factor ≥ 1.20 → λ_tarjetas × 1.20+
  P(Over 4.5 tarjetas ponderadas) puede subir de 40% a 62%+
  → Las casas no ajustan por árbitro en Liga MX: edge claro
```

### 4. BTTS No en partidos defensivos de altura
```
Visitante en Toluca/Pachuca (altura 2400-2680m):
  λ_visita ×0.88 → P(visita no anota) sube
  P(BTTS No) puede llegar a 65%+
  → Las casas usan λ genérica sin ajuste de altitud: edge sistemático
```

### 5. Asian Handicap con ELO + forma reciente
```
ELO gap grande + local en racha de 5+ victorias:
  Handicap asiático -1 al local puede tener EV > 8%
  Más difícil que corners/tarjetas, pero alta frecuencia de oportunidades
```

---

## Principio de portafolio: diversificación = menos varianza

Un portafolio de 10 apuestas independientes con p=0.60 y EV=+5% cada una:
```
EV total      = 10 × 5%  = +50% del capital apostado (en 10 apuestas)
Varianza      ↓ por diversificación entre ligas sin correlación
P(perder todo)= (0.40)^10 = 0.01% ← casi imposible

vs apuesta única con p=0.60: P(perder) = 40%
```

**La clave es la independencia entre ligas y mercados.**
Una jornada de Liga MX no está correlacionada con corners en Bundesliga.

### Reglas de portafolio
1. **Máximo 3 apuestas en la misma liga el mismo día** — correlación alta entre partidos de la misma jornada
2. **Máximo 30% del bankroll diario** — protege contra día malo completo
3. **Nunca apostar en la misma dirección dos veces** — si ya aposté Over corners en América-Chivas, no apostar Over en América-Cruz Azul
4. **Apuestas independientes** — corners y tarjetas en el mismo partido están correlacionadas (partido tenso → ambas suben). Elegir la de mayor EV, no ambas
5. **Stop-loss semanal 15%** — si el bankroll baja 15% en una semana, parar y auditar el modelo

---

## El flujo completo del sistema

```
Cada día 8am México (GitHub Actions):
                                          
  [DATOS]                                 
  FotMob → resultados Liga MX             
  FotMob → resultados internacionales     
  FotMob → corners/tarjetas/shots         
  The Odds API → cuotas del día ← PENDIENTE
                                          
  [MODELOS] (re-entrenan automáticamente) 
  modelo_corners.py   ← MLE Dixon-Coles  
  modelo_tarjetas.py  ← Poisson calibrado 
  modelo_btts.py      ← Dixon-Coles ρ    
  modelo_elo.py       ← ELO + Poisson    
                                          
  [BOTS]                                  
  daily_betting_bot.py → probabilidades + EV para 3 días
  retrain_bot.py       → re-calibra si hay datos nuevos  
  audit_bot.py         → valida que todo funcione ← CONSTRUIDO HOY
                                          
  [OUTPUT]                                
  output/reports/betting_FECHA.html       
  output/reports/audit_FECHA.json         
                                          
  [EMAIL]                                 
  Sección betting con value bets          
  Sección audit con estado del sistema    
```

---

## Métricas de éxito del portafolio

| Métrica | Target | Cómo medirlo |
|---------|--------|--------------|
| Brier Score corners | < 0.20 | `retrain_bot.py --evaluate` |
| Brier Score tarjetas | < 0.23 | `retrain_bot.py --evaluate` |
| CLV vs Pinnacle | > 0% en 200+ predicciones | `python scripts/scrape_odds.py --clv` |
| ROI portafolio | > 5% mensual sobre bankroll | `predicciones_log.csv` + bankroll_log |
| Drawdown máximo | < 20% del bankroll | Tracker automático |
| Hit rate corners EV>5% | > 58% | Log de resultados |

## Target junio 2026 — Ruta crítica al dinero real

**Meta del usuario: apostar en producción con bankroll real antes de fin de junio.**

### Lo que ya tenemos ✅
- Modelos corners/tarjetas con skill demostrado (+12/+13%)
- Pipeline automático diario
- 11,601 partidos con cuotas históricas para CLV backtesting
- Casas mexicanas mapeadas (ver `skills/CASAS_APUESTAS_MX.md`)

### Lo que falta — en orden de prioridad

**1. Cuotas Liga MX en tiempo real** ← BLOCKER más urgente
   - The Odds API key (free tier 500 req/mes — suficiente para Liga MX)
   - Script: `update_odds_ligamx.py`
   - Sin esto: no hay EV automático en Liga MX

**2. Logear corners/tarjetas en predicciones_log.csv** ← BLOCKER #2
   - Hoy solo logueamos el resultado 1X2
   - Necesitamos loguear: prob_corners_over85, prob_tarjetas_over45
   - Para tener 50+ predicciones validadas por mercado antes de junio

**3. CLV backtesting con Bundesliga/La Liga** ← validación del modelo
   - Ya tenemos odds_historico.csv con 11,601 partidos
   - Necesitamos predicciones de esas ligas en el tracker

### Proyección realista a junio
```
Jornadas Liga MX restantes Clausura 2026: ~6 (J16-J17 + Liguilla ~4 rondas)
Partidos estimados: ~60 en Liga MX + ~30 internacionales = ~90 predicciones
Corners/tarjetas logueadas: si se activa hoy → 90 picks por mercado para junio

Con 90 predicciones en corners:
  Hit rate esperado: ~62% (modelo tiene skill +12%)
  EV promedio por apuesta: +8-12% (basado en Brier score)
  Con Kelly 25% sobre bankroll $10,000 MXN:
    ~$300-600 por apuesta × 90 apuestas = $27,000-54,000 de turnover
    ROI esperado: 8% sobre turnover = ~$2,160-4,320 MXN de ganancia
    = 21-43% de retorno sobre bankroll inicial ← esto es lo que se puede lograr
```

**Nota científica importante sobre el ROI:**
- 30-50% de ROI sobre BANKROLL es posible con Kelly compuesto si el modelo tiene edge
- 30-50% de ROI sobre cada APUESTA individual NO es realista (eso sería cuotas de 1.30-1.50 con 100% hit rate)
- La magia es el VOLUMEN: muchas apuestas pequeñas con EV positivo → ROI total alto
- El riesgo real está en la varianza: con n=90 y hit rate 62%, puedes tener rachas de 8-10 pérdidas seguidas (absolutamente normal estadísticamente)
- Stop-loss: si el bankroll baja 20% en cualquier semana → parar y auditar

---

## Hoja de ruta por liga

### Fase A — Liga MX (EN PROCESO)
- [x] ELO histórico 2010-2026
- [x] Modelo corners MLE (126 partidos, expandiendo a 835+)
- [x] Modelo tarjetas + rivalidades
- [x] Modelo BTTS Dixon-Coles
- [x] Liga MX Knowledge Base (árbitros, altitud, clásicos)
- [x] daily_betting_bot.py autónomo
- [ ] scrape_odds.py → The Odds API (BLOCKER para EV automático)
- [ ] 200 predicciones en tracker para validar cada mercado

### Fase B — CONCACAF + MLS (próximas 4-6 semanas)
- [ ] Historial CCL completo 2010-2026 (ELO confiable)
- [ ] Data source MLS (FotMob cubre todo)
- [ ] Liga MX Knowledge Base → extender a MLS rivalidades
- [ ] Modelo corners CCL calibrado

### Fase C — Sudamérica (2-3 meses)
- [ ] Brasileirão data pipeline (FotMob league_id: 68)
- [ ] Liga Profesional Argentina (FotMob league_id: 112)
- [ ] Libertadores + Sudamericana
- [ ] Knowledge Base sudamericano (estadios, árbitros, altitud Bogotá/La Paz)

### Fase D — Europa (cuando LATAM esté validado)
- [ ] Bundesliga (liga más amigable para Over goles)
- [ ] Champions League (grupos — menos eficiente que etapas finales)
- [ ] La Liga + Serie A + Ligue 1

---

## Notas de disciplina

> "El peor error en value betting no es perder apuestas — es apostar sin EV positivo demostrado."

- Nunca apostar en un mercado sin 50+ predicciones validadas en ese mercado específico
- Nunca aumentar apuesta después de una racha negativa (Gambler's Fallacy)
- Registrar TODAS las apuestas en `predicciones_log.csv` — sin excepciones
- Auditar el modelo semanalmente con `audit_bot.py`
- Si el CLV promedio cae < 0 en 20 apuestas consecutivas → pausar y revisar el modelo
