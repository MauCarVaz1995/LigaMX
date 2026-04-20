# CASAS_APUESTAS_MX.md — Léxico y mercados de casas mexicanas

> Creado: 2026-04-19. Referencia para entender cómo presentar picks y qué mercados ofrece cada casa.

---

## Las casas principales en México

| Casa | Formato momio | Fortaleza | Límites Liga MX | Notas |
|---|---|---|---|---|
| **Caliente** | Americano / Decimal | La más grande de MX, omnipresente | Bajos (~$500-2,000 MXN) | Físico + online. Mueve mucho pero limita rápido a ganadores |
| **Bet365** | Decimal | Mercados más completos, corners/tarjetas | Medios (~$5,000 MXN) | Internacional, acepta MX. Mejor cobertura de mercados especiales |
| **Codere** | Decimal / Americano | Buena cobertura Liga MX | Medios | Español, fuerte en MX. App bastante usable |
| **Betcris** | Americano | Especialista LATAM, mucho Liga MX | Medios-Altos | La más popular entre apostadores MX serios |
| **Strendus** (ex Betsson) | Decimal | Europeo, mercados amplios | Medios | Buenas cuotas en partidos europeos |
| **1xBet** | Decimal | Amplísimo (miles de mercados) | Altos | Menos regulado. No limitación agresiva. Riesgoso como empresa |
| **Equipo Mexico** | Americano | Local, enfocado en LigaMX | Bajos | Relativamente nuevo, límites bajos |
| **Bodog** | Americano | Clásico, confiable | Medios | Fuerte en americano, fútbol americano + soccer |

**Regla práctica:** Para corners y tarjetas, Bet365 y Codere tienen mejor cobertura. Para maximizar volumen sin restricciones, 1xBet o Betcris. Caliente tiene presencia física pero limita muy rápido a cuentas ganadoras.

---

## Léxico de momios — Conversión de formatos

### Formato Americano (el más común en MX)
```
Positivo (+150): ganas $150 por cada $100 apostados
  → cuota decimal = (150/100) + 1 = 2.50
  → prob implícita = 100 / (150+100) = 40%

Negativo (-200): debes apostar $200 para ganar $100
  → cuota decimal = (100/200) + 1 = 1.50
  → prob implícita = 200 / (200+100) = 66.7%
```

### Formato Decimal (europeo — Bet365, Strendus)
```
2.50 → ganas 2.50 por cada 1 apostado (incluye el capital)
  → ganancia neta = 1.50 por cada 1
  → prob implícita = 1/2.50 = 40%
```

### Conversión rápida
| Americano | Decimal | Prob implícita |
|---|---|---|
| +300 | 4.00 | 25% |
| +200 | 3.00 | 33% |
| +150 | 2.50 | 40% |
| +100 | 2.00 | 50% |
| -110 | 1.91 | 52.4% |
| -120 | 1.83 | 54.5% |
| -150 | 1.67 | 60% |
| -200 | 1.50 | 66.7% |
| -300 | 1.33 | 75% |

---

## Mercados disponibles en casas MX (para Liga MX)

### Los que ofrecen corners/tarjetas
| Mercado | Nombre en Caliente | Nombre en Bet365 | Disponible |
|---|---|---|---|
| Corners Over/Under | "Total tiros de esquina" | "Corner Kicks" | Bet365, Codere, Betcris ✅ |
| Tarjetas Over/Under | "Total tarjetas" | "Cards" | Bet365, Codere ✅ |
| BTTS | "Ambos equipos anotan" | "Both Teams to Score" | Todas ✅ |
| Over/Under goles | "Total goles" | "Total Goals" | Todas ✅ |
| Resultado 1X2 | "3 caminos" / "1X2" | "Match Result" | Todas ✅ |
| Asian Handicap | "Hándicap asiático" | "Asian Handicap" | Bet365, Betcris ✅ |
| Draw No Bet | "Empate no apuesta" | "Draw No Bet" | Bet365, Codere ✅ |

### Caliente específico (léxico propio)
- **"Momio de la línea"** = cuota principal 1X2
- **"Corrida de línea"** = cuando la cuota cambia antes del partido
- **"Cappers"** = tipsters / pronosticadores
- **"En vivo" / "En directo"** = live betting
- **"Sistema"** = apuesta tipo parlay estructurado
- **"Pick de la semana"** = apuesta destacada

---

## Conceptos clave para el sistema

### Overround / Margen de la casa
```
La suma de prob implícitas SIEMPRE supera 100% — esa diferencia es la ganancia de la casa.

Ejemplo 1X2 en Caliente:
  Local: -130 → 56.5%
  Empate: +220 → 31.3%
  Visita: +250 → 28.6%
  Total: 116.4% → margen = 16.4% (MUY alto — Caliente cobra mucho)

Pinnacle (el más eficiente del mundo):
  Mismo partido: margen ≈ 2-3%

→ La estrategia es encontrar partidos donde la prob del MODELO > prob implícita de la casa
  incluso después de descontar el margen.
```

### Qué significa "cuota de valor"
```
Si nuestro modelo dice 65% y Bet365 paga a +140 (cuota 2.40, prob implícita 41.7%):
  EV = 0.65 × 2.40 - 1 = +0.56 → 56% de ventaja → APUESTA INMEDIATA

Si nuestro modelo dice 55% y Bet365 paga a +100 (cuota 2.00, prob implícita 50%):
  EV = 0.55 × 2.00 - 1 = +0.10 → 10% de ventaja → apostar (si EV > 0.05)

Si nuestro modelo dice 55% y Bet365 paga a -110 (cuota 1.91, prob implícita 52.4%):
  EV = 0.55 × 1.91 - 1 = +0.05 → apenas value → apostar con cautela
```

### Límites y restricciones
- **Caliente** limita cuentas ganadoras muy rápido (~50 apuestas ganadoras consecutivas)
- **Bet365** tiene límites por mercado — corners/tarjetas suelen ser más bajos que 1X2
- **Estrategia:** Distribuir apuestas entre 3-4 casas para evitar limitación prematura
- **Cuentas múltiples:** NO recomendado (viola ToS). Mejor: cónyuge/familiar de confianza con su cuenta
- **Betcris y 1xBet** son los más tolerantes con ganadores consistentes en LATAM

---

## Formato de picks para el sistema

Cuando el betting_bot detecta value, el formato para actuar es:

```
Partido:   América vs Cruz Azul — J16 Clausura 2026
Mercado:   Corners Over 8.5
Prob modelo: 68%
Cuota mínima para EV>5%: > 1.56 (cualquier momio mejor que -177 americano)

Bet365 ofrece: 1.72 → EV = 68% × 1.72 - 1 = +16.9% ✅ APOSTAR
Caliente ofrece: 1.55 → EV = 68% × 1.55 - 1 = +5.4% ✅ APOSTAR (mínimo)
Caliente ofrece: 1.45 → EV = 68% × 1.45 - 1 = -1.4% ❌ NO APOSTAR

Bankroll asignado a corners: $5,000 MXN
Kelly 25%: apostar ~$340 MXN en este partido
```

---

## Sobre parlays / combinadas

**NO usar parlays para value betting.**

```
Parlay de 3 picks con EV=+5% cada uno:
  EV combinada = (1.05)^3 - 1 = +15.7% sobre el capital apostado

Parlay de 3 picks con margen de casa incluido:
  Casa cobra margen en CADA pata → el margen se multiplica
  Con margen 5% por pata: EV real = (1.05 × 0.95)^3 - 1 ≈ -4%

→ Los parlays convierten EV positivo en EV NEGATIVO para el portafolio.
→ Excepción: parlays de correlación positiva (e.g., Over goles + BTTS Sí son el mismo partido
  pero son eventos correlacionados — algunas casas los pagan incorrectamente alto)
```

---

## The Odds API — Para Liga MX en tiempo real

```
Tier gratis: 500 requests/mes
Endpoint: GET https://api.the-odds-api.com/v4/sports/{sport}/odds/
          ?apiKey={KEY}&regions=mx&markets=h2h,totals,btts&bookmakers=bet365,betcris

Liga MX sport key: "soccer_mexico_ligamx"
Mercados disponibles: h2h (1X2), totals (Over/Under goles), btts (ambos anotan)
Corners/tarjetas: disponibles en plan pagado (~$10 USD/mes Basic)

Guardar en: data/processed/odds_ligamx.csv
Columnas: fecha, local, visitante, bookmaker, mercado, linea, odd_over, odd_under
```

Cuando tengamos el API key, agregar `update_odds_ligamx.py` al pipeline para EV automático en Liga MX.
