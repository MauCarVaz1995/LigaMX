# TWITTER_STRATEGY.md — Estrategia de contenido @Miau_Stats_MX

---

## Cuenta

- **Handle**: @Miau_Stats_MX
- **Nombre**: MAU-STATISTICS
- **Firma en imágenes**: `MAU-STATISTICS` (Bebas Neue, footer derecho, color accent rojo)
- **Nicho**: Análisis estadístico Liga MX + Selecciones Nacionales (México-centric)

---

## Tipos de contenido

| Tipo | Frecuencia | Script | Ejemplo |
|---|---|---|---|
| Predicción de partido | Por jornada (9 partidos) | `gen_predicciones_ligamx_20260404.py` | "Cruz Azul vs Pachuca — ¿quién gana?" |
| Predicción selecciones | En ventanas FIFA | `gen_predicciones_intl_3img.py` | "México vs Portugal — modelo ELO" |
| Infografía post-partido | Tras cada partido | `05_viz_player_performance.py` | Ratings + stats del partido |
| Ranking ELO semanal | Lunes tras jornada | `12_modelo_elo.py` | "ELO actualizado Clausura 2026" |
| Pizza chart P90 | 2-3 por semana | `05_radar_p90.py` | Perfil de jugador destacado |
| Monte Carlo clasificación | Mitad y final de torneo | `14_simulacion_montecarlo.py` | Probabilidades liguilla |
| Reporte de predicciones | Mensual | `04_predicciones_tracker.py reporte` | "X de Y predicciones correctas" |

---

## Formato de tweets por tipo

### Predicción de partido
```
⚽ JORNADA {N} | CLAUSURA 2026

{Local} vs {Visitante}
📊 Modelo ELO + Poisson

🔴 Local: {X}%
⚫ Empate: {X}%
🔵 Visitante: {X}%

#LigaMX #MauStats
```

### Predicción selecciones (ventana FIFA)
```
🌍 AMISTOSO INTERNACIONAL

🇲🇽 México vs {Rival} 🏳️
📊 Modelo ELO histórico (desde 1872)

ELO México: {X} | ELO {Rival}: {X}

#Mexico #SeleccionMexicana #MauStats
```

### Post-partido ratings
```
📊 RATINGS | {Local} {G}-{G} {Visitante}

⭐ MVP: {Jugador} ({Rating})

Estadísticas completas 👇
#LigaMX #{Equipo}
```

---

## Reglas de publicación

1. **No repetir paleta consecutiva** — rotar entre las disponibles
2. **Predicciones antes del partido** — idealmente 2-4 horas antes del kickoff
3. **Post-partido dentro de las primeras 24h** — mientras la conversación está viva
4. **Hashags fijos**: `#LigaMX`, `#MauStats` + `#{Equipo}` relevante
5. **Hilo para predicciones de jornada completa** — un tweet por partido, primer tweet con resumen
6. **Citar fuente del modelo** en tweets de predicción: "Modelo ELO + Poisson"

---

## Calendario tipo (jornada Liga MX)

| Día | Actividad |
|---|---|
| Jueves/Viernes | Publicar predicciones de todos los partidos del fin de semana |
| Sábado/Domingo | Post-partido de los juegos del día (ratings + stats) |
| Lunes | ELO actualizado tras jornada |
| Martes/Miércoles | Pizzas P90 de jugadores destacados del fin de semana |

---

## Ventanas FIFA

Durante ventanas FIFA internacionales:
1. Actualizar `results.csv` con `update_intl_results.py` antes de cada jornada
2. Actualizar ELO con `update_elo_selecciones.py`
3. Generar predicciones con `gen_predicciones_intl_3img.py` (3 imágenes: heatmap + ranking + últimos 5)
4. Publicar el día antes del partido

---

## Métricas de seguimiento

- Acierto de predicciones: ver `predicciones_log.csv` → `python3 04_predicciones_tracker.py reporte`
- Objetivo a largo plazo: >55% acierto en ganador (baseline: 45% si siempre local)
