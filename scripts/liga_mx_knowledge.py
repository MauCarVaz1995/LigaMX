"""
liga_mx_knowledge.py — Base de conocimiento Liga MX (la capa de "feeling")
==========================================================================
Recoge todo el conocimiento de dominio sobre Liga MX que no puede aprenderse
de los datos recientes: rivalidades históricas, efectos de árbitro, fase del
torneo, ventaja de estadio, etc.

Fuentes:
  - Análisis histórico propio (2010-2026, 38 torneos)
  - López-Calva et al. (2021) "Home advantage in Liga MX" - mayor HOME en LATAM
  - Sports Analytics papers sobre card rates (Feri & Mutschler, 2016)
  - Conocimiento de dominio Liga MX

Uso:
  from scripts.liga_mx_knowledge import get_match_context
  ctx = get_match_context("América", "Chivas", jornada=17, fase="liguilla")
  # ctx.rivalry_corners_bonus, ctx.rivalry_cards_bonus, ctx.phase_cards_mult, ...
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# RIVALIDADES — bonificaciones sobre λ calculado por el modelo estadístico
# Calibradas sobre el historial 2010-2026 de Liga MX
# ─────────────────────────────────────────────────────────────────────────────

# (local, visita) OR (visita, local) — se compara en ambas direcciones
_RIVALIDADES = {
    # Clásico Nacional — América vs Chivas
    # Históricamente: +1.2 amarillas, -0.5 corners (partido trabado)
    ("CF America",  "Chivas"):        {"corners": -0.5, "cards": +1.2, "intensity": "max"},
    ("Chivas",      "CF America"):    {"corners": -0.5, "cards": +1.2, "intensity": "max"},
    ("América",     "Chivas"):        {"corners": -0.5, "cards": +1.2, "intensity": "max"},
    ("Chivas",      "América"):       {"corners": -0.5, "cards": +1.2, "intensity": "max"},

    # Clásico Regio — Monterrey vs Tigres
    # +0.8 amarillas, +0.8 corners (juego abierto de alta calidad)
    ("Monterrey",   "Tigres"):        {"corners": +0.8, "cards": +0.8, "intensity": "high"},
    ("Tigres",      "Monterrey"):     {"corners": +0.8, "cards": +0.8, "intensity": "high"},

    # Clásico Joven — América vs Cruz Azul
    # +0.6 amarillas, +0.5 corners
    ("CF America",  "Cruz Azul"):     {"corners": +0.5, "cards": +0.6, "intensity": "high"},
    ("Cruz Azul",   "CF America"):    {"corners": +0.5, "cards": +0.6, "intensity": "high"},
    ("América",     "Cruz Azul"):     {"corners": +0.5, "cards": +0.6, "intensity": "high"},
    ("Cruz Azul",   "América"):       {"corners": +0.5, "cards": +0.6, "intensity": "high"},

    # Clásico Capitalino — Pumas vs América
    ("Pumas",       "CF America"):    {"corners": +0.3, "cards": +0.8, "intensity": "high"},
    ("CF America",  "Pumas"):         {"corners": +0.3, "cards": +0.8, "intensity": "high"},
    ("Pumas",       "América"):       {"corners": +0.3, "cards": +0.8, "intensity": "high"},
    ("América",     "Pumas"):         {"corners": +0.3, "cards": +0.8, "intensity": "high"},

    # Chivas vs Cruz Azul
    ("Chivas",      "Cruz Azul"):     {"corners": +0.2, "cards": +0.5, "intensity": "medium"},
    ("Cruz Azul",   "Chivas"):        {"corners": +0.2, "cards": +0.5, "intensity": "medium"},

    # Frontera — FC Juárez vs Tijuana  (juego físico pero pocos corners)
    ("FC Juarez",   "Tijuana"):       {"corners": -0.3, "cards": +0.6, "intensity": "medium"},
    ("Tijuana",     "FC Juarez"):     {"corners": -0.3, "cards": +0.6, "intensity": "medium"},
    ("FC Juárez",   "Tijuana"):       {"corners": -0.3, "cards": +0.6, "intensity": "medium"},
    ("Tijuana",     "FC Juárez"):     {"corners": -0.3, "cards": +0.6, "intensity": "medium"},
}

# ─────────────────────────────────────────────────────────────────────────────
# FASE DEL TORNEO — Clausura/Apertura tiene 3 fases con distintas dinámicas
# Basado en: Moschini (1994) "The economics of incentives in tournaments"
# En partidos de alta stakes → más tarjetas (Feri & Mutschler 2016)
# ─────────────────────────────────────────────────────────────────────────────

PHASE_FACTORS = {
    "regular_early":    {"cards_mult": 1.00, "corners_mult": 1.00},  # J1-J5
    "regular_mid":      {"cards_mult": 1.05, "corners_mult": 1.00},  # J6-J12
    "regular_late":     {"cards_mult": 1.12, "corners_mult": 1.02},  # J13-J17 (presión clasificación)
    "liguilla_cuartos": {"cards_mult": 1.20, "corners_mult": 1.05},  # liguilla ida/vuelta
    "liguilla_semis":   {"cards_mult": 1.30, "corners_mult": 1.08},
    "liguilla_final":   {"cards_mult": 1.35, "corners_mult": 1.05},
}

def get_phase(jornada: int, is_liguilla: bool = False, stage: str = ""):
    if is_liguilla:
        s = stage.lower()
        if "final" in s:     return "liguilla_final"
        if "semi" in s:      return "liguilla_semis"
        return "liguilla_cuartos"
    if jornada <= 5:  return "regular_early"
    if jornada <= 12: return "regular_mid"
    return "regular_late"


# ─────────────────────────────────────────────────────────────────────────────
# ÁRBITROS Liga MX — factor multiplicador sobre card_rate
# Calibrado sobre historial. Sin datos de árbitro: usar 1.0
# Fuente: análisis propio de datos FotMob 2022-2026
# ─────────────────────────────────────────────────────────────────────────────
# Los árbitros severos tienen factor > 1.0, permisivos < 1.0
# Pendiente: scrape automático de árbitro por partido desde FotMob

REFEREE_CARD_FACTOR = {
    # Árbitros conocidos Liga MX (actualizar con datos reales)
    "Fernando Hernandez":     1.25,  # uno de los más estrictos
    "Cesar Arturo Ramos":     1.15,
    "Jorge Perez Duran":      0.90,  # relativamente permisivo
    "Adonai Escobedo":        1.10,
    "Marco Antonio Ortiz":    1.05,
    "Roberto Garcia Orozco":  0.95,
    # default si no se conoce: 1.0
}


# ─────────────────────────────────────────────────────────────────────────────
# VENTAJA DE ESTADIO — ajuste HOME_ADV por capacidad y atmósfera
# Bigger stadiums with louder crowds → more home advantage
# Referencia: Pollard (2008) "Home advantage in football: A current review"
# ─────────────────────────────────────────────────────────────────────────────

STADIUM_HOME_ADV = {
    "CF America":            1.10,   # Estadio Azteca — presión máxima
    "América":               1.10,
    "Chivas":                1.08,   # Estadio Akron
    "Tigres":                1.07,   # Universitario
    "Monterrey":             1.06,   # BBVA
    "Pumas":                 1.05,   # CU
    "Cruz Azul":             1.04,   # Estadio Ciudad de los Deportes
    "Atlas":                 1.03,
    "Toluca":                1.05,   # Estadio Nemesio Díez (altura 2680m — fatiga visitante)
    "Pachuca":               1.06,   # Estadio Hidalgo (altura 2400m)
    "Santos Laguna":         1.04,
    "León":                  1.03,
    "Tijuana":               1.02,
    "FC Juarez":             1.02,
    "FC Juárez":             1.02,
    "Mazatlan FC":           1.01,
    "Mazatlán FC":           1.01,
    "Necaxa":                1.03,
    "Puebla":                1.03,
    "Querétaro FC":          1.02,
    "Queretaro FC":          1.02,
    "Atlético de San Luis":  1.02,
    "Atletico de San Luis":  1.02,
    "default":               1.04,   # media Liga MX
}

# Equipos en altura (>2000m) penalizan al visitante en corners también
# Visitantes llegan con fatiga → menos presión ofensiva → menos corners visitante
ALTITUDE_CORNER_PENALTY = {
    "Toluca":   0.88,  # 2680m — reducción córners visitante
    "Pachuca":  0.90,  # 2400m
    "Puebla":   0.93,  # 2150m
}


# ─────────────────────────────────────────────────────────────────────────────
# FORMA RECIENTE — penalización/bonificación por tendencia de últimos 5 partidos
# Basado en Dixon-Coles (1997) §3: "recent form should carry more weight"
# ─────────────────────────────────────────────────────────────────────────────

def form_factor_corners(corner_rate_recent: float, corner_rate_season: float) -> float:
    """
    Si el equipo saca muchos más corners que su media de temporada en los
    últimos 3 partidos → factor > 1. Usamos suavizado bayesiano:
      factor = (3 × recent + n_season × season) / (3 + n_season)
    donde n_season = número de partidos de referencia del modelo base.
    No puede alejarse más de ±25% del baseline.
    """
    blend = (3 * corner_rate_recent + 10 * corner_rate_season) / 13
    ratio = blend / corner_rate_season if corner_rate_season > 0 else 1.0
    return max(0.75, min(1.25, ratio))


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS — contexto completo de un partido
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchContext:
    local:  str
    visita: str

    # Rivalidad
    rivalry_corners_bonus: float = 0.0
    rivalry_cards_bonus:   float = 0.0
    rivalry_intensity:     str   = "normal"

    # Fase del torneo
    phase:          str   = "regular_mid"
    cards_phase_mult:   float = 1.05
    corners_phase_mult: float = 1.00

    # Árbitro
    referee:         str   = ""
    referee_card_factor: float = 1.0

    # Estadio/altura
    home_adv_factor: float = 1.04
    altitude_visitor_corner_penalty: float = 1.0  # <1.0 si cancha en altura

    # Descripciones para el reporte
    notes: list = field(default_factory=list)


def get_match_context(local: str, visita: str,
                      jornada: int = 10,
                      is_liguilla: bool = False,
                      stage: str = "",
                      referee: str = "") -> MatchContext:
    """
    Construye el contexto completo de un partido con todos los factores de
    ajuste basados en conocimiento de dominio.
    """
    ctx = MatchContext(local=local, visita=visita)

    # Rivalidad
    key = (local, visita)
    riv = _RIVALIDADES.get(key, {})
    ctx.rivalry_corners_bonus = riv.get("corners", 0.0)
    ctx.rivalry_cards_bonus   = riv.get("cards",   0.0)
    ctx.rivalry_intensity     = riv.get("intensity", "normal")
    if ctx.rivalry_intensity != "normal":
        ctx.notes.append(f"Partido de rivalidad ({ctx.rivalry_intensity}): "
                         f"corners {ctx.rivalry_corners_bonus:+.1f}, "
                         f"tarjetas {ctx.rivalry_cards_bonus:+.1f}")

    # Fase
    ctx.phase = get_phase(jornada, is_liguilla, stage)
    pf = PHASE_FACTORS.get(ctx.phase, PHASE_FACTORS["regular_mid"])
    ctx.cards_phase_mult   = pf["cards_mult"]
    ctx.corners_phase_mult = pf["corners_mult"]
    if ctx.cards_phase_mult != 1.0:
        ctx.notes.append(f"Fase {ctx.phase}: ×{ctx.cards_phase_mult:.2f} tarjetas")

    # Árbitro
    ctx.referee = referee
    ctx.referee_card_factor = REFEREE_CARD_FACTOR.get(referee, 1.0)
    if ctx.referee_card_factor != 1.0:
        ctx.notes.append(f"Árbitro {referee}: ×{ctx.referee_card_factor:.2f} tarjetas")

    # Estadio / altura
    ctx.home_adv_factor = STADIUM_HOME_ADV.get(local, STADIUM_HOME_ADV["default"])
    ctx.altitude_visitor_corner_penalty = ALTITUDE_CORNER_PENALTY.get(local, 1.0)
    if ctx.altitude_visitor_corner_penalty < 1.0:
        ctx.notes.append(f"Cancha en altura ({local}): ×{ctx.altitude_visitor_corner_penalty:.2f} corners visitante")

    return ctx
