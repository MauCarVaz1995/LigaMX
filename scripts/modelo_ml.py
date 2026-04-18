#!/usr/bin/env python3
"""
modelo_ml.py — Motor ML para mercados de apuestas (Liga MX)
============================================================
Implementa gradient boosting (LightGBM) sobre features de partido para:
  - corners_over_8.5 / over_9.5 / over_10.5
  - cards_over_3.5 / over_4.5 / over_5.5
  - btts (both teams score)
  - resultado_1X2 (home_win / draw / away_win)

Metodología:
  - Features: ELO diff, rolling form (5 partidos), tasa histórica por equipo,
    altitud, rivalidad, fase de torneo, head-to-head, xG histórico
  - Calibración: isotonic regression (Niculescu-Mizil & Caruana, 2005)
    → mejora las probabilidades bien ordenadas pero mal escaladas del GBM
  - Holdout: walk-forward validation (siempre entrena en pasado, evalúa en futuro)
  - SHAP: feature importance para entender qué mueve cada predicción
  - Persistencia: modelos serializados en data/processed/ml_models/

Uso:
  # Entrenar
  python scripts/modelo_ml.py --train

  # Predecir un partido
  python scripts/modelo_ml.py --local "Toluca" --visita "América" --jornada 15

  # Evaluar métricas
  python scripts/modelo_ml.py --evaluate

  # API (import)
  from scripts.modelo_ml import MLPredictor
  pred = MLPredictor().load()
  result = pred.predict("Toluca", "América", jornada=15)
"""

import json
import sys
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BASE        = Path(__file__).resolve().parent.parent
EVENTS_CSV  = BASE / "data/processed/match_events.csv"
ELO_CSV     = BASE / "data/processed/elo_historico.csv"
MODELS_DIR  = BASE / "data/processed/ml_models"
METRICS_F   = MODELS_DIR / "ml_metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Targets que el modelo predice ──────────────────────────────────────────
TARGETS = {
    "corners_over_8.5":  ("corners_total", 8.5),
    "corners_over_9.5":  ("corners_total", 9.5),
    "corners_over_10.5": ("corners_total", 10.5),
    "cards_over_3.5":    ("cards_total",   3.5),
    "cards_over_4.5":    ("cards_total",   4.5),
    "cards_over_5.5":    ("cards_total",   5.5),
    "btts":              ("btts",          None),
    "home_win":          ("resultado",     "H"),
    "draw":              ("resultado",     "D"),
    "away_win":          ("resultado",     "A"),
}

# ─── Altitud por estadio (metros) ────────────────────────────────────────────
ALTITUDE = {
    "Toluca":       2660, "Pachuca":     2400, "Puebla":   2135,
    "Queretaro FC": 1820, "Chivas":       1550, "Atlas":    1550,
    "Atletico de San Luis": 1850, "San Luis": 1850,
}

# ─── Nombres canónicos (normalización ELO) ──────────────────────────────────
ELO_NAME_MAP = {
    "CF America":          "América",
    "Atletico de San Luis":"San Luis",
    "Mazatlan FC":         "Mazatlán",
    "FC Juarez":           "FC Juárez",
    "Queretaro FC":        "Querétaro",
    "FC Juárez":           "FC Juárez",
}


def norm_elo(name: str) -> str:
    return ELO_NAME_MAP.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _load_elos() -> dict[str, float]:
    """Devuelve el ELO más reciente de cada equipo."""
    df = pd.read_csv(ELO_CSV)
    return df.sort_values("fecha").groupby("equipo")["elo"].last().to_dict()


def _rolling_stats(df: pd.DataFrame, equipo: str, before_date: str,
                   window: int = 5) -> dict:
    """Stats de los últimos `window` partidos del equipo antes de `before_date`."""
    mask_home  = (df["local"]     == equipo) & (df["fecha"] < before_date)
    mask_away  = (df["visitante"] == equipo) & (df["fecha"] < before_date)

    home_rows = df[mask_home].sort_values("fecha").tail(window)
    away_rows = df[mask_away].sort_values("fecha").tail(window)

    def corners_for(eq, rows_h, rows_a):
        c = pd.concat([
            rows_h["corners_local"].rename("c"),
            rows_a["corners_visitante"].rename("c"),
        ]).dropna()
        return float(c.mean()) if len(c) else 4.5

    def corners_against(eq, rows_h, rows_a):
        c = pd.concat([
            rows_h["corners_visitante"].rename("c"),
            rows_a["corners_local"].rename("c"),
        ]).dropna()
        return float(c.mean()) if len(c) else 4.5

    def cards_total(rows_h, rows_a):
        c = pd.concat([
            rows_h["amarillas_total"].rename("c"),
            rows_a["amarillas_total"].rename("c"),
        ]).dropna()
        return float(c.mean()) if len(c) else 4.5

    def goals_scored(rows_h, rows_a):
        g = pd.concat([
            rows_h["goles_local"].rename("g"),
            rows_a["goles_visitante"].rename("g"),
        ])
        return float(g.mean()) if len(g) else 1.2

    def goals_allowed(rows_h, rows_a):
        g = pd.concat([
            rows_h["goles_visitante"].rename("g"),
            rows_a["goles_local"].rename("g"),
        ])
        return float(g.mean()) if len(g) else 1.2

    def xg_scored(rows_h, rows_a):
        x = pd.concat([
            rows_h["xg_local"].rename("x"),
            rows_a["xg_visitante"].rename("x"),
        ]).dropna()
        return float(x.mean()) if len(x) else 1.2

    def win_rate(rows_h, rows_a):
        wins_h = (rows_h["goles_local"] > rows_h["goles_visitante"]).sum()
        wins_a = (rows_a["goles_visitante"] > rows_a["goles_local"]).sum()
        total = len(rows_h) + len(rows_a)
        return float(wins_h + wins_a) / total if total else 0.33

    n_recent = len(home_rows) + len(away_rows)
    return {
        "corners_for":     corners_for(equipo, home_rows, away_rows),
        "corners_against": corners_against(equipo, home_rows, away_rows),
        "cards_avg":       cards_total(home_rows, away_rows),
        "goals_scored":    goals_scored(home_rows, away_rows),
        "goals_allowed":   goals_allowed(home_rows, away_rows),
        "xg_scored":       xg_scored(home_rows, away_rows),
        "win_rate":        win_rate(home_rows, away_rows),
        "n_recent":        n_recent,
    }


def _h2h_stats(df: pd.DataFrame, local: str, visita: str,
               before_date: str, window: int = 6) -> dict:
    """Head-to-head entre local y visita."""
    mask = (
        ((df["local"] == local) & (df["visitante"] == visita)) |
        ((df["local"] == visita) & (df["visitante"] == local))
    ) & (df["fecha"] < before_date)
    h2h = df[mask].sort_values("fecha").tail(window)
    if len(h2h) == 0:
        return {"h2h_corners": 9.3, "h2h_cards": 4.4, "h2h_btts": 0.5, "h2h_n": 0}
    return {
        "h2h_corners": float(h2h["corners_total"].dropna().mean()) if h2h["corners_total"].notna().any() else 9.3,
        "h2h_cards":   float(h2h["amarillas_total"].dropna().mean()) if h2h["amarillas_total"].notna().any() else 4.4,
        "h2h_btts":    float(((h2h["goles_local"] > 0) & (h2h["goles_visitante"] > 0)).mean()),
        "h2h_n":       len(h2h),
    }


def _jornada_to_phase(jornada) -> float:
    """Jornada → float de presión de torneo [0,1]."""
    try:
        j = int(str(jornada).replace("J", "").strip())
    except Exception:
        return 0.5
    if j <= 5:   return 0.1
    if j <= 12:  return 0.4
    if j <= 17:  return 0.8
    return 1.0   # liguilla


def build_features(df: pd.DataFrame, elos: Optional[dict] = None) -> pd.DataFrame:
    """
    Construye la matriz de features a partir de match_events.
    Cada fila = un partido histórico con features pre-partido.
    """
    if elos is None:
        elos = _load_elos()

    df = df.copy().sort_values("fecha").reset_index(drop=True)
    df["fecha"] = df["fecha"].astype(str)

    # Targets
    df["cards_total"]   = df["amarillas_total"].fillna(4.4) + 2 * df["rojas_total"].fillna(0)
    df["btts"]          = ((df["goles_local"] > 0) & (df["goles_visitante"] > 0)).astype(int)
    df["resultado"]     = df.apply(
        lambda r: "H" if r["goles_local"] > r["goles_visitante"]
        else ("D" if r["goles_local"] == r["goles_visitante"] else "A"),
        axis=1,
    )

    rows = []
    for _, row in df.iterrows():
        fecha   = row["fecha"]
        local   = row["local"]
        visita  = row["visitante"]
        jornada = row["jornada"]

        # ELOs
        elo_l = elos.get(norm_elo(local),  1500.0)
        elo_v = elos.get(norm_elo(visita), 1500.0)
        elo_diff = elo_l - elo_v

        # Rolling form
        rl = _rolling_stats(df, local,  fecha, window=5)
        rv = _rolling_stats(df, visita, fecha, window=5)
        h2h = _h2h_stats(df, local, visita, fecha)

        # Altitud
        alt_l = ALTITUDE.get(local,  1200)
        alt_v = ALTITUDE.get(visita, 1200)

        feat = {
            # ELO
            "elo_diff":           elo_diff,
            "elo_local":          elo_l,
            "elo_visita":         elo_v,
            "elo_we":             1 / (1 + 10 ** (-elo_diff / 400)),   # win expected
            # Form local
            "l_corners_for":      rl["corners_for"],
            "l_corners_against":  rl["corners_against"],
            "l_cards_avg":        rl["cards_avg"],
            "l_goals_scored":     rl["goals_scored"],
            "l_goals_allowed":    rl["goals_allowed"],
            "l_xg_scored":        rl["xg_scored"],
            "l_win_rate":         rl["win_rate"],
            # Form visita
            "v_corners_for":      rv["corners_for"],
            "v_corners_against":  rv["corners_against"],
            "v_cards_avg":        rv["cards_avg"],
            "v_goals_scored":     rv["goals_scored"],
            "v_goals_allowed":    rv["goals_allowed"],
            "v_xg_scored":        rv["xg_scored"],
            "v_win_rate":         rv["win_rate"],
            # Suma corners esperados (feature directa)
            "sum_corners_rate":   rl["corners_for"] + rv["corners_for"],
            "sum_cards_rate":     rl["cards_avg"] + rv["cards_avg"],
            # H2H
            "h2h_corners":        h2h["h2h_corners"],
            "h2h_cards":          h2h["h2h_cards"],
            "h2h_btts":           h2h["h2h_btts"],
            "h2h_n":              h2h["h2h_n"],
            # Contexto
            "altitude_local":     alt_l,
            "altitude_visita":    alt_v,
            "altitude_diff":      alt_l - alt_v,
            "phase_pressure":     _jornada_to_phase(jornada),
            # Targets
            "corners_total":      row["corners_total"],
            "cards_total":        row["cards_total"],
            "btts":               row["btts"],
            "resultado":          row["resultado"],
        }
        rows.append(feat)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # ELO y capacidad relativa
    "elo_diff", "elo_local", "elo_visita", "elo_we",
    # Forma reciente local (5 partidos)
    "l_corners_for", "l_corners_against", "l_cards_avg",
    "l_goals_scored", "l_goals_allowed", "l_xg_scored", "l_win_rate",
    # Forma reciente visitante
    "v_corners_for", "v_corners_against", "v_cards_avg",
    "v_goals_scored", "v_goals_allowed", "v_xg_scored", "v_win_rate",
    # Rates combinados
    "sum_corners_rate", "sum_cards_rate",
    # Contexto
    "altitude_local", "altitude_visita", "altitude_diff",
    "phase_pressure",
    # H2H — solo el número de partidos (no el promedio, que da leakage perfecto)
    "h2h_n",
]
# h2h_corners y h2h_cards NO están en FEATURE_COLS porque su promedio histórico
# predice el resultado con 100% de precisión en el holdout (data leakage estructural).
# Se usan como AJUSTE POSTERIOR en MLPredictor.predict() para corregir la lambda."""


def _lgbm_binary():
    """
    LightGBM conservador para datos de fútbol (~686 partidos).
    Parámetros calibrados para evitar sobreajuste:
      - max_depth=4, num_leaves=15: árboles pequeños
      - min_child_samples=20: mínimo de partidos por hoja
      - n_estimators=200: menos árboles, usar early stopping
      - reg_alpha/lambda=0.5: regularización fuerte
    Ref: Chen & He (2016) — XGBoost: A Scalable Tree Boosting System
    """
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def _lgbm_multiclass():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def train(verbose: bool = True) -> dict:
    """
    Entrena modelos para todos los targets.
    Usa walk-forward validation: holdout = últimos 20% de partidos.
    Calibra con IsotonicRegression (mejor que Platt para GBM).
    Retorna métricas de Brier y log-loss.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.preprocessing import LabelEncoder

    df_raw = pd.read_csv(EVENTS_CSV)
    df_raw = df_raw.dropna(subset=["corners_total", "amarillas_total"])
    elos   = _load_elos()

    if verbose:
        print(f"\n── modelo_ml.py — entrenando ({len(df_raw)} partidos) ──")

    feat_df = build_features(df_raw, elos)
    feat_df = feat_df.dropna(subset=FEATURE_COLS)

    X = feat_df[FEATURE_COLS].values
    n = len(feat_df)
    n_holdout = max(int(n * 0.20), 20)
    X_train, X_test = X[:-n_holdout], X[-n_holdout:]

    metrics = {}
    saved   = []

    # ─── Targets binarios ───────────────────────────────────────────────────
    binary_targets = [
        ("corners_over_8.5",  feat_df["corners_total"] > 8.5),
        ("corners_over_9.5",  feat_df["corners_total"] > 9.5),
        ("corners_over_10.5", feat_df["corners_total"] > 10.5),
        ("cards_over_3.5",    feat_df["cards_total"]   > 3.5),
        ("cards_over_4.5",    feat_df["cards_total"]   > 4.5),
        ("cards_over_5.5",    feat_df["cards_total"]   > 5.5),
        ("btts",              feat_df["btts"].astype(bool)),
    ]

    for name, y_series in binary_targets:
        y = y_series.values.astype(int)
        y_train, y_test = y[:-n_holdout], y[-n_holdout:]

        # Calibración con split temporal (no random k-fold) para evitar leakage
        n_cal = max(int(len(y_train) * 0.20), 20)
        X_tr2, X_cal_s = X_train[:-n_cal], X_train[-n_cal:]
        y_tr2, y_cal_s = y_train[:-n_cal], y_train[-n_cal:]

        base = _lgbm_binary()
        base.fit(X_tr2, y_tr2)
        # cv="prefit": usa el estimador ya entrenado, calibra solo en X_cal_s
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_cal_s, y_cal_s)

        prob_test = cal.predict_proba(X_test)[:, 1]
        bs   = float(brier_score_loss(y_test, prob_test))
        bs_naive = float(np.mean((np.full(len(y_test), y_test.mean()) - y_test) ** 2))

        metrics[name] = {
            "brier":       round(bs, 4),
            "brier_naive": round(bs_naive, 4),
            "skill":       round(1 - bs / bs_naive, 3) if bs_naive > 0 else 0,
            "prevalence":  round(float(y_train.mean()), 3),
            "n_train":     len(y_train),
            "n_test":      len(y_test),
        }

        path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(cal, path)
        saved.append(name)

        if verbose:
            sk = metrics[name]["skill"]
            print(f"  {name:25s}  Brier={bs:.4f}  naive={bs_naive:.4f}  skill={sk:+.1%}")

    # ─── Resultado 1X2 (multiclase) ─────────────────────────────────────────
    label_enc = LabelEncoder()
    y_res     = label_enc.fit_transform(feat_df["resultado"].values)
    y_r_train, y_r_test = y_res[:-n_holdout], y_res[-n_holdout:]

    n_cal_mc  = max(int(len(y_r_train) * 0.20), 20)
    X_tr_mc, X_cal_mc = X_train[:-n_cal_mc], X_train[-n_cal_mc:]
    y_tr_mc, y_cal_mc = y_r_train[:-n_cal_mc], y_r_train[-n_cal_mc:]
    base_mc   = _lgbm_multiclass()
    base_mc.fit(X_tr_mc, y_tr_mc)
    cal_mc    = CalibratedClassifierCV(base_mc, method="isotonic", cv="prefit")
    cal_mc.fit(X_cal_mc, y_cal_mc)

    prob_mc   = cal_mc.predict_proba(X_test)
    ll = float(log_loss(y_r_test, prob_mc))
    metrics["resultado_1X2"] = {
        "log_loss":    round(ll, 4),
        "log_loss_naive": round(-np.log(1/3), 4),
        "classes":     list(label_enc.classes_),
        "n_train":     int(len(y_r_train)),
        "n_test":      int(len(y_r_test)),
    }

    path_mc  = MODELS_DIR / "resultado_1X2.pkl"
    path_enc = MODELS_DIR / "resultado_1X2_enc.pkl"
    joblib.dump(cal_mc,    path_mc)
    joblib.dump(label_enc, path_enc)
    saved.append("resultado_1X2")

    if verbose:
        print(f"  {'resultado_1X2':25s}  LogLoss={ll:.4f}  naive={-np.log(1/3):.4f}")

    # ─── SHAP importance (corners_over_9.5) ─────────────────────────────────
    try:
        import shap
        clf = joblib.load(MODELS_DIR / "corners_over_9.5.pkl")
        # CalibratedClassifierCV → acceder al estimador interno
        base_est = clf.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_est)
        shap_vals = explainer.shap_values(X_train[:200])
        mean_abs  = np.abs(shap_vals[1] if isinstance(shap_vals, list) else shap_vals).mean(0)
        importance = {FEATURE_COLS[i]: round(float(mean_abs[i]), 4)
                      for i in range(len(FEATURE_COLS))}
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
        if verbose:
            print(f"\n  SHAP top-8 (corners_over_9.5): {[f'{k}={v}' for k,v in top]}")
        metrics["shap_corners_over_9.5"] = dict(top)
    except Exception as e:
        if verbose:
            print(f"  SHAP skipped: {e}")

    # Guardar métricas
    meta = {
        "trained_at":    datetime.now().isoformat(),
        "n_partidos":    int(len(feat_df)),
        "n_holdout":     int(n_holdout),
        "targets_saved": saved,
        "metrics":       metrics,
    }
    METRICS_F.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if verbose:
        print(f"\n  Métricas → {METRICS_F}")
        print(f"  Modelos  → {MODELS_DIR} ({len(saved)} archivos)")

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

class MLPredictor:
    """Carga todos los modelos entrenados y predice para un partido nuevo."""

    def __init__(self):
        self._models  = {}
        self._enc     = None
        self._elos    = None
        self._df_hist = None
        self._loaded  = False

    def load(self) -> "MLPredictor":
        if not METRICS_F.exists():
            raise FileNotFoundError(
                "Modelos ML no encontrados. Ejecutar: python scripts/modelo_ml.py --train"
            )
        for p in MODELS_DIR.glob("*.pkl"):
            key = p.stem
            self._models[key] = joblib.load(p)
        self._enc     = self._models.pop("resultado_1X2_enc", None)
        self._elos    = _load_elos()
        self._df_hist = pd.read_csv(EVENTS_CSV)
        self._df_hist["fecha"] = self._df_hist["fecha"].astype(str)
        self._loaded  = True
        return self

    def predict(self, local: str, visita: str, jornada=15,
                target_date: Optional[str] = None) -> dict:
        """
        Retorna probabilidades calibradas para todos los mercados.
        target_date: fecha del partido (YYYY-MM-DD). Default: hoy.
        """
        if not self._loaded:
            self.load()

        today_s = target_date or date.today().isoformat()

        # Construir feature row
        elo_l = self._elos.get(norm_elo(local),  1500.0)
        elo_v = self._elos.get(norm_elo(visita), 1500.0)
        elo_diff = elo_l - elo_v

        rl  = _rolling_stats(self._df_hist, local,  today_s, window=5)
        rv  = _rolling_stats(self._df_hist, visita, today_s, window=5)
        h2h = _h2h_stats(self._df_hist, local, visita, today_s)

        alt_l = ALTITUDE.get(local,  1200)
        alt_v = ALTITUDE.get(visita, 1200)

        x = [[
            elo_diff, elo_l, elo_v, 1 / (1 + 10 ** (-elo_diff / 400)),
            rl["corners_for"],    rl["corners_against"], rl["cards_avg"],
            rl["goals_scored"],   rl["goals_allowed"],   rl["xg_scored"],  rl["win_rate"],
            rv["corners_for"],    rv["corners_against"], rv["cards_avg"],
            rv["goals_scored"],   rv["goals_allowed"],   rv["xg_scored"],  rv["win_rate"],
            rl["corners_for"] + rv["corners_for"],
            rl["cards_avg"]   + rv["cards_avg"],
            alt_l, alt_v, alt_l - alt_v,
            _jornada_to_phase(jornada),
            h2h["h2h_n"],
        ]]

        result: dict = {
            "local":   local,
            "visita":  visita,
            "jornada": jornada,
            "elo_local":  round(elo_l, 1),
            "elo_visita": round(elo_v, 1),
        }

        # Binarios (modelo base sin h2h_corners)
        binary_keys = [
            "corners_over_8.5", "corners_over_9.5", "corners_over_10.5",
            "cards_over_3.5",   "cards_over_4.5",   "cards_over_5.5",
            "btts",
        ]
        for k in binary_keys:
            if k in self._models:
                p = float(self._models[k].predict_proba(x)[0, 1])
                # Ajuste posterior por h2h si hay suficientes antecedentes
                # h2h_corners actúa como prior bayesiano: mezcla 30% h2h con 70% modelo
                if k.startswith("corners") and h2h["h2h_n"] >= 3:
                    threshold = float(k.split("_")[-1])
                    h2h_signal = 1.0 if h2h["h2h_corners"] > threshold else 0.0
                    p = 0.70 * p + 0.30 * h2h_signal
                result[k] = round(p, 4)

        # Enforcer monotonicity: P(Over 8.5) >= P(Over 9.5) >= P(Over 10.5)
        # y P(Over 3.5) >= P(Over 4.5) >= P(Over 5.5)
        # Los modelos independientes pueden violar esto — corregir post-hoc
        for lo, mid, hi in [
            ("corners_over_8.5", "corners_over_9.5", "corners_over_10.5"),
            ("cards_over_3.5",   "cards_over_4.5",   "cards_over_5.5"),
        ]:
            if all(k in result for k in [lo, mid, hi]):
                # P(lo) >= P(mid) >= P(hi)
                p_hi  = result[hi]
                p_mid = max(result[mid], p_hi)
                p_lo  = max(result[lo],  p_mid)
                result[lo] = round(p_lo, 4)
                result[mid] = round(p_mid, 4)

        # 1X2
        if "resultado_1X2" in self._models and self._enc is not None:
            probs = self._models["resultado_1X2"].predict_proba(x)[0]
            for cls, p in zip(self._enc.classes_, probs):
                key = {"H": "p_home", "D": "p_draw", "A": "p_away"}[cls]
                result[key] = round(float(p), 4)

        # Métricas de contexto
        result["context"] = {
            "elo_diff":          round(elo_diff, 1),
            "sum_corners_rate":  round(rl["corners_for"] + rv["corners_for"], 2),
            "sum_cards_rate":    round(rl["cards_avg"]   + rv["cards_avg"],   2),
            "h2h_n":             h2h["h2h_n"],
            "altitude_local":    alt_l,
            "altitude_visita":   alt_v,
        }

        return result

    def ev(self, pred: dict, odds: dict) -> dict:
        """
        Calcula Expected Value para cada mercado con cuotas dadas.
        odds: {"corners_over_9.5": 1.90, "btts": 1.75, ...}
        Retorna dict con EV y tag VALUE/borderline/skip.
        """
        evs = {}
        for market, cuota in odds.items():
            prob = pred.get(market)
            if prob is None:
                continue
            ev = prob * cuota - 1
            evs[market] = {
                "prob":  round(prob, 4),
                "cuota": cuota,
                "ev":    round(ev, 4),
                "tag":   "✅ VALUE" if ev > 0.05 else ("⚠️ borderline" if ev > 0 else "❌"),
            }
        return evs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Motor ML para apuestas Liga MX")
    parser.add_argument("--train",    action="store_true", help="Entrenar modelos")
    parser.add_argument("--evaluate", action="store_true", help="Mostrar métricas")
    parser.add_argument("--local",    type=str, help="Equipo local")
    parser.add_argument("--visita",   type=str, help="Equipo visitante")
    parser.add_argument("--jornada",  type=int, default=15)
    parser.add_argument("--odds",     type=str, default="{}", help="JSON de cuotas")
    args = parser.parse_args()

    if args.train:
        train(verbose=True)
        return

    if args.evaluate:
        if not METRICS_F.exists():
            print("No hay métricas. Ejecutar --train primero.")
            return
        m = json.loads(METRICS_F.read_text())
        print(f"\n── Métricas ML ({m['trained_at'][:10]}) ──")
        print(f"  Dataset: {m['n_partidos']} partidos | holdout: {m['n_holdout']}")
        for name, met in m["metrics"].items():
            if "brier" in met:
                sk = met.get("skill", "?")
                print(f"  {name:25s}  Brier={met['brier']:.4f}  skill={sk:+.1%}")
            elif "log_loss" in met:
                print(f"  {name:25s}  LogLoss={met['log_loss']:.4f}  naive={met['log_loss_naive']:.4f}")
        return

    if args.local and args.visita:
        pred = MLPredictor().load().predict(args.local, args.visita, args.jornada)
        print(json.dumps(pred, indent=2, ensure_ascii=False))
        odds = json.loads(args.odds)
        if odds:
            ev = MLPredictor().load().ev(pred, odds)
            print("\n── EV ──")
            print(json.dumps(ev, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
