"""Scoring and exact per-alert attribution.

`contributions()` is the reason this project uses logistic regression for
alerting: for a linear model, coef[i] * scaled_x[i] is the exact additive
contribution to the logit, so an alert can say *why* the model disagreed with
the market. No SHAP, no surrogate, no s390x build problem.

Acceptance: `python -m models.score --map-uid <uid>` prints probabilities and
asserts sum(contributions) + intercept == raw logit within 1e-9.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import train as train_mod
from pipeline import align, features, io

ARTIFACT_DIR = Path(os.environ.get("CS2_ARTIFACT_DIR", "models/artifacts"))
MODEL_VERSION = "v1"
_CACHE: dict = {}

DEFAULTS = {
    "score_team1": 0.0, "score_team2": 0.0, "round_num": 1.0,
    "team1_is_ct": 1.0, "equip_team1": 0.0, "equip_team2": 0.0,
    "balance_team1": 0.0, "balance_team2": 0.0,
    "spend_team1": 0.0, "spend_team2": 0.0,
    "armor_diff": 0.0, "helmet_diff": 0.0, "defuser_diff": 0.0,
    "map_index": 1.0, "hltv_stars": 0.0,
}


class ModelsMissing(RuntimeError):
    pass


def _paths(out_dir: Path) -> dict[str, Path]:
    return {
        "logit": out_dir / f"logit-{MODEL_VERSION}.joblib",
        "logit_cal": out_dir / f"logit-{MODEL_VERSION}-cal.joblib",
        "xgb": out_dir / f"xgb-{MODEL_VERSION}-cal.joblib",
        "metrics": out_dir / "metrics.json",
    }


def load_models(out_dir: Path | None = None, refresh: bool = False) -> dict:
    """Load artifacts once into a module cache."""
    if _CACHE and not refresh:
        return _CACHE
    out_dir = Path(out_dir) if out_dir else ARTIFACT_DIR
    p = _paths(out_dir)
    if not p["logit"].exists():
        raise ModelsMissing(f"{p['logit']} not found: run `python -m models.train`")
    import json
    metrics = json.loads(p["metrics"].read_text()) if p["metrics"].exists() else {}
    _CACHE.update({
        "logit": train_mod.load(p["logit"]),
        "logit_cal": train_mod.load(p["logit_cal"]) if p["logit_cal"].exists() else None,
        "xgb_cal": train_mod.load(p["xgb"]) if p["xgb"].exists() else None,
        "features": metrics.get("features") or [],
        "metrics": metrics,
    })
    return _CACHE


def _align_columns(X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    if names and list(X.columns) != names:
        X = X.reindex(columns=names, fill_value=0.0)
    return X.fillna(0.0)


def score_panel(X: pd.DataFrame) -> np.ndarray:
    m = load_models()
    X = _align_columns(X, m["features"])
    model = m["logit_cal"] or m["logit"]
    return model.predict_proba(X)[:, 1]


def score_panel_xgb(X: pd.DataFrame) -> np.ndarray | None:
    m = load_models()
    if m["xgb_cal"] is None:
        return None
    return m["xgb_cal"].predict_proba(_align_columns(X, m["features"]))[:, 1]


def contributions(X_row: pd.Series | pd.DataFrame, top: int | None = 5) -> list[dict]:
    """Exact additive contributions to the logit, largest magnitude first.

    `top=None` returns every feature, which is what the reconstruction check
    needs: sum(all contributions) + intercept must equal the raw logit.
    """
    m = load_models()
    row = X_row.to_frame().T if isinstance(X_row, pd.Series) else X_row.iloc[[0]]
    row = _align_columns(row, m["features"]).astype(float)
    pipe = m["logit"]
    scaler: StandardScalerLike = pipe.named_steps["scaler"]  # type: ignore[assignment]
    clf = pipe.named_steps["clf"]
    scaled = scaler.transform(row)[0]
    coef = clf.coef_[0]
    contrib = coef * scaled
    order = np.argsort(-np.abs(contrib))
    if top is not None:
        order = order[:top]
    out = []
    for i in order:
        name = m["features"][i]
        out.append({
            "name": name,
            "value": round(float(row.iloc[0, i]), 4),
            "contribution": float(contrib[i]),  # unrounded: sums must reconstruct the logit
        })
    return out


def raw_logit(X_row: pd.Series | pd.DataFrame) -> float:
    m = load_models()
    row = X_row.to_frame().T if isinstance(X_row, pd.Series) else X_row.iloc[[0]]
    row = _align_columns(row, m["features"]).astype(float)
    pipe = m["logit"]
    z = pipe.named_steps["scaler"].transform(row)[0]
    clf = pipe.named_steps["clf"]
    return float(z @ clf.coef_[0] + clf.intercept_[0])


def intercept() -> float:
    return float(load_models()["logit"].named_steps["clf"].intercept_[0])


def round_state_frame(state: dict) -> pd.DataFrame:
    """Build the one-row feature frame for /api/score from a request body."""
    m = load_models()
    names = m["features"] or list(DEFAULTS)
    row = {n: float(DEFAULTS.get(n, 0.0)) for n in names}

    def put(k: str, v) -> None:
        if k in row and v is not None:
            row[k] = float(v)

    s1 = float(state.get("score_team1", 0) or 0)
    s2 = float(state.get("score_team2", 0) or 0)
    e1 = float(state.get("equip_team1", 0) or 0)
    e2 = float(state.get("equip_team2", 0) or 0)
    rnd = float(state.get("round_num", 1) or 1)
    for k, v in (("score_team1", s1), ("score_team2", s2),
                 ("equip_team1", e1), ("equip_team2", e2),
                 ("round_num", rnd),
                 ("balance_team1", state.get("balance_team1")),
                 ("balance_team2", state.get("balance_team2")),
                 ("spend_team1", state.get("spend_team1")),
                 ("spend_team2", state.get("spend_team2")),
                 ("map_index", state.get("map_index")),
                 ("hltv_stars", state.get("hltv_stars"))):
        put(k, v)
    put("score_diff", s1 - s2)
    put("equip_diff", e1 - e2)
    put("balance_diff", row.get("balance_team1", 0) - row.get("balance_team2", 0))
    put("spend_diff", row.get("spend_team1", 0) - row.get("spend_team2", 0))
    put("is_pistol", 1 if rnd in (1, 13) else 0)
    put("is_second_half", 1 if rnd >= 13 else 0)
    put("team1_is_ct", 1 if str(state.get("team1_side", "ct")).lower() == "ct" else 0)

    for col, key in (("map_name", "map_name"),
                     ("buy_team1", "buy_team1"), ("buy_team2", "buy_team2")):
        val = str(state.get(key, "") or "").strip()
        target = f"{col}__{val}"
        if val and target in row:
            row[target] = 1.0
    return pd.DataFrame([row], columns=names)


def score_round_state(state: dict) -> dict:
    t0 = time.perf_counter()
    X = round_state_frame(state)
    p_logit = float(score_panel(X)[0])
    px = score_panel_xgb(X)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "p_logit": round(p_logit, 4),
        "p_xgb": round(float(px[0]), 4) if px is not None else None,
        "model": f"logit-{MODEL_VERSION}",
        "contributions": contributions(X),
        "latency_ms": latency_ms,
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-uid")
    ap.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    args = ap.parse_args()

    load_models(Path(args.artifact_dir), refresh=True)
    print(f"models loaded from {args.artifact_dir}; arch={platform.machine()}")
    if not args.map_uid:
        r = score_round_state({"score_team1": 7, "score_team2": 6, "round_num": 14,
                               "team1_side": "ct", "equip_team1": 12500,
                               "equip_team2": 11250, "buy_team1": "full",
                               "buy_team2": "semi", "map_name": "de_overpass"})
        print(r)
        return

    panel = align.attach_market(align.map_panel(args.map_uid))
    df = features.build_map(panel)
    X, y, _ = features.feature_matrix(df)
    p = score_panel(X)
    for i in range(min(5, len(p))):
        row = X.iloc[i]
        c_all = contributions(row, top=None)
        total = sum(x["contribution"] for x in c_all) + intercept()
        c = c_all[:5]
        assert abs(raw_logit(row) - total) < 1e-9, "attribution does not reconstruct the logit"
        print(f"round {int(df['round_num'].iloc[i]):>2}  p={p[i]:.3f}  "
              f"top={c[0]['name']} {c[0]['contribution']:+.4f}")
    print("attribution check: sum(contrib)+intercept == raw logit (within 1e-9)")


if __name__ == "__main__":
    main()
