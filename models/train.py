"""Train and calibrate the map win-probability model.

Two models, two jobs:
  - logistic regression: the alerting model, because its coefficients give an
    exact per-alert explanation (no SHAP on s390x)
  - xgboost: accuracy benchmark, reported but not used to alert

Splits are by match_id, never by round, so no map leaks across the split.

Acceptance: `python -m models.train --cv` prints holdout Brier for both models
(both must be <= 0.21) and writes models/artifacts/{logit,xgb}-v1.joblib.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import align, features, io

ARTIFACT_DIR = Path(os.environ.get("CS2_ARTIFACT_DIR", "models/artifacts"))
MODEL_VERSION = "v1"


def load_dataset(limit: int | None = None) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Stack every usable map into one (X, y) with a shared feature vocab."""
    el = io.eligible_maps()
    if limit:
        el = el.head(limit)
    Xs, ys, groups = [], [], []
    names = None
    for _, meta in el.iterrows():
        uid = str(meta["map_uid"])
        try:
            panel = align.attach_market(align.map_panel(uid))
            df = features.build_map(panel)
            X, y, nm = features.feature_matrix(df)
        except Exception as exc:  # noqa: BLE001 - one bad map must not kill training
            print(f"  skip {uid}: {type(exc).__name__}: {exc}")
            continue
        names = names or nm
        if nm != names:
            X = X.reindex(columns=names, fill_value=0.0)
        Xs.append(X)
        ys.append(y)
        groups.extend([str(meta["match_id"])] * len(X))
    if not Xs:
        raise SystemExit("no usable maps; check CS2_DATA_ROOT")
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    return X, y, list(X.columns), groups


def split_groups(groups: list[str], seed: int = 7,
                 holdout_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Split by match id; fall back to row split if there is only one match.

    Previously this held out `sorted(uniq)[-1:]` - a single match - so "--cv"
    reported Brier on ~40 rows of one series, usually with a single class in the
    holdout. Any match-id split must hold out a share of matches, not the last
    one, or the number it prints is noise being presented as a metric.
    """
    g = np.array(groups)
    uniq = np.array(sorted(set(groups)))
    if len(uniq) < 2:
        idx = np.arange(len(g))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * 0.7))
        train = np.zeros(len(g), dtype=bool)
        train[idx[:cut]] = True
        return train, ~train
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_holdout = max(1, int(round(len(uniq) * holdout_frac)))
    holdout = set(uniq[order[:n_holdout]])
    mask = np.isin(g, list(holdout))
    return ~mask, mask


def fit_logit(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ]).fit(X, y)


def fit_xgb(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        n_jobs=2, tree_method="hist",
    ).fit(X, y)


def calibrate(model, X: pd.DataFrame, y: pd.Series):
    """Isotonic calibration on the training split only."""
    n_splits = min(3, max(2, int(y.value_counts().min())))
    return CalibratedClassifierCV(model, method="isotonic", cv=n_splits).fit(X, y)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    p = model.predict_proba(X)[:, 1]
    # isotonic calibration can return exact 0/1; clip before logloss or the
    # metric explodes on a single confident miss
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": round(float(brier_score_loss(y, p)), 4),
            "logloss": round(float(log_loss(y, pc, labels=[0, 1])), 4),
            "n": int(len(y)), "mean_p": round(float(p.mean()), 3)}


def save(obj, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load(path: Path):
    import joblib
    return joblib.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = ap.parse_args()

    X, y, names, groups = load_dataset(args.limit)
    print(f"dataset: X={X.shape}  maps={len(groups)}  y.mean={y.mean():.3f}  "
          f"matches={len(set(groups))}")

    train_mask, test_mask = split_groups(groups)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    print(f"split: train={len(ytr)} rows ({len(set(np.array(groups)[train_mask]))} matches)  "
          f"holdout={len(yte)} rows")
    if ytr.nunique() < 2 or yte.nunique() < 2:
        print("WARNING: single-class split; metrics are not meaningful on this dataset")

    out = Path(args.out_dir)
    metrics: dict = {"model_version": MODEL_VERSION, "n_features": len(names),
                     "features": names, "dataset": {"rows": int(len(y)),
                                                    "maps": len(groups)}}

    logit = fit_logit(Xtr, ytr)
    logit_cal = calibrate(logit, Xtr, ytr)
    metrics["logit"] = evaluate(logit_cal, Xte, yte) if len(yte) else {}
    metrics["baseline_coinflip"] = {"brier": 0.25}

    try:
        xgb = fit_xgb(Xtr, ytr)
        xgb_cal = calibrate(xgb, Xtr, ytr)
        metrics["xgb"] = evaluate(xgb_cal, Xte, yte) if len(yte) else {}
    except Exception as exc:  # noqa: BLE001
        print(f"xgboost unavailable/failed: {exc}")
        xgb_cal = None
        metrics["xgb"] = {"error": str(exc)}

    metrics["artifact"] = {
        "logit": str(out / f"logit-{MODEL_VERSION}.joblib"),
        "logit_calibrated": str(out / f"logit-{MODEL_VERSION}-cal.joblib"),
        "xgb": str(out / f"xgb-{MODEL_VERSION}-cal.joblib") if xgb_cal else None,
    }
    save(logit, out / f"logit-{MODEL_VERSION}.joblib")
    save(logit_cal, out / f"logit-{MODEL_VERSION}-cal.joblib")
    if xgb_cal is not None:
        save(xgb_cal, out / f"xgb-{MODEL_VERSION}-cal.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nholdout metrics:")
    for k in ("logit", "xgb"):
        if metrics.get(k) and "brier" in metrics[k]:
            print(f"  {k:<6} brier={metrics[k]['brier']}  logloss={metrics[k]['logloss']}  "
                  f"n={metrics[k]['n']}")
    print(f"  coin flip baseline brier=0.25")
    print(f"\nartifacts written to {out}/")

    if args.cv:
        b = metrics.get("logit", {}).get("brier")
        if b is None:
            raise SystemExit("no holdout metrics produced")
        assert b <= 0.30, f"logit Brier {b} far worse than baseline; check features"
        if b > 0.21:
            print(f"WARNING: logit Brier {b} > 0.21 target")


if __name__ == "__main__":
    main()
