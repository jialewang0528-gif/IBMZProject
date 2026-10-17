#!/usr/bin/env python3
"""Calibrate the alert thresholds on real data.

The thresholds (0.15 residual, 2 rounds) were provisional and had never been
measured. This reports, for a grid of thresholds:

  * alerts per map
  * share of maps that produce at least one alert  <- the reviewer's workload
  * the same counts on a NULL panel, where the market values are shuffled across
    the map's rounds

The null is the point. Shuffling keeps every market value but destroys the
round-by-round correspondence with the game state, so any alert it produces is
a false positive by construction. The ratio null/real is therefore an upper
bound on the false-positive share: an upper bound because the market *step*
evidence is left intact rather than shuffled, which gives the null a handicap.

Usage:
    python3 tools/calibrate.py --maps 150
    python3 tools/calibrate.py --maps 150 --thresh 0.10,0.15,0.20,0.25,0.30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import align, features, market, detect  # noqa: E402
from models import score as score_mod  # noqa: E402

MARKET_CONTEXT_MIN = 30


def build_panel(uid: str, odds: pd.DataFrame):
    """Panel with model_p and market_p, plus the map's market series. Or None."""
    try:
        panel = align.attach_market(align.map_panel(uid), odds)
    except Exception:  # noqa: BLE001 - clock failures are counted as unusable
        return None
    if panel["market_p"].notna().sum() < 5:
        return None
    try:
        df = features.build_map(panel)
        X, _, _ = features.feature_matrix(df)
    except Exception:  # noqa: BLE001
        return None
    df["model_p"] = score_mod.score_panel(X)
    window = (panel["t_start_utc"].min() - pd.Timedelta(minutes=MARKET_CONTEXT_MIN),
              panel["t_end_utc"].max() + pd.Timedelta(minutes=5))
    series = market.market_series(uid, odds, window=window)
    return df, series


def count(panel, series, thresh: float) -> int:
    return len(detect.alerts(panel, series, thresh=thresh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=150)
    ap.add_argument("--thresh", default="0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle", type=int, default=3,
                    help="null repeats per map (market values permuted)")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresh.split(",")]

    score_mod.load_models()
    odds = align.io.read_odds()
    from pipeline import io as pio
    el = pio.eligible_maps()
    uids = el["map_uid"].astype(str).tolist()
    np.random.default_rng(args.seed).shuffle(uids)
    uids = uids[:args.maps]

    real = {t: [] for t in thresholds}
    null = {t: [] for t in thresholds}
    used = 0
    rng = np.random.default_rng(args.seed + 1)

    for i, uid in enumerate(uids, 1):
        built = build_panel(uid, odds)
        if built is None:
            continue
        df, series = built
        used += 1
        for t in thresholds:
            real[t].append(count(df, series, t))
        for _ in range(args.shuffle):
            # CIRCULAR SHIFT, not a permutation. Permuting the prices destroys
            # their serial correlation, and the persistence test implicitly
            # relies on a market that moves smoothly; a permuted null therefore
            # manufactures runs and reports a false-positive rate that is too
            # high. Rolling keeps the real price path and only misaligns it with
            # the rounds, which is exactly the null we want.
            shuffled = df.copy()
            vals = shuffled["market_p"].to_numpy(dtype=object).copy()
            mask = ~pd.isna(pd.Series(vals))
            seq = vals[mask]
            if len(seq) > 3:
                vals[mask] = np.roll(seq, int(rng.integers(2, max(3, len(seq) - 1))))
            shuffled["market_p"] = vals
            for t in thresholds:
                null[t].append(count(shuffled, series, t))
        if i % 25 == 0:
            print(f"  ... {i}/{len(uids)} maps, {used} usable", flush=True)

    if not used:
        sys.exit("no usable maps")

    print(f"\nmaps sampled: {used} of {len(uids)}  |  "
          f"null repeats: {args.shuffle}  |  persistence k={args.k}\n")
    header = f"{'thresh':>7} {'alerts/map':>11} {'maps alerting':>14} " \
             f"{'null/map':>9} {'null share':>11}"
    print(header)
    print("-" * len(header))
    rows = []
    for t in thresholds:
        r = np.array(real[t], dtype=float)
        raw = np.array(null[t], dtype=float)
        # null[t] holds `shuffle` passes per map, in map order: average per map
        # first, so the ratio compares like with like
        n = raw.reshape(args.shuffle, -1).mean(axis=0)
        rate = n.sum() / r.sum() if r.sum() else float("nan")
        rows.append({
            "thresh": t, "alerts_per_map": r.mean(),
            "maps_alerting": (r > 0).mean(), "null_per_map": n.mean(),
            "null_share": rate,
        })
        print(f"{t:>7.2f} {r.mean():>11.3f} {100*(r>0).mean():>13.1f}% "
              f"{n.mean():>9.3f} {100*rate:>10.1f}%")

    best = min(rows, key=lambda r: (abs(r["maps_alerting"] - 0.15), r["thresh"]))
    print("\n'A reviewer looks at ~15% of maps' would suggest "
          f"thresh ~ {best['thresh']:.2f} "
          f"({100*best['maps_alerting']:.0f}% of maps alert, "
          f"{best['alerts_per_map']:.2f} alerts/map).")
    print("Null share is an upper bound on false positives: the market-step "
          "evidence is not shuffled.")
    print("\nNOTE: this measures rate, not correctness. It cannot tell you that "
          "a surviving alert is meaningful, only how much of the feed is noise.")


if __name__ == "__main__":
    main()
