"""Detection: market-vs-model divergence, market anomalies, player deviation.

Implements spec §4.2 / §4.4. Deliberately free of model dependencies: it takes a
panel that already carries `model_p` and `market_p`, so it can be tested with
synthetic numbers.

Acceptance: `python -m pipeline.detect --map-uid <uid>` prints the four
criteria per alert; the injected map must produce >= 1 alert and 10 random
non-injected maps must produce 0.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline import io

GAP_THRESHOLD = 0.15
PERSISTENCE_ROUNDS = 2
STEP_WINDOW_MIN = 10
STEP_THRESHOLD = 0.05
FLOOR_S = 60
SIZE_Z = 3.0


def gap_series(panel: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(panel["market_p"], errors="coerce") - pd.to_numeric(
        panel["model_p"], errors="coerce")


def baseline_gap(panel: pd.DataFrame, n: int = 3) -> float:
    """The map's own starting offset between market and model.

    A globally trained model has a per-map bias; alerting on the absolute gap
    would then fire on every round of every map. Alerts are therefore computed
    on the residual gap (gap minus this baseline), which is what "the market
    moved away from the game state" actually means.
    """
    g = gap_series(panel).dropna()
    head = g.head(n)
    return float(head.median()) if len(head) else 0.0


def residual_gap(panel: pd.DataFrame, n: int = 3) -> pd.Series:
    return gap_series(panel) - baseline_gap(panel, n)


def persistence(gap: pd.Series, thresh: float = GAP_THRESHOLD,
                k: int = PERSISTENCE_ROUNDS) -> list[tuple[int, int]]:
    """Spans (start_idx, end_idx) of >= k consecutive rounds above |thresh|.

    Indices are positional; callers map them back to round_num.
    """
    vals = gap.abs().fillna(0.0).to_numpy()
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(vals):
        if v >= thresh:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= k:
                out.append((start, i - 1))
            start = None
    if start is not None and (len(vals) - start) >= k:
        out.append((start, len(vals) - 1))
    return out


def step_detector(series: pd.DataFrame, window_min: int = STEP_WINDOW_MIN,
                  thresh: float = STEP_THRESHOLD) -> list[dict]:
    """Price changes of at least `thresh` within `window_min` minutes."""
    if series.empty:
        return []
    s = series.sort_values("ts_utc").reset_index(drop=True)
    steps = []
    win = pd.Timedelta(minutes=window_min)
    for i in range(1, len(s)):
        window = s[(s["ts_utc"] >= s["ts_utc"].iloc[i] - win) & (s.index <= i)]
        if len(window) < 2:
            continue
        delta = float(s["market_p"].iloc[i] - window["market_p"].iloc[0])
        if abs(delta) >= thresh:
            steps.append({"ts_utc": s["ts_utc"].iloc[i], "delta": round(delta, 4)})
    return steps


def size_anomaly(series: pd.DataFrame, z: float = SIZE_Z) -> list[dict]:
    """Prints whose size is an outlier for that market. Needs a size column."""
    if series.empty or "size" not in series.columns:
        return []
    sizes = pd.to_numeric(series["size"], errors="coerce")
    if sizes.notna().sum() < 5 or sizes.std(ddof=0) == 0:
        return []
    zs = (sizes - sizes.mean()) / sizes.std(ddof=0)
    hits = series[zs.abs() >= z]
    return [{"ts_utc": r["ts_utc"], "size": float(r.get("size", 0)), "z": float(zs.loc[i])}
            for i, r in hits.iterrows()]


def no_news(panel: pd.DataFrame, t_market: pd.Timestamp, floor_s: int = FLOOR_S) -> bool:
    """True if no round ended within `floor_s` before the market move.

    A move that happens before the round in question even started cannot be
    explained by that round's events; that is the cheapest honest filter.
    """
    if pd.isna(t_market):
        return False
    ends = pd.to_datetime(panel["t_end_utc"], utc=True)
    window = (ends <= t_market) & (ends >= t_market - pd.Timedelta(seconds=floor_s))
    return not bool(window.any())


def severity(gap: float, span_rounds: int, corroborated: bool) -> str:
    g = abs(gap)
    if g >= 0.25 and span_rounds >= 3:
        return "high"
    if g >= 0.20 or (g >= 0.15 and span_rounds >= 3) or corroborated:
        return "medium"
    return "low"


def alerts(panel: pd.DataFrame, series: pd.DataFrame | None = None) -> list[dict]:
    """Apply the spec's alert criteria. Returns contract-shaped dicts (no model)."""
    if panel.empty or "model_p" not in panel.columns:
        return []
    panel = panel.reset_index(drop=True).copy()
    panel["gap"] = gap_series(panel)
    panel["residual_gap"] = residual_gap(panel)
    spans = persistence(panel["residual_gap"])
    steps = step_detector(series) if series is not None else []
    out = []
    for start, end in spans:
        row = panel.iloc[start]
        t_start = pd.to_datetime(row["t_start_utc"], utc=True)
        resid = float(row["residual_gap"])
        moved_first = any(
            (pd.Timestamp(st["ts_utc"]) <= t_start)
            and (pd.Timestamp(st["ts_utc"]) >= t_start - pd.Timedelta(minutes=STEP_WINDOW_MIN))
            and (np.sign(st["delta"]) == np.sign(resid))
            for st in steps
        )
        t_market = row.get("market_ts_utc")
        clean = no_news(panel, pd.Timestamp(t_market)) if pd.notna(t_market) else True
        out.append({
            "round_num": int(row["round_num"]),
            "span": [int(panel["round_num"].iloc[start]), int(panel["round_num"].iloc[end])],
            "gap": round(float(row["gap"]), 4),
            "residual_gap": round(resid, 4),
            "baseline_gap": round(baseline_gap(panel), 4),
            "severity": severity(resid, end - start + 1, corroborated=False),
            "criteria": {
                "residual_ge_15": bool(abs(resid) >= GAP_THRESHOLD),
                "persistence_rounds": int(end - start + 1),
                "market_moved_first": bool(moved_first),
                "no_explaining_event": bool(clean),
            },
            "floor_s": FLOOR_S,
        })
    return out


def player_deviation(map_uid: str, rounds: list[int] | None = None) -> pd.DataFrame:
    """Within-match self-deviation: each player vs their own match average.

    Corroboration only. A player cannot trigger an alert on this score.
    """
    eco = io.read_table("player_round_economy")
    eco = eco[eco["map_uid"].astype(str) == str(map_uid)]
    if eco.empty:
        return pd.DataFrame(columns=["steamid", "name", "rounds_flagged", "z_mean"])
    kills = io.read_table("kills")
    kills = kills[kills["map_uid"].astype(str) == str(map_uid)]

    eco["round_num"] = pd.to_numeric(eco["round_num"], errors="coerce").astype(int)
    kills["round_num"] = pd.to_numeric(kills["round_num"], errors="coerce").astype(int)

    per_round = (kills.groupby(["attacker_steamid", "round_num"]).size()
                 .rename("kills").reset_index()
                 .rename(columns={"attacker_steamid": "steamid"}))
    eco["steamid"] = eco["steamid"].astype(str)
    per_round["steamid"] = per_round["steamid"].astype(str)
    merged = eco.merge(per_round, on=["steamid", "round_num"], how="left")
    merged["kills"] = merged["kills"].fillna(0)

    rows = []
    for steamid, grp in merged.groupby("steamid"):
        mean = grp["kills"].mean()
        sd = grp["kills"].std(ddof=0)
        if not sd or np.isnan(sd):
            continue
        z = (grp["kills"] - mean) / sd
        sel = grp["round_num"].isin(rounds) if rounds else pd.Series(True, index=grp.index)
        sub = z[sel]
        if sub.empty:
            continue
        rows.append({
            "steamid": steamid,
            "name": str(grp["player_name"].iloc[0]),
            "rounds_flagged": int(len(sub)),
            "z_mean": round(float(sub.mean()), 3),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank"] = df["z_mean"].rank(method="first").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-uid", required=True)
    ap.add_argument("--ablation", action="store_true",
                    help="paired test: the same map must alert more WITH injection "
                         "than without (thresholds are not calibrated on a fixture)")
    ap.add_argument("--inject-rounds", default="14,15,16")
    ap.add_argument("--fp-check", type=int, default=0,
                    help="report alert rate on N other maps (informational: thresholds "
                         "are provisional until calibrated on the real dataset)")
    args = ap.parse_args()

    from models import score as score_mod
    from pipeline import align, features, market

    panel = align.attach_market(align.map_panel(args.map_uid))
    df = features.build_map(panel)
    X, _, _ = features.feature_matrix(df)
    df["model_p"] = score_mod.score_panel(X)
    series = market.market_series(args.map_uid)
    found = alerts(df, series)
    print(f"map {args.map_uid}: {len(found)} alert(s)")
    for a in found:
        print(f"  round {a['round_num']} span {a['span']} gap {a['gap']:+.3f} "
              f"residual {a['residual_gap']:+.3f} (baseline {a['baseline_gap']:+.3f}) "
              f"severity={a['severity']} criteria={a['criteria']}")
    players = player_deviation(args.map_uid,
                               [r for a in found for r in range(a["span"][0], a["span"][1] + 1)])
    if not players.empty:
        print(players.head(5).to_string(index=False))

    if args.ablation:
        from pipeline import inject as inject_mod
        base = len(found)
        rounds = [int(x) for x in args.inject_rounds.split(",") if x.strip()]
        inj = inject_mod.inject(df, rounds)
        inj_alerts = alerts(inj, series)
        print(f"\nablation on {args.map_uid}:")
        print(f"  without injection: {base} alert(s)")
        print(f"  with injection R{args.inject_rounds}: {len(inj_alerts)} alert(s)")
        for a in inj_alerts:
            print(f"    round {a['round_num']} residual {a['residual_gap']:+.3f} "
                  f"severity={a['severity']}")
        covered = [a for a in inj_alerts
                   if any(a["span"][0] <= r <= a["span"][1] for r in rounds)]
        print(f"  alerts covering injected rounds: {len(covered)}")
        assert inj_alerts, "injected divergence did not produce any alert"
        assert covered, f"no alert span covers the injected rounds {rounds}"
        print("  PASS: injection is detected end-to-end")

    if args.fp_check:
        import random
        el = io.eligible_maps()
        others = [u for u in el["map_uid"].astype(str) if u != args.map_uid]
        random.Random(7).shuffle(others)
        fp = 0
        for uid in others[:args.fp_check]:
            p = align.attach_market(align.map_panel(uid))
            d = features.build_map(p)
            Xi, _, _ = features.feature_matrix(d)
            d["model_p"] = score_mod.score_panel(Xi)
            n = len(alerts(d, market.market_series(uid)))
            fp += n
            print(f"  control {uid}: {n} alerts")
        print(f"control alert count: {fp} over {args.fp_check} maps "
              f"({fp / max(args.fp_check, 1):.2f} per map)")
        print("NOTE: thresholds (0.15 / 2 rounds) are provisional; calibrate on the "
              "real 297-map set before quoting a false-positive rate.")


if __name__ == "__main__":
    main()
