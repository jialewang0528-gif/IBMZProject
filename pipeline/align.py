"""Clock model: one timeline for demo ticks and market prints.

Implements docs/implementation-plan.md §2.3 and §4.2:
  map_start_utc = maps_meta.match_date
  round_start  = map_start + round_detail.start / tickrate
  tickrate is detected, never assumed.

Acceptance: `python -m pipeline.align --validate` must report >= 20 maps with
a resolved tickrate and 100% team1/CT orientation agreement.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from pipeline import io

TICKRATES = (64, 128)
# A CS2 round from freeze-end to round-end is typically 50-110 s. The band is
# deliberately narrow: at 5000 mean ticks, /64 = 78 s (inside) and /128 = 39 s
# (outside), so the band alone usually resolves the tickrate.
PLAUSIBLE_ROUND_S = (45.0, 115.0)
# Cross-check tolerance: map length implied by ticks vs observed live window.
MARKET_SPAN_TOLERANCE = 0.40
MARKET_LOOKBACK_S = 600
# Inter-map break assumption (spec §4.3): used to place map k's start after the
# earlier maps of the same series. maps_meta.match_date is the SERIES start, not
# the map start, so map 2+ must be offset or every later map is misaligned.
BREAK_S = 600


class ClockError(RuntimeError):
    pass


def detect_tickrate(map_uid: str) -> int:
    """Pick the tickrate whose implied round duration is physically plausible.

    If both or neither are plausible the clock is unresolved: we raise rather
    than guess, because every downstream number depends on it.
    """
    det = io.read_table("round_detail")
    rows = det[det["map_uid"].astype(str) == str(map_uid)]
    if rows.empty:
        raise ClockError(f"no round_detail rows for map_uid {map_uid}")
    ticks = pd.to_numeric(rows["duration_ticks"], errors="coerce").dropna()
    if ticks.empty:
        raise ClockError(f"duration_ticks missing for {map_uid}")
    mean_ticks = float(ticks.mean())

    plausible = [tr for tr in TICKRATES
                 if PLAUSIBLE_ROUND_S[0] <= mean_ticks / tr <= PLAUSIBLE_ROUND_S[1]]
    if len(plausible) == 1:
        return plausible[0]

    # Ambiguous by duration alone: cross-check against the market's live window,
    # an independent clock. Only accept a clear winner.
    if len(plausible) > 1:
        observed = _market_span_s(map_uid)
        if observed and observed > 60:
            total_ticks = float(pd.to_numeric(rows["official_end"], errors="coerce").max())
            scores = {tr: abs(total_ticks / tr - observed) / observed for tr in plausible}
            best = min(scores, key=scores.get)
            if scores[best] <= MARKET_SPAN_TOLERANCE:
                return best
    raise ClockError(
        f"tickrate ambiguous for {map_uid}: mean duration_ticks={mean_ticks:.0f} "
        f"implies {[round(mean_ticks / tr, 1) for tr in TICKRATES]}s for {TICKRATES}"
    )


def _market_span_s(map_uid: str) -> float | None:
    """Seconds between map start and the last live market print, or None."""
    try:
        meta = io.get_map(map_uid)
        odds = io.read_odds()
    except (KeyError, FileNotFoundError):
        return None
    sel = odds[odds["selection"].astype(str).str.lower() == "team1"]
    sel = sel[sel["market"].astype(str).str.lower().isin({"map_winner", "map winner"})]
    by_uid = sel["map_uid"].astype(str) == str(map_uid)
    by_idx = (sel["match_id"] == meta["match_id"]) & (sel["map_index"] == meta["map_index"])
    series = sel[by_uid | by_idx]
    if series.empty:
        return None
    live = series[series["is_live"]]
    if live.empty:
        return None
    return (live["ts_utc"].max() - map_start_utc(map_uid)).total_seconds()


def _map_duration_s(map_index_meta: pd.Series, tickrate: int = 64) -> float:
    """Duration of one map in seconds, from its last official_end tick."""
    det = io.read_table("round_detail")
    rows = det[det["map_uid"].astype(str) == str(map_index_meta["map_uid"])]
    if rows.empty:
        return 0.0
    last_tick = pd.to_numeric(rows["official_end"], errors="coerce").max()
    if pd.isna(last_tick):
        return 0.0
    return float(last_tick) / tickrate


def map_start_utc(map_uid: str) -> pd.Timestamp:
    """Wall-clock start of THIS map.

    maps_meta.match_date is the series start (all maps of a match share it), so
    map k's start = series start + the durations and breaks of maps 1..k-1.
    """
    meta = io.get_map(map_uid)
    series_start = meta["map_start_utc"]
    map_index = int(meta["map_index"])
    if map_index <= 1:
        return series_start
    meta_all = io.maps()
    prior = meta_all[(meta_all["match_id"] == meta["match_id"])
                     & (meta_all["map_index"] < map_index)]
    offset = 0.0
    for _, row in prior.iterrows():
        offset += _map_duration_s(row) + BREAK_S
    return series_start + pd.Timedelta(seconds=offset)


def map_panel(map_uid: str, tickrate: int | None = None) -> pd.DataFrame:
    """One row per round with wall-clock start/end and the resolution floor."""
    meta = io.get_map(map_uid)
    tickrate = tickrate or detect_tickrate(map_uid)
    det = io.read_table("round_detail")
    det = det[det["map_uid"].astype(str) == str(map_uid)].copy()
    if det.empty:
        raise ClockError(f"no round_detail rows for {map_uid}")

    start = map_start_utc(map_uid)
    panel = pd.DataFrame({
        "map_uid": str(map_uid),
        "match_id": meta["match_id"],
        "map_index": meta["map_index"],
        "round_num": pd.to_numeric(det["round_num"], errors="coerce").astype(int),
        "tickrate": tickrate,
        "map_start_utc": start,
        "t_start_utc": start + pd.to_timedelta(
            pd.to_numeric(det["start"], errors="coerce") / tickrate, unit="s"),
        "t_end_utc": start + pd.to_timedelta(
            pd.to_numeric(det["end"], errors="coerce") / tickrate, unit="s"),
    })
    panel = panel.sort_values("round_num").reset_index(drop=True)
    if not panel["t_start_utc"].is_monotonic_increasing:
        raise ClockError(f"non-monotonic round starts for {map_uid}")
    if panel["round_num"].duplicated().any():
        raise ClockError(f"duplicate round_num for {map_uid}")
    panel["floor_s"] = 60
    return panel


def attach_market(panel: pd.DataFrame, odds: pd.DataFrame | None = None,
                  lookback_s: int = MARKET_LOOKBACK_S) -> pd.DataFrame:
    """As-of join: for each round, the last market print <= round start.

    Only prints within `lookback_s` of the round start are considered, so a
    stale price from before the match cannot masquerade as a live signal.
    """
    if panel.empty:
        return panel
    odds = io.read_odds() if odds is None else odds
    map_uid = str(panel["map_uid"].iloc[0])
    match_id = panel["match_id"].iloc[0]
    map_index = panel["map_index"].iloc[0]

    sel = odds[odds["selection"].astype(str).str.lower() == "team1"]
    sel = sel[sel["market"].astype(str).str.lower().isin({"map_winner", "map winner"})]
    by_uid = sel["map_uid"].astype(str) == map_uid
    by_idx = (sel["match_id"] == match_id) & (sel["map_index"] == map_index)
    series = sel[by_uid | by_idx].copy()
    if series.empty:
        panel["market_p"] = pd.NA
        panel["market_ts_utc"] = pd.NaT
        panel["market_ts_delta_s"] = pd.NA
        return panel

    series = (series[["ts_utc", "fair_prob"]]
              .rename(columns={"ts_utc": "market_ts_utc", "fair_prob": "market_p"})
              .dropna()
              .sort_values("market_ts_utc"))

    left = panel[["round_num", "t_start_utc"]].sort_values("t_start_utc")
    merged = pd.merge_asof(
        left, series, left_on="t_start_utc", right_on="market_ts_utc",
        direction="backward", tolerance=pd.Timedelta(seconds=lookback_s),
    )[["round_num", "market_ts_utc", "market_p"]]
    out = panel.drop(columns=[c for c in ("market_p", "market_ts_utc",
                                          "market_ts_delta_s") if c in panel.columns])
    out = out.merge(merged, on="round_num", how="left")
    out["market_ts_delta_s"] = (
        (out["market_ts_utc"] - out["t_start_utc"]).dt.total_seconds()
    )
    return out


def validate(n_maps: int = 20) -> pd.DataFrame:
    """T1 acceptance: tickrate resolved and team1/CT orientation agrees."""
    el = io.eligible_maps().head(n_maps)
    rt = io.read_table("round_table")
    rows = []
    for _, meta in el.iterrows():
        uid = str(meta["map_uid"])
        try:
            tr = detect_tickrate(uid)
            panel = map_panel(uid, tr)
        except ClockError as exc:
            rows.append({"map_uid": uid, "tickrate": None, "rounds": 0,
                         "orientation_ok": False, "note": str(exc)[:60]})
            continue
        r = rt[rt["map_uid"].astype(str) == uid]
        ct = r["ct_team_name_resolved"].astype(str)
        # team1 must appear as exactly one side, consistently
        def norm(x) -> str:
            return " ".join(str(x).split()).casefold()

        team1 = norm(meta["team1"])
        sides = set()
        for _, row in r.iterrows():
            if norm(row["ct_team_name_resolved"]) == team1:
                sides.add("ct")
            elif norm(row["t_team_name_resolved"]) == team1:
                sides.add("t")
            else:
                sides.add("unknown")
        # market coverage: catches a wrong map-start anchor (e.g. using the
        # series start for map 2, which shifts every round past the lookback)
        try:
            cov = float(attach_market(panel)["market_p"].notna().mean())
        except Exception:  # noqa: BLE001
            cov = 0.0
        rows.append({
            "map_uid": uid, "tickrate": tr, "rounds": len(panel),
            "orientation_ok": sides <= {"ct", "t"} and bool(sides),
            "market_cov": round(cov, 2),
            "note": "" if sides <= {"ct", "t"} else f"team1 not found in sides: {sides}",
        })
    out = pd.DataFrame(rows)
    ok = out[out["tickrate"].notna()]
    print(f"maps checked: {len(out)}  tickrate resolved: {len(ok)}  "
          f"orientation ok: {int(out['orientation_ok'].sum())}  "
          f"market coverage median: {out['market_cov'].median():.2f}")
    print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-uid")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n-maps", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.validate:
        validate(args.n_maps)
        return
    if not args.map_uid:
        ap.error("need --map-uid or --validate")
    panel = attach_market(map_panel(args.map_uid))
    cols = ["round_num", "t_start_utc", "market_p", "market_ts_delta_s"]
    print(panel[cols].to_string(index=False))
    if args.out:
        panel.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
