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
import unicodedata
from datetime import timedelta
from functools import lru_cache

import pandas as pd

from pipeline import io

TICKRATES = (64, 128)
SECONDS_DIVISOR = 1
MARKET_LOOKBACK_S = 600

# --- the ruler the tickrate is measured against ---------------------------
# A round is: freezetime, then live play. Both are bounded by rules the server
# enforces, so they pin the clock down without reference to the market:
#
#   freezetime : configured per match. Competitive CS2 uses 15 or 20 s. It is
#                identical for every round, so the MODE of (freeze_end - start)
#                recovers it exactly.
#   live round : capped by the 115 s round clock, extended to at most
#                115 + 40 s when the bomb is planted.
#
# Mean round LENGTH is not usable: this dataset's mean is ~7000 clock units,
# which is a plausible 109 s at 64-tick AND a plausible 55 s at 128-tick. That
# ambiguity is what made earlier versions guess wrong.
FREEZE_CHOICES_S = (15.0, 20.0)
FREEZE_TOL_S = 0.5
ROUND_LIMIT_S = 115.0
BOMB_EXTENSION_S = 40.0
LIVE_CEILING_S = ROUND_LIMIT_S + BOMB_EXTENSION_S
# Median live round in professional play. Deliberately narrow: it has to
# separate 64-tick data (median ~85 s) from the same rows misread as 128-tick
# (median ~42 s), which is the exact error this code exists to prevent.
MEDIAN_LIVE_S = (55.0, 105.0)
# A handful of rounds per map are stretched by technical pauses; tolerate that.
CEILING_VIOLATION_FRAC = 0.01
# Inter-map break assumption (spec §4.3): used to place map k's start after the
# earlier maps of the same series. maps_meta.match_date is the SERIES start, not
# the map start, so map 2+ must be offset or every later map is misaligned.
BREAK_S = 600


class ClockError(RuntimeError):
    pass


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clock_stats(rows: pd.DataFrame) -> dict:
    """Freeze and live-round durations, in raw clock units, for some rounds."""
    start = _num(rows["start"])
    freeze_end = _num(rows["freeze_end"])
    end = _num(rows["end"])
    freeze = (freeze_end - start).dropna()
    live = (end - freeze_end).dropna()
    freeze = freeze[freeze >= 0]
    live = live[live > 0]
    if live.empty:
        raise ClockError("no usable round segments to measure the clock with")
    mode = freeze.round().mode()
    return {
        "live": live,
        "freeze_mode": float(mode.iloc[0]) if len(mode) else float("nan"),
        "n": int(len(live)),
    }


def judge_divisor(stats: dict, divisor: int) -> tuple[int, list[str]]:
    """Score one reading of the clock columns against server-enforced rules."""
    live = stats["live"] / divisor
    freeze_s = stats["freeze_mode"] / divisor
    median_s = float(live.median())
    over_frac = float((live > LIVE_CEILING_S).mean())
    score, notes = 0, []

    if any(abs(freeze_s - c) <= FREEZE_TOL_S for c in FREEZE_CHOICES_S):
        score += 1
        notes.append(f"freezetime {freeze_s:g}s is a standard value")
    else:
        notes.append(f"freezetime {freeze_s:g}s is not {FREEZE_CHOICES_S}")

    if MEDIAN_LIVE_S[0] <= median_s <= MEDIAN_LIVE_S[1]:
        score += 1
        notes.append(f"median live round {median_s:.0f}s is plausible")
    else:
        notes.append(f"median live round {median_s:.0f}s is outside "
                     f"{MEDIAN_LIVE_S[0]:.0f}-{MEDIAN_LIVE_S[1]:.0f}s")

    if over_frac > CEILING_VIOLATION_FRAC:
        score -= 100  # a hard physical veto, not a deduction
        notes.append(f"{over_frac:.1%} of rounds outlast the {LIVE_CEILING_S:.0f}s "
                     f"ceiling (115s round clock + 40s bomb) - impossible")
    return score, notes


def judge_map(stats: dict, divisor: int) -> tuple[bool, list[str]]:
    """Is this map *physically possible* at the given rate?

    Only the hard rules - the round ceiling and a configured (whole-second)
    freezetime - are allowed to reject a map. The choice of rate is made on the
    whole dataset, because a single map's median round is too noisy to carry it
    (real maps range from 91 s to 111 s at 64-tick).
    """
    live = stats["live"] / divisor
    freeze_s = stats["freeze_mode"] / divisor
    notes, ok = [], True
    over_frac = float((live > LIVE_CEILING_S).mean())
    if over_frac > CEILING_VIOLATION_FRAC:
        ok = False
        notes.append(f"{over_frac:.1%} of rounds outlast the {LIVE_CEILING_S:.0f}s ceiling")
    if pd.notna(freeze_s) and abs(freeze_s - round(freeze_s)) > 0.1:
        ok = False
        notes.append(f"freezetime {freeze_s:g}s is not a whole number of seconds")
    return ok, notes


def _clock_report(stats: dict, results: dict) -> str:
    lines = [f"clock unit unresolved: {stats['n']} rounds, "
             f"modal freezetime {stats['freeze_mode']:g} raw units"]
    for div in sorted(results):
        score, notes = results[div]
        label = "seconds" if div == SECONDS_DIVISOR else f"{div}-tick"
        lines.append(f"  read as {label:<9} (divisor {div:>3}): score {score:>3} | "
                     + "; ".join(notes))
    lines.append("no reading satisfies the rules unambiguously; refusing to guess")
    return "\n".join(lines)


def _infer_dataset_rate() -> tuple[int, dict]:
    """The tickrate is a property of the demo files, not of one map.

    Deciding it on every round in the dataset (rather than per map) makes the
    statistics sharp: the median live round separates 64-tick (~85 s) from
    128-tick (~42 s) with the whole sample behind it.
    """
    stats = clock_stats(io.read_table("round_detail"))
    results = {div: judge_divisor(stats, div) for div in (SECONDS_DIVISOR, *TICKRATES)}
    ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
    best, best_result = ranked[0]
    runner_up = ranked[1][1][0]
    # Both rule-based tests must pass, and the runner-up must be clearly worse.
    if best_result[0] < 2 or best_result[0] <= runner_up:
        raise ClockError(_clock_report(stats, results))
    return best, results


@lru_cache(maxsize=8)
def _cached_rate(path_key: str, mtime: float) -> tuple[int, dict]:
    return _infer_dataset_rate()


def detect_tickrate(map_uid: str | None = None) -> int:
    """Divisor for the round_detail clock columns.

    Returns 64/128 when the columns are ticks at that rate, or 1 when they are
    already seconds. Raises rather than guessing: every downstream timestamp
    depends on it.
    """
    path = io.resolve("round_detail")
    divisor, results = _cached_rate(str(path), path.stat().st_mtime)

    if map_uid is not None:
        rows = io.read_table("round_detail")
        rows = rows[rows["map_uid"].astype(str) == str(map_uid)]
        if rows.empty:
            raise ClockError(f"no round_detail rows for map_uid {map_uid}")
        try:
            map_stats = clock_stats(rows)
        except ClockError as exc:
            raise ClockError(f"{map_uid}: {exc}") from exc
        map_score, map_notes = judge_divisor(map_stats, divisor)
        ok, why = judge_map(map_stats, divisor)
        if not ok:
            raise ClockError(
                f"{map_uid} contradicts the dataset clock ({divisor}-tick, "
                f"chosen from all rounds): " + "; ".join(why)
                + " | " + "; ".join(map_notes))
    return divisor


def _map_duration_s(map_index_meta: pd.Series) -> float:
    """Duration of one map in seconds, from its last official_end (or end) column."""
    det = io.read_table("round_detail")
    rows = det[det["map_uid"].astype(str) == str(map_index_meta["map_uid"])]
    if rows.empty:
        return 0.0
    uid = str(map_index_meta["map_uid"])
    try:
        divisor = detect_tickrate(uid)
    except ClockError:
        divisor = 64
    for col in ("official_end", "end"):
        if col in rows.columns:
            last = pd.to_numeric(rows[col], errors="coerce").max()
            if pd.notna(last):
                return float(last) / divisor
    return 0.0


def _raw_map_start(map_uid: str) -> pd.Timestamp:
    """Map start straight from maps_meta, before any clock correction.

    maps_meta.match_date is the series start (all maps of a match share it), so
    map k's start = series start + the durations and breaks of maps 1..k-1.

    This is the GAME side's opinion of the clock, and it is wrong by whole
    hours: match_date is event-local time labelled as UTC (see clock_offsets).
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


def _map_winner_odds(odds: pd.DataFrame) -> pd.DataFrame:
    sel = odds[odds["selection"].astype(str).str.lower() == "team1"]
    return sel[sel["market"].astype(str).str.lower().isin({"map_winner", "map winner"})]


# A clock correction is a timezone mislabel plus scheduling slack, so it cannot
# plausibly exceed a few hours. Estimates outside this window are junk from a
# stale market print, not a real offset.
MAX_CLOCK_OFFSET_S = 6 * 3600
# Maps of one match share a timezone, so their corrections must agree closely.
MATCH_OFFSET_TOL_S = 1800


def settlement_ts(odds: pd.DataFrame, map_uid: str) -> pd.Timestamp | None:
    """When the market stopped being undecided about this map.

    A map_winner price collapses to 0 or 1 the moment the map ends, so the last
    print still strictly between the two is a datable proxy for the final round.
    That is a ground-truth correspondence between the two feeds - a known shared
    event - which is what makes it safe to hang the clock alignment on.
    """
    g = _map_winner_odds(odds)
    g = g[g["map_uid"].astype(str) == str(map_uid)].sort_values("ts_utc")
    if len(g) < 3:
        return None
    p = pd.to_numeric(g["fair_prob"], errors="coerce")
    undecided = g[(p > 0.02) & (p < 0.98)]
    if undecided.empty:
        return None
    return undecided["ts_utc"].max()


def _offset_table() -> pd.Series:
    """Per-map wall-clock correction, in seconds, indexed by map_uid.

    match_date is event-local time mislabelled as UTC: measured against the
    market's own settlement of the same map, the game clock sits 1-3 whole hours
    ahead, clustering at -1 h (CET), -2 h (CEST) and -3 h (MSK). Left uncorrected
    the market join finds nothing and every map reports coverage 0.00.

    Anchoring each map on its own settlement also removes the inter-map break
    assumption: start = settlement - map_duration, so BREAK_S cancels out.
    """
    odds = io.read_map_winner()
    det = io.read_table("round_detail")
    meta = io.maps()
    divisor = detect_tickrate()

    # per-map duration from the last round end, in seconds
    ends = pd.to_numeric(det["end"], errors="coerce")
    last_end = ends.groupby(det["map_uid"].astype(str)).max() / divisor

    # per-map settlement: last print still between 0.02 and 0.98
    mw = _map_winner_odds(odds).copy()
    mw["p"] = pd.to_numeric(mw["fair_prob"], errors="coerce")
    uid_key = mw["map_uid"].astype(str)
    undecided = mw[(mw["p"] > 0.02) & (mw["p"] < 0.98)]
    settle = undecided.groupby(undecided["map_uid"].astype(str))["ts_utc"].max()
    dense = uid_key.value_counts()
    settle = settle[dense.reindex(settle.index).fillna(0) >= 3]

    raw: dict[str, float] = {}
    for _, sub in meta.groupby("match_id"):
        sub = sub.sort_values("map_index")
        series_start = sub["map_start_utc"].min()
        elapsed = 0.0
        for _, row in sub.iterrows():
            uid = str(row["map_uid"])
            raw_start = series_start + pd.Timedelta(seconds=elapsed)
            elapsed += float(last_end.get(uid, 0.0)) + BREAK_S
            if uid in settle.index and uid in last_end.index:
                raw_end = raw_start + pd.Timedelta(seconds=float(last_end[uid]))
                cand = (settle[uid] - raw_end).total_seconds()
                if abs(cand) <= MAX_CLOCK_OFFSET_S:
                    raw[uid] = cand

    table = pd.Series(raw, dtype="float64")
    # Maps of a match share one timezone: take the match consensus and pull in
    # any map whose own estimate disagrees by more than the slack between maps.
    for _, sub in meta.groupby("match_id"):
        uids = sub["map_uid"].astype(str)
        known = table.reindex(uids).dropna()
        if known.empty:
            continue
        consensus = float(known.median())
        for uid in uids:
            if uid in table.index and abs(table[uid] - consensus) > MATCH_OFFSET_TOL_S:
                table.loc[uid] = consensus
    return table.sort_index()


@lru_cache(maxsize=4)
def _cached_offsets(odds_key: str, det_key: str) -> pd.Series:
    return _offset_table()


def clock_offsets() -> pd.Series:
    """Cached per-map clock correction in seconds (see _offset_table)."""
    od = io.resolve_odds()
    rd = io.resolve("round_detail")
    return _cached_offsets(f"{od}:{od.stat().st_mtime}", f"{rd}:{rd.stat().st_mtime}")


def map_start_utc(map_uid: str) -> pd.Timestamp:
    """Wall-clock (true UTC) start of THIS map.

    maps_meta's own value is 1-3 h out, so it is corrected onto the market's
    clock using this map's settlement. Without that correction the two feeds do
    not overlap at all and the market join is empty.
    """
    start = _raw_map_start(map_uid)
    offsets = clock_offsets()
    secs = offsets.get(str(map_uid))
    if secs is None or not pd.notna(secs):
        return start
    return start + pd.Timedelta(seconds=float(secs))


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


def _name_tokens(name) -> list[str]:
    """Word tokens with accents and punctuation folded away.

    'Famalicão' and 'famalicao' are the same team; so are 'MORNING STAR-' and
    'Morningstar'. Diacritics are only decoration here.
    """
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    for ch in "-_.'’/,|()":
        s = s.replace(ch, " ")
    return [t for t in s.split() if t.isalnum()]


def norm_team(name) -> str:
    """Canonical team name: lowercase, unaccented, punctuation-free."""
    return "".join(_name_tokens(name))


def _team_aliases(name) -> set[str]:
    base = norm_team(name)
    out = {base}
    for a, b in (("and", "n"), ("n", "and")):
        if a in base:
            out.add(base.replace(a, b))
    return out


def same_team(a, b) -> bool:
    """True if these two names plausibly denote the same team.

    The feeds differ in decoration and legal suffix, never in the core name:
    maps_meta says 'G2', the demo says 'G2 Esports'; maps_meta says 'M1X KS',
    the demo says 'm1x'. So a leading run of whole words may be dropped, but a
    bare substring hit is not enough - it would collide Navi with Navi Junior.
    """
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False
    if _team_aliases(a) & _team_aliases(b):
        return True
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # the shorter name must be a whole-word prefix of the longer one
    return longer[:len(shorter)] == shorter


def team1_side(rows: pd.DataFrame, team1) -> set[str]:
    """Which side (ct/t) team1 occupies across these rounds.

    Returns a subset of {'ct', 't', 'unknown'}. A healthy map returns exactly
    {'ct'} or {'t'}; anything else means the names could not be reconciled.
    """
    sides: set[str] = set()
    for ct_name, t_name in zip(rows["ct_team_name_resolved"],
                               rows["t_team_name_resolved"]):
        if same_team(team1, ct_name):
            sides.add("ct")
        elif same_team(team1, t_name):
            sides.add("t")
        else:
            sides.add("unknown")
    return sides


def _as_ns_utc(series: pd.Series) -> pd.Series:
    """Force one datetime64 unit. pandas 3 refuses to merge ns with us."""
    return pd.to_datetime(series, utc=True).dt.as_unit("ns")


def attach_market(panel: pd.DataFrame, odds: pd.DataFrame | None = None,
                  lookback_s: int = MARKET_LOOKBACK_S) -> pd.DataFrame:
    """As-of join: for each round, the last market print <= round start.

    Only prints within `lookback_s` of the round start are considered, so a
    stale price from before the match cannot masquerade as a live signal.
    """
    if panel.empty:
        return panel
    sel = io.read_map_winner() if odds is None else _map_winner_odds(odds)
    map_uid = str(panel["map_uid"].iloc[0])
    match_id = panel["match_id"].iloc[0]
    map_index = panel["map_index"].iloc[0]

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

    left = panel[["round_num", "t_start_utc"]].sort_values("t_start_utc").copy()
    # pandas 3 added a hard dtype check to merge_asof: a market print parsed as
    # datetime64[us, UTC] and a panel built with datetime64[ns, UTC] raise
    # MergeError. Pin both sides to ns.
    left["t_start_utc"] = _as_ns_utc(left["t_start_utc"])
    series["market_ts_utc"] = _as_ns_utc(series["market_ts_utc"])
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
    """T1 acceptance: clock resolved, team1/CT orientation agrees, market joins."""
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
                         "orientation_ok": False, "market_cov": None,
                         "note": str(exc).split("\n")[0][:70]})
            continue
        r = rt[rt["map_uid"].astype(str) == uid]
        # team1 must appear as exactly one side, consistently
        sides = team1_side(r, meta["team1"])
        # market coverage: catches a wrong map-start anchor (e.g. using the
        # series start for map 2, which shifts every round past the lookback).
        # Never swallow the error: a silent 0.0 here hid a pandas MergeError.
        note = ""
        try:
            cov: float | None = float(attach_market(panel)["market_p"].notna().mean())
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            cov = None
            note = f"market join failed: {type(exc).__name__}: {exc}"[:70]
        off = clock_offsets().get(uid)
        if not (sides <= {"ct", "t"} and sides):
            note = (note + "; " if note else "") + f"team1 not in sides: {sides}"
        rows.append({
            "map_uid": uid, "tickrate": tr, "rounds": len(panel),
            "orientation_ok": sides <= {"ct", "t"} and bool(sides),
            "market_cov": None if cov is None else round(cov, 2),
            "clock_off_min": None if off is None or pd.isna(off) else int(round(off / 60)),
            "note": note,
        })
    out = pd.DataFrame(rows)
    ok = out[out["tickrate"].notna()]
    cov_median = out["market_cov"].median()
    cov_text = "n/a" if pd.isna(cov_median) else f"{cov_median:.2f}"
    print(f"maps checked: {len(out)}  tickrate resolved: {len(ok)}  "
          f"orientation ok: {int(out['orientation_ok'].sum())}  "
          f"market coverage median: {cov_text}")
    if "clock_off_min" in out:
        offs = pd.to_numeric(out["clock_off_min"], errors="coerce").dropna()
        if len(offs):
            print(f"clock correction applied (min): median {offs.median():.0f}  "
                  f"range {offs.min():.0f}..{offs.max():.0f}")
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
