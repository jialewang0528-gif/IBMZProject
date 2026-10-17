"""Market series extraction and quality gating.

The real market table carries `market=map_winner` rows per map with
`selection=team1`; `fair_prob` is already a probability (Polymarket has no
margin). A map whose series is too sparse cannot support divergence analysis
and is excluded with a reason rather than silently scored.

Acceptance: `python -m pipeline.market --map-uid <uid>` prints the quality dict.
"""

from __future__ import annotations

import argparse

import pandas as pd

from pipeline import io

MIN_PRINTS = 10
MIN_SPAN_HOURS = 0.5
MAX_GAP_S = 1800


def market_series(map_uid: str, odds: pd.DataFrame | None = None,
                  window: tuple | None = None) -> pd.DataFrame:
    """Return ts_utc, market_p (+ meta) for one map, sorted, deduplicated.

    `window` (start, end) restricts the series to the period the map was being
    traded. A map_winner market opens days ahead, so the raw series contains
    overnight gaps of hours; the quality gate measures continuity, and it has to
    measure it over the part we analyse, not over the pre-match history.
    """
    meta = io.get_map(map_uid)
    odds = io.read_odds() if odds is None else odds
    sel = odds[odds["selection"].astype(str).str.lower() == "team1"]
    sel = sel[sel["market"].astype(str).str.lower().isin({"map_winner", "map winner"})]
    by_uid = sel["map_uid"].astype(str) == str(map_uid)
    by_idx = (sel["match_id"] == meta["match_id"]) & (sel["map_index"] == meta["map_index"])
    series = sel[by_uid | by_idx].copy()
    if window is not None and not series.empty:
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        series = series[(series["ts_utc"] >= lo) & (series["ts_utc"] <= hi)].copy()
    if series.empty:
        return pd.DataFrame(columns=["ts_utc", "market_p", "is_live", "source", "injected"])

    series = series.rename(columns={"fair_prob": "market_p"})[
        ["ts_utc", "market_p", "is_live", "source"]
    ]
    series["market_p"] = pd.to_numeric(series["market_p"], errors="coerce")
    series = (series.dropna(subset=["market_p"])
                    .sort_values("ts_utc")
                    .drop_duplicates(subset=["ts_utc"], keep="last")
                    .reset_index(drop=True))
    series["injected"] = False
    return series


def series_quality(series: pd.DataFrame) -> dict:
    if series.empty:
        return {"n_prints": 0, "n_live_prints": 0, "span_hours": 0.0,
                "max_gap_seconds": None, "usable": False, "reason": "no_market_rows"}
    ts = series["ts_utc"]
    span_h = (ts.max() - ts.min()).total_seconds() / 3600
    gaps = ts.diff().dt.total_seconds().dropna()
    max_gap = float(gaps.max()) if len(gaps) else None
    n_live = int(series["is_live"].fillna(False).sum()) if "is_live" in series else 0

    reason = ""
    if len(series) < MIN_PRINTS:
        reason = f"too_few_prints({len(series)}<{MIN_PRINTS})"
    elif span_h < MIN_SPAN_HOURS:
        reason = f"span_too_short({span_h:.2f}h)"
    elif max_gap is not None and max_gap > MAX_GAP_S:
        reason = f"gap_too_large({max_gap:.0f}s)"
    return {
        "n_prints": int(len(series)),
        "n_live_prints": n_live,
        "span_hours": round(span_h, 2),
        "max_gap_seconds": round(max_gap, 1) if max_gap is not None else None,
        "usable": reason == "",
        "reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-uid", required=True)
    args = ap.parse_args()
    s = market_series(args.map_uid)
    q = series_quality(s)
    print(q)
    if not s.empty:
        print(s.head(3).to_string(index=False))
        print("...")
        print(s.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
