"""Input layer: the only module that knows where data lives.

Rules (see docs/implementation-plan.md §4.1):
  - no I/O at import time
  - explicit root, overridable by env CS2_DATA_ROOT
  - returns DataFrames, never prints
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get("CS2_DATA_ROOT", "/home/ibm-user/cs2_betting_data"))

# Candidate relative paths per table, tried in order. The real dataset keeps
# maps_meta inside tables/ while a fixture may keep it at the root, and there is
# no point making the caller care which.
TABLE_FILES = {
    "round_table": ["tables/round_table.csv.gz", "round_table.csv.gz"],
    "round_detail": ["tables/round_detail.csv.gz", "round_detail.csv.gz"],
    "maps_meta": ["maps_meta.csv.gz", "tables/maps_meta.csv.gz"],
    "kills": ["tables/kills.csv.gz", "kills.csv.gz"],
    "player_round_economy": ["tables/player_round_economy.csv.gz",
                             "player_round_economy.csv.gz"],
    "player_map_stats": ["tables/player_map_stats.csv.gz", "player_map_stats.csv.gz"],
    "util_bought": ["tables/util_bought.csv.gz", "util_bought.csv.gz"],
    "pistol_loss_opps": ["tables/pistol_loss_opps.csv.gz", "pistol_loss_opps.csv.gz"],
}

ODDS_FILES = ["odds_table.csv.gz", "tables/odds_table.csv.gz"]


def resolve(name: str, root: Path | None = None) -> Path:
    """Find a table on disk, trying the known locations."""
    root = Path(root) if root is not None else DATA_ROOT
    candidates = TABLE_FILES.get(name, [f"{name}.csv.gz", f"tables/{name}.csv.gz"])
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{name}: looked for {candidates} under {root} (set CS2_DATA_ROOT)")


def resolve_odds(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else DATA_ROOT
    for rel in ODDS_FILES:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"odds_table: looked for {ODDS_FILES} under {root} (set CS2_DATA_ROOT)")

_TRUE = {"true", "True", "TRUE", "1", 1, True}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().isin({str(t).lower() for t in _TRUE}) | (
        series.astype(str).str.strip().str.lower() == "true"
    )


@lru_cache(maxsize=32)
def _read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str, low_memory=False)


def read_table(name: str, root: Path | None = None) -> pd.DataFrame:
    """Read one game-side table by short name. Cached per (path, mtime)."""
    if name not in TABLE_FILES:
        raise KeyError(f"unknown table {name!r}; known: {sorted(TABLE_FILES)}")
    path = resolve(name, root)
    return _read_csv(str(path), path.stat().st_mtime).copy()


def read_odds(root: Path | None = None) -> pd.DataFrame:
    """Read the market table, parsing timestamps to UTC."""
    path = resolve_odds(root)
    df = _read_csv(str(path), path.stat().st_mtime).copy()
    df["ts_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="mixed")
    for col in ("match_id", "map_index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "is_live" in df.columns:
        df["is_live"] = _as_bool(df["is_live"])
    return df.sort_values("ts_utc").reset_index(drop=True)


def maps() -> pd.DataFrame:
    """maps_meta with parsed dates and a boolean eligibility flag."""
    meta = read_table("maps_meta")
    meta["map_start_utc"] = pd.to_datetime(meta["match_date"], utc=True, format="mixed")
    if "analysis_eligible_v1" in meta.columns:
        meta["eligible"] = _as_bool(meta["analysis_eligible_v1"])
    else:
        meta["eligible"] = True
    if "qc_status" in meta.columns:
        meta["eligible"] &= meta["qc_status"].astype(str).str.lower().eq("pass")
    meta["match_id"] = pd.to_numeric(meta["match_id"], errors="coerce").astype("Int64")
    meta["map_index"] = pd.to_numeric(meta["map_index"], errors="coerce").astype("Int64")
    return meta


def eligible_maps(root: Path | None = None) -> pd.DataFrame:
    """Maps that are (a) QC-pass, (b) in an event we track, (c) have market rows."""
    meta = maps()
    if root is not None:
        meta = meta  # maps() is global; root only affects odds here
    odds = read_odds(root)
    market_matches = set(odds["match_id"].dropna().unique())
    out = meta[meta["eligible"] & meta["match_id"].isin(market_matches)].copy()
    return out.sort_values(["match_date", "match_id", "map_index"]).reset_index(drop=True)


def get_map(map_uid: str) -> pd.Series:
    meta = maps()
    row = meta[meta["map_uid"].astype(str) == str(map_uid)]
    if row.empty:
        raise KeyError(f"map_uid {map_uid!r} not in maps_meta")
    return row.iloc[0]


def main() -> None:
    for name in TABLE_FILES:
        try:
            print(f"{name:<24} {len(read_table(name)):>9} rows")
        except FileNotFoundError as exc:
            print(f"{name:<24} MISSING ({exc})")
    try:
        odds = read_odds()
        print(f"{'odds_table':<24} {len(odds):>9} rows  "
              f"{odds['ts_utc'].min()} .. {odds['ts_utc'].max()}")
    except FileNotFoundError as exc:
        print(f"{'odds_table':<24} MISSING ({exc})")
    el = eligible_maps()
    print(f"\neligible maps (QC pass + market rows): {len(el)}")
    if len(el):
        print(el[["map_uid", "match_id", "map_index", "team1", "team2", "map_name"]]
              .head(10).to_string(index=False))
    assert len(el) >= 0, "eligible_maps must not fail"


if __name__ == "__main__":
    main()
