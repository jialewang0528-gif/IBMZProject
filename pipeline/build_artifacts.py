"""Build the static artifacts the site and API serve.

Writes the frozen contract from docs/implementation-plan.md §3:
    site/data/matches.json
    site/data/series/<map_uid>.json
    site/data/summary.json

The same bundle is what the page falls back to when the API is unreachable, so
`--static-only` is not a different code path: it is the same build.

Acceptance: bundle parses, has >= 200 maps on real data, every referenced
series file exists, total < 20 MB.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import align, detect, features, inject, io, market
from models import score as score_mod

MARKET_CONTEXT_MIN = 30

DISCLAIMER = ("Divergence signal only. Not a finding about any person. "
              "Part of the market feed may be simulated (see injected flags).")


def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def build_map(uid: str, odds: pd.DataFrame, inject_rounds: list[int] | None = None
              ) -> tuple[dict | None, dict | None]:
    """Return (matches_entry, series_doc) or (None, None) if the map is unusable."""
    meta = io.get_map(uid)
    try:
        panel = align.attach_market(align.map_panel(uid), odds)
    except Exception as exc:  # noqa: BLE001
        return None, {"map_uid": uid, "skipped": f"clock: {type(exc).__name__}: {exc}"}

    # the map's own trading window, with a margin so the opening rounds have a
    # print to compare against
    window = (panel["t_start_utc"].min() - pd.Timedelta(minutes=MARKET_CONTEXT_MIN),
              panel["t_end_utc"].max() + pd.Timedelta(minutes=5))
    series_mkt = market.market_series(uid, odds, window=window)
    quality = market.series_quality(series_mkt)
    if not quality["usable"]:
        return None, {"map_uid": uid, "skipped": f"market: {quality['reason']}"}

    df = features.build_map(panel)
    X, _, _ = features.feature_matrix(df)
    df["model_p"] = score_mod.score_panel(X)
    df["market_p"] = pd.to_numeric(df["market_p"], errors="coerce")

    series_for_detect = series_mkt
    if inject_rounds:
        df = inject.inject(df, inject_rounds)
        series_for_detect = inject.inject_series(series_mkt, df, inject_rounds)

    found = detect.alerts(df, series_for_detect)
    flagged_rounds = [r for a in found for r in range(a["span"][0], a["span"][1] + 1)]
    players = detect.player_deviation(uid, flagged_rounds)
    contrib = {}
    for a in found:
        row_idx = df.index[df["round_num"] == a["round_num"]]
        if len(row_idx):
            contrib[a["round_num"]] = score_mod.contributions(X.iloc[row_idx[0]], top=5)

    team1 = str(meta["team1"])
    team2 = str(meta["team2"])
    rounds = []
    for _, row in df.iterrows():
        rnd = int(row["round_num"])
        rounds.append({
            "round_num": rnd,
            "t_start_utc": _iso(row["t_start_utc"]),
            "score_team1": int(row["score_team1"]),
            "score_team2": int(row["score_team2"]),
            "team1_side": "ct" if int(row["team1_is_ct"]) else "t",
            "alive_team1": None,   # not present in the round tables; see summary.data_notes
            "alive_team2": None,
            "armor_team1": int(row.get("armor_diff", 0) >= 0),
            "equip_diff_team1_minus_team2": float(row["equip_diff"]),
            "buy_team1": str(row["buy_team1"]),
            "buy_team2": str(row["buy_team2"]),
            "model_p": round(float(row["model_p"]), 4),
            "market_p": None if pd.isna(row["market_p"]) else round(float(row["market_p"]), 4),
            "gap": None if pd.isna(row["market_p"]) else round(float(row["market_p"]) - float(row["model_p"]), 4),
            "market_ts_utc": _iso(row.get("market_ts_utc")),
            "market_ts_delta_s": None if pd.isna(row.get("market_ts_delta_s")) else int(row["market_ts_delta_s"]),
            "injected": bool(row.get("injected", False)),
        })

    alerts_doc = []
    for a in found:
        alerts_doc.append({
            **a,
            "contributions": contrib.get(a["round_num"], []),
            "disclaimer": DISCLAIMER,
        })

    series_doc = {
        "meta": {
            "map_uid": uid,
            "match_id": int(meta["match_id"]),
            "event": str(meta.get("event", "")),
            "map_name": str(meta["map_name"]),
            "team1": team1,
            "team2": team2,
            "map_start_utc": _iso(meta["map_start_utc"]),
            "tickrate": int(panel["tickrate"].iloc[0]),
            "resolution_floor_s": int(panel["floor_s"].iloc[0]),
            "model": "logit-v1",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "rounds": rounds,
        "alerts": alerts_doc,
        "players": players.to_dict(orient="records") if not players.empty else [],
    }

    max_gap = max((abs(r["gap"]) for r in rounds if r["gap"] is not None), default=0.0)
    sev_rank = {"high": 3, "medium": 2, "low": 1}
    matches_entry = {
        "map_uid": uid,
        "match_id": int(meta["match_id"]),
        "match_date": _iso(meta["map_start_utc"]),
        "event": str(meta.get("event", "")),
        "map_name": str(meta["map_name"]),
        "team1": team1,
        "team2": team2,
        "n_rounds": len(rounds),
        "hltv_stars": float(meta.get("hltv_stars", 0) or 0),
        "max_gap": round(float(max_gap), 4),
        "severity": max((a["severity"] for a in found), key=lambda s: sev_rank.get(s, 0),
                        default="none"),
        "n_alerts": len(found),
        "has_injection": bool(inject_rounds),
        "injected_rounds": inject.injected_rounds(df),
        "resolution_floor_s": int(panel["floor_s"].iloc[0]),
    }
    return matches_entry, series_doc


def build(out_dir: Path, limit: int | None = None, inject_map: str | None = None,
          inject_rounds: list[int] | None = None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "series").mkdir(parents=True, exist_ok=True)
    odds = io.read_odds()
    el = io.eligible_maps()
    if limit:
        el = el.head(limit)

    entries, skipped = [], []
    for _, meta in el.iterrows():
        uid = str(meta["map_uid"])
        inj = inject_rounds if (inject_map and uid == inject_map) else None
        try:
            entry, doc = build_map(uid, odds, inj)
        except Exception as exc:  # noqa: BLE001 - one bad map must not stop the build
            entry, doc = None, {"map_uid": uid,
                                "skipped": f"{type(exc).__name__}: {exc}"[:200]}
        if entry is None:
            skipped.append(doc)
            continue
        entries.append(entry)
        (out_dir / "series" / f"{uid}.json").write_text(json.dumps(doc, indent=1))

    entries.sort(key=lambda e: (-e["n_alerts"], -e["max_gap"]))
    (out_dir / "matches.json").write_text(json.dumps(entries, indent=1))

    metrics_path = Path("models/artifacts/metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_maps": len(entries),
        "n_alerts": sum(e["n_alerts"] for e in entries),
        "n_skipped": len(skipped),
        "skipped": skipped[:50],
        "injected_maps": [e["map_uid"] for e in entries if e["has_injection"]],
        "resolution_floor_s": 60,
        "model_metrics": {k: metrics.get(k) for k in ("logit", "xgb", "baseline_coinflip")
                          if metrics.get(k)},
        "disclaimer": DISCLAIMER,
        "data_notes": [
            "alive_team1/alive_team2 are null: the round tables carry team-level "
            "armor/helmet/defuser counts but not per-side alive counts.",
            "market timestamps are 1-minute snapshots; ordering below ~60s plus "
            "map-start error is not measurable.",
            "alerts are computed on the gap relative to each map's own opening "
            "offset (baseline_gap), not on the absolute gap.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    total = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    print(f"built {len(entries)} maps, {summary['n_alerts']} alerts, "
          f"{len(skipped)} skipped, {total / 1e6:.2f} MB -> {out_dir}")
    return {"entries": entries, "summary": summary, "bytes": total, "skipped": skipped}


def validate(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    matches = json.loads((out_dir / "matches.json").read_text())
    summary = json.loads((out_dir / "summary.json").read_text())
    assert isinstance(matches, list) and matches, "matches.json must be a non-empty list"
    assert summary["n_maps"] == len(matches), "summary.n_maps disagrees with matches.json"
    for m in matches:
        p = out_dir / "series" / f"{m['map_uid']}.json"
        assert p.exists(), f"missing series file for {m['map_uid']}"
        doc = json.loads(p.read_text())
        for key in ("meta", "rounds", "alerts", "players"):
            assert key in doc, f"{p.name} missing {key}"
        assert len(doc["rounds"]) == m["n_rounds"], f"{p.name} round count mismatch"
        for r in doc["rounds"]:
            assert 0.0 <= r["model_p"] <= 1.0, f"{p.name} model_p out of range"
        for a in doc["alerts"]:
            assert a["disclaimer"], f"{p.name} alert without disclaimer"
    total = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    assert total < 20e6, f"bundle too large: {total / 1e6:.1f} MB"
    print(f"validated: {len(matches)} maps, {total / 1e6:.2f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inject-map")
    ap.add_argument("--inject-rounds", default="14,15,16")
    ap.add_argument("--static-only", action="store_true",
                    help="same bundle; static-only is the fallback the page uses")
    args = ap.parse_args()
    rounds = [int(x) for x in args.inject_rounds.split(",") if x.strip()]
    build(Path(args.out), args.limit, args.inject_map, rounds if args.inject_map else None)
    validate(Path(args.out))


if __name__ == "__main__":
    main()
