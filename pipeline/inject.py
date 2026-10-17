"""Injection harness: synthetic market prints at chosen rounds.

First-class component (spec §5.3), not a demo hack:
  - every injected row carries injected=True
  - the shift is applied to the market side only, in the direction that moves
    the market *away* from the model, so the divergence is unambiguous
  - original rows are untouched apart from the flagged ones

Acceptance: `python -m pipeline.inject --map-uid <uid> --rounds 14,15,16`
writes a CSV where injected rows are flagged and all other rows are identical.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline import align

INJECTION_DELTA = 0.22
INJECTION_NOTE = "injected prints (simulated)"


def inject(panel: pd.DataFrame, rounds: list[int], delta: float = INJECTION_DELTA,
           rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Move the market probability at `rounds` away from the model.

    `panel` must already carry `model_p` (so direction is defined). The market
    is pushed further from the model, never toward it.
    """
    if "model_p" not in panel.columns:
        raise ValueError("panel needs model_p to choose the injection direction")
    rng = rng or np.random.default_rng(0)
    out = panel.copy()
    out["injected"] = False
    # real market coverage is not 100%: a round inside the injection window can
    # have no print. Fall back to the nearest print, then to the model, rather
    # than dying on a NAType float().
    market_col = pd.to_numeric(out["market_p"], errors="coerce")
    filled = market_col.ffill().bfill()
    for r in rounds:
        m = out["round_num"] == r
        if not m.any():
            continue
        market = out.loc[m, "market_p"].iloc[0]
        if pd.isna(market):
            market = filled.loc[m].iloc[0]
        if pd.isna(market):
            market = out.loc[m, "model_p"].iloc[0]
        market = float(market)
        model = float(out.loc[m, "model_p"].iloc[0])
        direction = 1.0 if market >= model else -1.0
        bump = direction * delta * float(rng.uniform(0.9, 1.1))
        out.loc[m, "market_p"] = float(np.clip(market + bump, 0.01, 0.99))
        out.loc[m, "injected"] = True
    return out


def inject_series(series: pd.DataFrame, panel: pd.DataFrame, rounds: list[int],
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Add the synthetic *prints* the injected rounds imply.

    Without this, the market series shows no step and the "market moved first"
    criterion can never be satisfied for an injected divergence — the panel
    value would change but no print would explain it.
    """
    if series.empty or panel.empty:
        return series
    rng = rng or np.random.default_rng(1)
    extra = []
    for r in rounds:
        row = panel[panel["round_num"] == r]
        if row.empty or pd.isna(row["market_p"].iloc[0]):
            continue
        # the step must be visible BEFORE the round starts: the alert rule is
        # "the market moved first", and it looks for a step in the 10 minutes
        # preceding the round start. Printing at round end made the injected
        # divergence unable to satisfy its own criterion.
        t_start = pd.Timestamp(row["t_start_utc"].iloc[0])
        base = float(row["market_p"].iloc[0])
        for offset_s in (240, 120):
            extra.append({
                "ts_utc": t_start - pd.Timedelta(seconds=offset_s),
                "market_p": float(np.clip(base + rng.normal(0, 0.004), 0.01, 0.99)),
                "is_live": True,
                "source": "injected",
                "injected": True,
            })
    if not extra:
        return series
    out = pd.concat([series, pd.DataFrame(extra)], ignore_index=True)
    if "injected" not in out.columns:
        out["injected"] = False
    out["injected"] = out["injected"].fillna(False)
    return (out.sort_values("ts_utc")
               .drop_duplicates(subset=["ts_utc"], keep="last")
               .reset_index(drop=True))


def injected_rounds(panel: pd.DataFrame) -> list[int]:
    if "injected" not in panel.columns:
        return []
    return sorted(int(r) for r in panel.loc[panel["injected"], "round_num"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-uid", required=True)
    ap.add_argument("--rounds", required=True, help="comma separated, e.g. 14,15,16")
    ap.add_argument("--delta", type=float, default=INJECTION_DELTA)
    ap.add_argument("--out")
    args = ap.parse_args()

    from models import score as score_mod
    from pipeline import features

    panel = align.attach_market(align.map_panel(args.map_uid))
    df = features.build_map(panel)
    X, _, _ = features.feature_matrix(df)
    df["model_p"] = score_mod.score_panel(X)

    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    injected = inject(df, rounds, args.delta)

    changed = injected["injected"].sum()
    unchanged = injected.loc[~injected["injected"], "market_p"].equals(
        df.loc[~injected["injected"], "market_p"])
    print(f"injected rounds: {injected_rounds(injected)} ({changed} rows)")
    print(f"other rows unchanged: {unchanged}")
    assert unchanged, "injection must not alter non-injected rows"

    cols = ["round_num", "market_p", "model_p", "injected"]
    print(injected[cols].to_string(index=False))
    if args.out:
        injected.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
