"""Start-of-round feature matrix.

Leakage is the main risk in this module. Every feature is knowable before the
round is played: economy/equipment/buy at round start, the score *before* the
round, side, map, and match context. Columns derived from the round outcome
(`winner_side`, `winner_team`, `end_reason`, `duration_ticks`, `*_end`) are
asserted absent.

Acceptance: `python -m pipeline.features --map-uid <uid>` prints shapes with
zero nulls and asserts no leakage columns are present.
"""

from __future__ import annotations

import argparse

import pandas as pd

from pipeline import io

LEAKY = {
    "winner_side", "winner_team", "end_reason", "duration_ticks",
    "t_end_utc", "official_end", "end", "freeze_end",
}

NUMERIC = [
    "score_team1", "score_team2", "score_diff", "round_num",
    "is_pistol", "is_second_half", "team1_is_ct",
    "equip_team1", "equip_team2", "equip_diff",
    "balance_team1", "balance_team2", "balance_diff",
    "spend_team1", "spend_team2", "spend_diff",
    "armor_diff", "helmet_diff", "defuser_diff",
    "map_index", "hltv_stars",
]

# map_name is deliberately NOT a feature: a one-hot on map identity lets the model
# memorise that map's outcome (the label is constant per map), which is leakage
# by another name. map_index and the buy categories stay.
CATEGORICAL = ["buy_team1", "buy_team2"]

FEATURE_NAMES = NUMERIC + [f"{c}__{v}" for c in CATEGORICAL for v in ()]  # filled at runtime


def _buy_norm(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.where(s.isin({"eco", "semi", "force", "full"}), "unknown")


def build_map(panel: pd.DataFrame) -> pd.DataFrame:
    """Join a round panel with round_table rows and derive features + label."""
    if panel.empty:
        raise ValueError("empty panel")
    map_uid = str(panel["map_uid"].iloc[0])
    meta = io.get_map(map_uid)
    rt = io.read_table("round_table")
    rt = rt[rt["map_uid"].astype(str) == map_uid].copy()
    if rt.empty:
        raise ValueError(f"no round_table rows for {map_uid}")

    rt["round_num"] = pd.to_numeric(rt["round_num"], errors="coerce").astype(int)
    df = panel.merge(rt, on=["round_num", "map_uid"], how="left", suffixes=("", "_rt"))
    if df["ct_team_name_resolved"].isna().any():
        raise ValueError(f"round_table does not cover every round of {map_uid}")

    team1 = str(meta["team1"])
    team2 = str(meta["team2"])
    def norm(x) -> str:
        return " ".join(str(x).split()).casefold()

    df["team1_is_ct"] = (
        df["ct_team_name_resolved"].map(norm) == norm(team1)
    ).astype(int)

    ct = df["team1_is_ct"].astype(bool)

    def pick(ct_col: str, t_col: str) -> pd.Series:
        return df[ct_col].where(ct, df[t_col])

    df["equip_team1"] = pd.to_numeric(pick("ct_team_current_equip_value",
                                           "t_team_current_equip_value"), errors="coerce")
    df["equip_team2"] = pd.to_numeric(pick("t_team_current_equip_value",
                                           "ct_team_current_equip_value"), errors="coerce")
    df["equip_diff"] = df["equip_team1"] - df["equip_team2"]
    df["balance_team1"] = pd.to_numeric(pick("ct_team_balance", "t_team_balance"), errors="coerce")
    df["balance_team2"] = pd.to_numeric(pick("t_team_balance", "ct_team_balance"), errors="coerce")
    df["balance_diff"] = df["balance_team1"] - df["balance_team2"]
    df["spend_team1"] = pd.to_numeric(pick("ct_team_spend", "t_team_spend"), errors="coerce")
    df["spend_team2"] = pd.to_numeric(pick("t_team_spend", "ct_team_spend"), errors="coerce")
    df["spend_diff"] = df["spend_team1"] - df["spend_team2"]
    df["armor_diff"] = (pd.to_numeric(pick("ct_armor_players", "t_armor_players"),
                                      errors="coerce")
                        - pd.to_numeric(pick("t_armor_players", "ct_armor_players"),
                                        errors="coerce"))
    df["helmet_diff"] = (pd.to_numeric(pick("ct_helmet_players", "t_helmet_players"),
                                       errors="coerce")
                         - pd.to_numeric(pick("t_helmet_players", "ct_helmet_players"),
                                         errors="coerce"))
    df["defuser_diff"] = (pd.to_numeric(pick("ct_defuser_players", "t_defuser_players"),
                                        errors="coerce")
                          - pd.to_numeric(pick("t_defuser_players", "ct_defuser_players"),
                                          errors="coerce"))
    df["buy_team1"] = _buy_norm(pick("ct_buy", "t_buy"))
    df["buy_team2"] = _buy_norm(pick("t_buy", "ct_buy"))

    df = df.sort_values("round_num").reset_index(drop=True)
    # score *before* each round: previous rounds only, never the current one
    win1 = (df["winner_team"].map(norm) == norm(team1)).astype(int)
    df["score_team1"] = win1.cumsum().shift(1).fillna(0).astype(int)
    df["score_team2"] = (df["round_num"] - 1 - df["score_team1"]).astype(int)
    df["score_diff"] = df["score_team1"] - df["score_team2"]
    df["is_pistol"] = df["round_num"].isin([1, 13]).astype(int)
    df["is_second_half"] = (df["round_num"] >= 13).astype(int)
    df["map_name"] = str(meta["map_name"])
    df["map_index"] = int(meta["map_index"])
    df["hltv_stars"] = float(meta.get("hltv_stars", 0) or 0)

    label = int(norm(meta["map_winner_team_resolved"]) == norm(team1))
    df["label"] = label
    df["team1"] = team1
    df["team2"] = team2
    return df


def _one_hot(df: pd.DataFrame, columns: list[str], vocab: dict[str, list[str]]) -> pd.DataFrame:
    out = {}
    for col in columns:
        values = vocab.get(col, sorted(df[col].dropna().astype(str).unique()))
        for v in values:
            out[f"{col}__{v}"] = (df[col].astype(str) == v).astype(float)
    return pd.DataFrame(out, index=df.index)


def feature_matrix(panel: pd.DataFrame, vocab: dict[str, list[str]] | None = None
                   ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names). Nulls are filled with 0 for numerics."""
    df = build_map(panel) if "label" not in panel.columns else panel
    X = df[NUMERIC].astype(float).fillna(0.0)
    Xc = _one_hot(df, CATEGORICAL, vocab or {})
    X = pd.concat([X, Xc], axis=1)
    names = list(X.columns)
    leaked = set(names) & LEAKY
    assert not leaked, f"leakage: {sorted(leaked)} are outcome columns"
    y = pd.Series(df["label"].astype(int).values, index=X.index)
    return X, y, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-uid", required=True)
    args = ap.parse_args()

    from pipeline import align
    panel = align.attach_market(align.map_panel(args.map_uid))
    df = build_map(panel)
    X, y, names = feature_matrix(df)
    print(f"X.shape={X.shape}  y.mean()={y.mean():.3f}  n_features={len(names)}")
    nulls = X.isna().sum()
    assert int(nulls.sum()) == 0, f"nulls present: {nulls[nulls > 0].to_dict()}"
    leaked = set(names) & LEAKY
    assert not leaked, f"leakage: {sorted(leaked)}"
    print("no nulls, no leakage columns")
    print(X.head(3).to_string())
    print(f"\nlabel: team1={df['team1'].iloc[0]} wins map = {int(y.iloc[0])}")


if __name__ == "__main__":
    main()
