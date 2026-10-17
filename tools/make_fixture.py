#!/usr/bin/env python3
"""Generate a synthetic cs2_betting_data tree that mirrors the real schemas.

Purpose: let the pipeline, models, server and site be developed and tested
on a laptop, without the 485 MB real dataset. Column names are copied from
the real tables so nothing silently diverges.

Usage:
    python tools/make_fixture.py --out /tmp/fixture [--maps 6] [--rounds 20]

Layout produced (same as the real dir):
    maps_meta.csv.gz, odds_table.csv.gz
    tables/{round_table,round_detail,player_round_economy,kills,
            player_map_stats,util_bought,pistol_loss_opps}.csv.gz

Deliberately includes:
  - 3 matches x 2 maps, best-of-3, real-ish team names
  - 1-minute market series with pre-match and live prints
  - one map whose market drifts away from the game state (the demo case)
"""

import argparse
import csv
import gzip
import os
import random
from datetime import datetime, timedelta, timezone

TICKRATE = 64
MAPS = ["de_overpass", "de_ancient", "de_mirage", "de_nuke", "de_inferno"]

def teams():
    """Synthetic team pairs; enough matches for the model to be well specified."""
    names = ["magic", "Inner Circle Academy", "Johnny Speeds", "FC Famalicão Esports",
             "BetBoom", "Sangal", "Nemiga", "BESTIA", "Wildcard", "B8",
             "Aurora", "Heroic Academy", "9INE", "Fnatic Rising", "Endpoint", "Sampi",
             "Monte", "GamerLegion", "Zero Tenacity", "Insilio"]
    out = []
    for i in range(0, len(names) - 1, 2):
        for suffix in ("", " B"):
            a, b = names[i] + suffix, names[i + 1] + suffix
            out.append((a, b, f"{a}-vs-{b}.dem"))
    return out


def w(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


def buy_for(equip):
    if equip < 1500:
        return "eco"
    if equip < 4000:
        return "semi"
    if equip < 5500:
        return "force"
    return "full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fixture")
    ap.add_argument("--maps", type=int, default=80)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = args.out
    os.makedirs(root, exist_ok=True)

    maps_meta, rounds_tbl, rounds_det = [], [], []
    players_eco, kills, pmstats, util = [], [], [], []
    odds = []

    base = datetime(2026, 3, 23, 18, 4, 9, tzinfo=timezone(timedelta(hours=-7)))
    n_maps_total = args.maps
    demo_map_index = 1  # second map of the first match gets the drifting market
    map_counter = 0

    all_teams = teams()
    for mi, (team1, team2, demofile) in enumerate(all_teams):
        # matches_meta.match_date is the SERIES start and is identical for every
        # map of the match, exactly like the real data.
        match_start = base + timedelta(days=mi)
        prev_end_s = 0.0
        for k in range(2):
            if map_counter >= n_maps_total:
                break
            match_id = 2392432 + mi
            map_index = k + 1
            map_uid = f"{match_id}-{map_index}"
            map_name = MAPS[map_counter % len(MAPS)]
            map_start_offset_s = 0.0 if k == 0 else prev_end_s + 600
            start = match_start + timedelta(seconds=map_start_offset_s)
            is_demo_map = map_counter == demo_map_index
            map_counter += 1

            # team1 starts CT on even map_index, T on odd
            team1_starts_ct = (map_index % 2 == 0)
            score = {team1: 0, team2: 0}
            start_tick = 0
            round_rows = []
            timeline = []  # (t_round_start, score_team1_at_round_start, equip_diff)
            for r in range(1, args.rounds + 1):
                team1_ct = team1_starts_ct == (r <= args.rounds // 2)
                ct_name, t_name = (team1, team2) if team1_ct else (team2, team1)

                # economy: pistol rounds cheap, then snowball from the score
                if r in (1, args.rounds // 2 + 1):
                    eq_ct = rng.choice([4000, 4000, 800])
                    eq_t = rng.choice([4000, 800, 4000])
                else:
                    lead = score[ct_name] - score[t_name]
                    eq_ct = max(600, 4200 + 380 * lead + rng.randint(-1800, 1800))
                    eq_t = max(600, 4200 - 380 * lead + rng.randint(-1800, 1800))

                p_ct = 0.5 + (eq_ct - eq_t) / 32000 + (0.03 if rng.random() < 0.1 else 0)
                p_ct = min(max(p_ct, 0.06), 0.94)
                ct_wins = rng.random() < p_ct
                winner = ct_name if ct_wins else t_name
                winner_side = "ct" if ct_wins else "t"
                score[winner] += 1

                bal_ct = max(0, 800 + 120 * score[ct_name] + rng.randint(-300, 300))
                bal_t = max(0, 800 + 120 * score[t_name] + rng.randint(-300, 300))

                rounds_tbl.append([
                    map_uid, match_id, map_index, f"Event {mi}", map_name,
                    start.strftime("%Y-%m-%d %H:%M:%S%z"), r, winner_side,
                    "bomb_exploded" if winner_side == "t" else "ct_killed",
                    str(rng.random() < 0.35).upper(),
                    ct_name, f"roster-{ct_name[:3]}", bal_ct, eq_ct, eq_ct,
                    min(5, int(eq_ct > 2000) * 5), min(5, int(eq_ct > 3800) * 5), 0,
                    t_name, f"roster-{t_name[:3]}", bal_t, eq_t, eq_t,
                    min(5, int(eq_t > 2000) * 5), min(5, int(eq_t > 3800) * 5), 0,
                    buy_for(eq_ct), buy_for(eq_t), float(eq_ct - eq_t), winner,
                ])

                timeline.append((start + timedelta(seconds=start_tick / TICKRATE),
                                 score[team1] - score[team2],
                                 (eq_ct - eq_t) if team1_ct else (eq_t - eq_ct)))
                freeze = start_tick + int(20 * TICKRATE)
                dur = int(rng.randint(55, 105) * TICKRATE)
                end_tick = freeze + dur
                rounds_det.append([
                    map_uid, r, start_tick, freeze, end_tick, end_tick + int(3 * TICKRATE),
                    winner_side, "bomb_exploded" if winner_side == "t" else "ct_killed",
                    str(rng.random() < 0.35).upper(), "bombsite_b", dur,
                ])
                start_tick = end_tick + int(7 * TICKRATE)

                for side, name, eq in (("ct", ct_name, eq_ct), ("t", t_name, eq_t)):
                    for p in range(5):
                        steamid = 76561198000000000 + (hash((name, p)) % 100000)
                        players_eco.append([
                            map_uid, r, steamid, f"{name[:4]}_{p}", side,
                            f"roster-{name[:3]}", name, 800, max(0, eq // 5 - 650),
                            max(0, eq // 5 - 650), eq // 5, 100 if eq > 2000 else 0,
                            str(eq > 3800).upper(), "FALSE",
                        ])
                        util.append([map_uid, r, steamid, f"{name[:4]}_{p}", side, name,
                                     int(rng.random() < .1), int(rng.random() < .3),
                                     int(rng.random() < .6), int(rng.random() < .4),
                                     int(rng.random() < .7)])
                        if rng.random() < 0.55:
                            victim_side = "t" if side == "ct" else "ct"
                            victim_team = team2 if name == team1 else team1
                            kills.append([
                                map_uid, freeze + int(rng.random() * dur), r,
                                f"{name[:4]}_{p}", steamid, side,
                                f"{victim_team[:4]}_{p}", 76561198000000000 + (hash((victim_team, p)) % 100000),
                                victim_side, "", "", "ak47" if eq > 3000 else "glock",
                                str(rng.random() < .4).upper(), round(rng.uniform(3, 40), 2),
                                "FALSE", "FALSE", "FALSE", "FALSE", rng.randint(15, 120), "head",
                            ])

            prev_end_s = float(rounds_det[-1][5]) / TICKRATE  # official_end of last round

            # map result
            winner_team = team1 if score[team1] > score[team2] else team2
            rounds_played = min(args.rounds, max(score.values()))
            maps_meta.append([
                map_uid, match_id, map_index, f"Event {mi}", team1, team2,
                score[team1], score[team2], map_name, match_start.isoformat(),
                score[team1] + score[team2], "regulation", winner_team, 0,
                "True", "pass",
            ])

            for side, name in (("ct", team1), ("t", team2)):
                for p in range(5):
                    kills_c = sum(1 for k in kills if k[0] == map_uid and k[3].startswith(name[:4]))
                    pmstats.append([
                        map_uid, f"{name[:4]}_{p}", 76561198000000000 + (hash((name, p)) % 100000),
                        side, float(args.rounds), round(rng.uniform(.6, 1.4), 3),
                        round(rng.uniform(.6, 1.4), 3), float(args.rounds - 4),
                        round(rng.uniform(50, 95), 1), float(rng.randint(600, 1400)),
                        round(rng.uniform(40, 90), 1), float(kills_c),
                        rng.randint(0, 8), float(rng.randint(6, 20)), rng.randint(0, 6),
                        round(rng.uniform(0, .8), 2), f"roster-{name[:3]}", name,
                    ])

            # --- market series: 1-minute prints, pre-match and live -------------
            inj_lo = inj_hi = None
            if is_demo_map:
                by_round = {r: tl for r, tl in zip(range(1, args.rounds + 1), timeline)}
                if 14 in by_round and 16 in by_round:
                    inj_lo = by_round[14][0] - timedelta(minutes=2)
                    inj_hi = by_round[16][0]
            series_start = start - timedelta(minutes=90)
            series_end = start + timedelta(minutes=int(args.rounds * 1.6))
            t = series_start
            true_p = 0.5
            drift = 0.0
            while t <= series_end:
                # game state moves the true probability; market mostly follows
                if t >= start and timeline:
                    past = [x for x in timeline if x[0] <= t]
                    diff, ediff = (past[-1][1], past[-1][2]) if past else (0, 0)
                    z = 0.30 * diff + ediff / 5000.0
                    true_p = min(max(1.0 / (1.0 + 2.718281828 ** (-z)), 0.03), 0.97)
                p = min(max(true_p + drift + rng.gauss(0, 0.012), 0.01), 0.99)
                # No drift baked in: divergence is produced at runtime by
                # pipeline/inject.py, which is what the demo must exercise.
                odds.append([
                    match_id, map_index, "polymarket", "polymarket", "map_winner",
                    1000000 + map_counter, "team1", "", round(1 / p, 4), round(p, 4),
                    round(p, 4), t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    str(t >= start).upper(), map_uid, "", "",
                ])
                t += timedelta(minutes=1)

    w(f"{root}/maps_meta.csv.gz",
      ["map_uid", "match_id", "map_index", "event", "team1", "team2", "series_score1",
       "series_score2", "map_name", "match_date", "rounds_played", "regulation_or_overtime",
       "map_winner_team_resolved", "hltv_stars", "analysis_eligible_v1", "qc_status"],
      maps_meta)

    w(f"{root}/tables/round_table.csv.gz",
      ["map_uid", "match_id", "map_index", "event", "map_name", "match_date", "round_num",
       "winner_side", "end_reason", "bomb_planted", "ct_team_name_resolved", "ct_roster_uid",
       "ct_team_balance", "ct_team_spend", "ct_team_current_equip_value", "ct_armor_players",
       "ct_helmet_players", "ct_defuser_players", "t_team_name_resolved", "t_roster_uid",
       "t_team_balance", "t_team_spend", "t_team_current_equip_value", "t_armor_players",
       "t_helmet_players", "t_defuser_players", "ct_buy", "t_buy",
       "equip_diff_ct_minus_t", "winner_team"], rounds_tbl)

    w(f"{root}/tables/round_detail.csv.gz",
      ["map_uid", "round_num", "start", "freeze_end", "end", "official_end", "winner",
       "reason", "bomb_plant", "bomb_site", "duration_ticks"], rounds_det)

    w(f"{root}/tables/player_round_economy.csv.gz",
      ["map_uid", "round_num", "steamid", "player_name", "side", "roster_uid",
       "team_name_resolved", "start_balance", "balance", "cash_spent_this_round",
       "current_equip_value", "armor_value", "has_helmet", "has_defuser"], players_eco)

    w(f"{root}/tables/kills.csv.gz",
      ["map_uid", "tick", "round_num", "attacker_name", "attacker_steamid", "attacker_side",
       "victim_name", "victim_steamid", "victim_side", "assister_name", "assister_steamid",
       "weapon", "headshot", "distance", "noscope", "thrusmoke", "penetrated",
       "attackerblind", "dmg_health", "hitgroup"], kills)

    w(f"{root}/tables/player_map_stats.csv.gz",
      ["map_uid", "name", "steamid", "side", "n_rounds", "impact", "rating", "kast_rounds",
       "kast", "dmg", "adr", "kills", "headshots", "deaths", "assists", "hs_rate",
       "roster_uid", "team_name_resolved"], pmstats)

    w(f"{root}/tables/util_bought.csv.gz",
      ["map_uid", "round_num", "steamid", "player_name", "side", "team_name_resolved",
       "decoy", "fire", "flash", "he", "smoke"], util)

    w(f"{root}/tables/pistol_loss_opps.csv.gz",
      ["map_uid", "match_id", "map_index", "event", "map_name", "match_date", "pistol_round",
       "half", "pistol_winner_side", "decision_round", "pistol_loser_team"],
      [[m[0], m[1], m[2], m[3], m[8], m[9], 1, 1, "ct", 2, m[4]] for m in maps_meta])

    w(f"{root}/odds_table.csv.gz",
      ["match_id", "map_index", "source", "bookmaker", "market", "market_id", "selection",
       "line", "decimal_odds", "implied_prob", "fair_prob", "timestamp_utc", "is_live",
       "map_uid", "est_live_map_index", "est_live_round"], odds)

    print(f"fixture written to {root}")
    print(f"  maps={len(maps_meta)} rounds={len(rounds_tbl)} odds_rows={len(odds)} "
          f"kills={len(kills)} eco_rows={len(players_eco)}")
    print(f"  demo map with drifting market: map_uid={maps_meta[demo_map_index][0]}")


if __name__ == "__main__":
    main()
