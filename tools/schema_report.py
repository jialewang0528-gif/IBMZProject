#!/usr/bin/env python3
"""Print a schema + join-rate report for the CS2 betting data dir.

Stdlib only. No quoting tricks needed: run it as a single command.

    python3 tools/schema_report.py [DATA_ROOT]

DATA_ROOT defaults to /home/ibm-user/cs2_betting_data.
Read-only: opens files, prints, writes nothing.
"""

import gzip
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/ibm-user/cs2_betting_data"

FILES = [
    "matches.csv", "maps.csv", "markets.csv", "odds_join_keys.csv",
    "qa_report.csv", "odds_features.csv", "odds_features_match.csv",
    "unmatched_series.csv", "odds_join_report.csv",
]
TABLES = ["round_table", "round_detail", "player_round_economy", "kills",
          "maps_meta", "util_bought", "pistol_loss_opps", "player_map_stats"]
CAP = 600


def opener(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace", newline="")
    return open(path, "r", errors="replace", newline="")


def delimiter(header):
    counts = {d: header.count(d) for d in [",", "\t", ";", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def scan(path, key_hints, max_rows=None):
    """Return (n_rows, header_list, delim, key_values_set, first_two_rows)."""
    n = 0
    with opener(path) as fh:
        header_line = fh.readline().rstrip("\r\n")
        if not header_line:
            return 0, [], ",", set(), []
        d = delimiter(header_line)
        header = [h.strip() for h in header_line.split(d)]
        key_idx = [
            i for i, h in enumerate(header)
            if any(hint in h.lower() for hint in key_hints)
        ]
        keys = set()
        first = []
        for line in fh:
            if max_rows and n >= max_rows:
                break
            n += 1
            if len(first) < 2:
                first.append(line.rstrip("\r\n")[:CAP])
            if key_idx:
                parts = line.split(d)
                for i in key_idx:
                    if i < len(parts) and parts[i].strip():
                        keys.add(parts[i].strip())
    return n, header, d, keys, first


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"no such directory: {ROOT}")
    print(f"# schema report for {ROOT}\n")

    print("== top level ==")
    for name in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, name)
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        print(f"  {'dir ' if os.path.isdir(p) else f'{sz/1e6:9.2f} MB'}  {name}")
    for sub in ("tables", "teammate_data", "prices", "trades"):
        p = os.path.join(ROOT, sub)
        if os.path.isdir(p):
            print(f"\n== {sub}/ ({len(os.listdir(p))} entries) ==")
            for name in sorted(os.listdir(p))[:15]:
                q = os.path.join(p, name)
                if os.path.isfile(q):
                    print(f"  {os.path.getsize(q)/1e6:9.2f} MB  {name}")

    print("\n== files ==")
    id_sets = {}
    for name in FILES:
        p = os.path.join(ROOT, name)
        if not os.path.isfile(p):
            continue
        n, header, d, keys, first = scan(p, ("match", "series", "id"))
        print(f"\n-- {name}: {n} rows, {len(header)} cols, delim={d!r}")
        print(f"   cols: {header[:25]}{' ...' if len(header) > 25 else ''}")
        for row in first:
            print(f"   row: {row}")
        if keys and len(keys) < 50000:
            id_sets[name] = keys
            print(f"   key values: {len(keys)}")

    print("\n== tables/ ==")
    table_keys = {}
    for t in TABLES:
        p = os.path.join(ROOT, "tables", f"{t}.csv.gz")
        if not os.path.isfile(p):
            p = os.path.join(ROOT, f"{t}.csv.gz")
        if not os.path.isfile(p):
            continue
        n, header, d, keys, first = scan(p, ("match", "demo", "game", "series"))
        print(f"\n-- {t}: {n} rows, {len(header)} cols")
        print(f"   cols: {header[:30]}{' ...' if len(header) > 30 else ''}")
        for row in first:
            print(f"   row: {row}")
        if keys:
            table_keys[t] = keys
            print(f"   key values: {len(keys)}")

    print("\n== JOIN RATES ==")
    match_ids = id_sets.get("matches.csv")
    if match_ids:
        print(f"  matches.csv ids: {len(match_ids)}")
        for t, keys in table_keys.items():
            inter = match_ids & keys
            print(f"  {t:<24} ids={len(keys):<8} overlap={len(inter):<8} "
                  f"({100*len(inter)/max(len(match_ids),1):.1f}% of matches)")
    else:
        print("  matches.csv missing or no id-like column found")

    odds_keys = None
    p = os.path.join(ROOT, "odds_table.csv.gz")
    if os.path.isfile(p):
        n, header, d, keys, first = scan(p, ("match_id",), max_rows=400000)
        odds_keys = keys
        print(f"\n  odds_table.csv.gz: {n} rows scanned, {len(header)} cols")
        print(f"   cols: {header}")
        for row in first:
            print(f"   row: {row}")
        if match_ids:
            inter = match_ids & keys
            print(f"   match_id overlap with matches.csv: {len(inter)} "
                  f"({100*len(inter)/max(len(match_ids),1):.1f}%)")
        for t, keys2 in table_keys.items():
            inter = keys2 & keys
            print(f"   match_id overlap with {t}: {len(inter)}")


if __name__ == "__main__":
    main()
