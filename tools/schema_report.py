#!/usr/bin/env python3
"""Print a schema + join-rate report for the CS2 betting data dir.

Stdlib only. No quoting tricks needed: run it as a single command.

    python3 tools/schema_report.py [DATA_ROOT]

DATA_ROOT defaults to /home/ibm-user/cs2_betting_data.
Read-only: opens files, prints, writes nothing.

HISTORY - read before changing the key logic
--------------------------------------------
An earlier version of this script produced the two numbers the whole project
then quoted as measurements:

  * it chose "key" columns by substring hint (("match", "demo", "game",
    "series")), so maps_meta contributed match_id AND match_date AND
    series_score1 AND series_score2 to ONE set, and it printed that set's size
    as "key values". On this data that is 860 + 860 + 3 = 1,723 - which was
    copied into the spec as "1,723 matches". 1,988 maps / 1,723 series = 1.15
    maps per series, impossible for a best-of-3.
  * it capped the odds-table scan at 400,000 rows and did not say so, so the
    "overlap" it printed was an intersection with a truncated slice. That is
    where "297" came from.

So: key columns are now named explicitly, every distinct count is reported PER
COLUMN (never unioned), and a truncated scan says TRUNCATED and refuses to
print a ratio it would be lying about.
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

# Columns that are genuinely keys. Anything else is data, and two data columns
# must never be merged into one "key count".
KEY_COLUMNS = ("match_id", "map_uid", "match_uid", "series_id", "event_id")


def opener(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace", newline="")
    return open(path, "r", errors="replace", newline="")


def delimiter(header):
    counts = {d: header.count(d) for d in [",", "\t", ";", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def scan(path, max_rows=None):
    """Read one delimited file. Counts are kept SEPARATELY per key column."""
    n = 0
    truncated = False
    with opener(path) as fh:
        header_line = fh.readline().rstrip("\r\n")
        if not header_line:
            return {"n": 0, "header": [], "delim": ",", "values": {},
                    "first": [], "truncated": False}
        d = delimiter(header_line)
        header = [h.strip() for h in header_line.split(d)]
        key_idx = {h: i for i, h in enumerate(header) if h.lower() in KEY_COLUMNS}
        values = {h: set() for h in key_idx}
        first = []
        for line in fh:
            if max_rows is not None and n >= max_rows:
                truncated = True
                break
            n += 1
            if len(first) < 2:
                first.append(line.rstrip("\r\n")[:CAP])
            parts = line.split(d)
            for h, i in key_idx.items():
                if i < len(parts) and parts[i].strip():
                    values[h].add(parts[i].strip())
    return {"n": n, "header": header, "delim": d, "values": values,
            "first": first, "truncated": truncated}


def report_key_columns(label, result):
    """Print every distinct count on its own line. Never print a union."""
    for column, values in result["values"].items():
        suffix = "  TRUNCATED - not a total" if result["truncated"] else ""
        print(f"   {column}: {len(values)} distinct{suffix}")
    if len(result["values"]) > 1:
        names = ", ".join(result["values"])
        print(f"   NOTE: {len(result['values'])} key-like columns present ({names});"
              f" they are counted separately on purpose")


def find_file(name):
    for base in (ROOT, os.path.join(ROOT, "tables")):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return None


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
    file_keys = {}
    for name in FILES:
        p = find_file(name)
        if not p:
            continue
        res = scan(p)
        print(f"\n-- {name}: {res['n']} rows, {len(res['header'])} cols, "
              f"delim={res['delim']!r}")
        print(f"   cols: {res['header'][:25]}{' ...' if len(res['header']) > 25 else ''}")
        for row in res["first"]:
            print(f"   row: {row}")
        report_key_columns(name, res)
        if "match_id" in res["values"]:
            file_keys[name] = res["values"]["match_id"]

    print("\n== tables/ ==")
    table_keys = {}
    for t in TABLES:
        p = find_file(f"{t}.csv.gz")
        if not p:
            continue
        res = scan(p)
        print(f"\n-- {t}: {res['n']} rows, {len(res['header'])} cols")
        print(f"   cols: {res['header'][:30]}{' ...' if len(res['header']) > 30 else ''}")
        for row in res["first"]:
            print(f"   row: {row}")
        report_key_columns(t, res)
        if "match_id" in res["values"]:
            table_keys[t] = res["values"]["match_id"]

    print("\n== JOIN RATES ==")
    print("  All rates below join ONE named column against ONE named column.")
    market_ids = file_keys.get("odds_join_keys.csv")
    if market_ids:
        print(f"\n  odds_join_keys.csv match_id: {len(market_ids)} distinct")
        for t, keys in table_keys.items():
            inter = keys & market_ids
            print(f"    {t:<24} match_id={len(keys):<6} "
                  f"shared with join keys={len(inter):<6} "
                  f"({100*len(inter)/max(len(market_ids),1):.1f}% of market side)")

    p = find_file("odds_table.csv.gz")
    if p:
        # No row cap: a truncated scan cannot honestly report an intersection.
        res = scan(p)
        print(f"\n  odds_table.csv.gz: {res['n']} rows, {len(res['header'])} cols")
        print(f"   cols: {res['header']}")
        for row in res["first"]:
            print(f"   row: {row}")
        report_key_columns("odds_table", res)
        for column, values in res["values"].items():
            for t, keys in table_keys.items():
                if not keys:
                    continue
                inter = keys & values
                print(f"   {column} overlap with {t}: {len(inter)}")
    else:
        print("\n  odds_table.csv.gz: MISSING - no join rate computed")


if __name__ == "__main__":
    main()
