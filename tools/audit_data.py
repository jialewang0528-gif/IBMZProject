#!/usr/bin/env python3
"""Audit a directory of .gz tables and print a compact, paste-back summary.

Stdlib only (no pandas), so it runs on the IBM Z box, the Jupyter host,
or a laptop with nothing installed.

Usage:
    python3 audit_data.py /cs2_betting_data
    python3 audit_data.py /cs2_betting_data --max-rows 200000 --uniq-cap 5000

Prints a per-file column profile (type guess, null %, distinct count, range,
samples), join-key hints, and a JSON block to paste back.
Read-only and streaming: safe on multi-GB files.
"""

import argparse
import gzip
import json
import os
import re
import sys

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?|\d{10}(\.\d+)?|\d{13})$")


def open_any(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace", newline="")
    return open(path, "r", errors="replace", newline="")


def sniff_delimiter(header_line):
    counts = {d: header_line.count(d) for d in [",", "\t", ";", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def is_num(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def profile_jsonl(path, max_rows):
    """Profile a JSON Lines file: top-level keys, nesting, sample values."""
    keys = {}
    lines = 0
    with open_any(path) as fh:
        for line in fh:
            if lines >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                keys.setdefault("<unparseable>", {"n": 0, "types": set(), "sample": ""})
                keys["<unparseable>"]["n"] += 1
                continue
            for k, v in obj.items():
                st = keys.setdefault(k, {"n": 0, "types": set(), "sample": "", "nested": set()})
                st["n"] += 1
                st["types"].add(type(v).__name__)
                if st["sample"] == "":
                    st["sample"] = str(v)[:60]
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    st["nested"].update(list(v[0].keys())[:12])
                elif isinstance(v, dict):
                    st["nested"].update(list(v.keys())[:12])
    cols = []
    for k, st in sorted(keys.items(), key=lambda kv: -kv[1]["n"]):
        c = {"name": k, "kind": "/".join(sorted(st["types"])) or "null",
             "present_pct": round(100 * st["n"] / max(lines, 1), 1),
             "sample": st["sample"]}
        if st["nested"]:
            c["nested_keys"] = sorted(st["nested"])
        cols.append(c)
    return {"path": path, "format": "jsonl", "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "lines_scanned": lines, "hit_row_cap": lines >= max_rows, "columns": cols}


def profile(path, max_rows, uniq_cap, max_cols=80):
    size = os.path.getsize(path)
    with open_any(path) as fh:
        header = fh.readline().rstrip("\r\n")
        if not header:
            return {"path": path, "error": "empty file"}
        delim = sniff_delimiter(header)
        cols = [c.strip() for c in header.split(delim)]

        state = [{"nulls": 0, "num": 0, "nonnum": 0, "ts": 0, "uniq": set(),
                  "uniq_overflow": False, "min": None, "max": None, "sum": 0.0,
                  "ts_min": None, "ts_max": None, "samples": [], "maxlen": 0}
                 for _ in cols]
        rows = 0
        for line in fh:
            if rows >= max_rows:
                break
            rows += 1
            parts = line.rstrip("\r\n").split(delim)
            for i in range(len(cols)):
                st = state[i]
                val = parts[i] if i < len(parts) else ""
                if val == "" or val.upper() in ("NA", "NAN", "NULL", "NONE", "-"):
                    st["nulls"] += 1
                    continue
                st["maxlen"] = max(st["maxlen"], len(val))
                if not st["uniq_overflow"]:
                    st["uniq"].add(val)
                    if len(st["uniq"]) > uniq_cap:
                        st["uniq_overflow"] = True
                        st["uniq"] = set()
                if TS_RE.match(val):
                    st["ts"] += 1
                    st["ts_min"] = val if st["ts_min"] is None else min(st["ts_min"], val)
                    st["ts_max"] = val if st["ts_max"] is None else max(st["ts_max"], val)
                if is_num(val):
                    st["num"] += 1
                    v = float(val)
                    st["sum"] += v
                    st["min"] = v if st["min"] is None else min(st["min"], v)
                    st["max"] = v if st["max"] is None else max(st["max"], v)
                else:
                    st["nonnum"] += 1
                    if len(st["samples"]) < 3 and val not in st["samples"]:
                        st["samples"].append(val[:40])

    col_reports = []
    for name, st in zip(cols[:max_cols], state[:max_cols]):
        seen = max(st["nulls"] + st["num"] + st["nonnum"], 1)
        if st["ts"] == seen - st["nulls"] and st["nulls"] < seen:
            kind = "timestamp"
        elif st["nonnum"] == 0 and st["nulls"] < seen:
            kind = "numeric"
        elif st["num"] == 0 and st["nulls"] < seen:
            kind = "string"
        else:
            kind = "mixed"
        rep = {
            "name": name,
            "kind": kind,
            "null_pct": round(100 * st["nulls"] / seen, 1),
            "distinct": None if st["uniq_overflow"] else len(st["uniq"]),
            "max_len": st["maxlen"],
        }
        if kind == "timestamp" and st["ts_min"] is not None:
            rep["min"] = st["ts_min"]
            rep["max"] = st["ts_max"]
        elif kind == "numeric" and st["min"] is not None:
            rep["min"] = st["min"]
            rep["max"] = st["max"]
            non_null = seen - st["nulls"]
            rep["mean"] = round(st["sum"] / max(non_null, 1), 4)
        else:
            rep["samples"] = st["samples"]
        col_reports.append(rep)

    return {
        "path": path,
        "size_mb": round(size / 1e6, 2),
        "delimiter": {"\t": "TAB"}.get(delim, delim),
        "n_cols": len(cols),
        "columns_truncated": len(cols) > max_cols,
        "rows_scanned": rows,
        "hit_row_cap": rows >= max_rows,
        "columns": col_reports,
    }


def guess_roles(rep):
    if "columns" not in rep:
        return {}
    roles = {"time": [], "match": [], "round": [], "player": [], "price": []}
    for c in rep["columns"]:
        n = c["name"].lower()
        if c["kind"] == "timestamp" or any(k in n for k in ("time", "_ts", "stamp", "date")):
            roles["time"].append(c["name"])
        if any(k in n for k in ("match", "demo", "game_id")):
            roles["match"].append(c["name"])
        if "round" in n:
            roles["round"].append(c["name"])
        if any(k in n for k in ("player", "steam", "user", "name")):
            roles["player"].append(c["name"])
        if any(k in n for k in ("price", "odds", "prob")):
            roles["price"].append(c["name"])
    return {k: v for k, v in roles.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--uniq-cap", type=int, default=5000)
    ap.add_argument("--json", action="store_true", help="JSON only")
    args = ap.parse_args()

    if os.path.isfile(args.root):
        files = [args.root]
    elif os.path.isdir(args.root):
        files = []
        for dirpath, dirs, names in os.walk(args.root):
            dirs[:] = [d for d in dirs if d not in
                       ("_out_of_scope", ".ipynb_checkpoints", "__pycache__", ".git")]
            for fn in sorted(names):
                if fn.endswith((".gz", ".csv")):
                    files.append(os.path.join(dirpath, fn))
    else:
        sys.exit(f"not a file or directory: {args.root}")
    if not files:
        sys.exit(f"no .gz/.csv files under {args.root}")

    print(f"# audit of {args.root}: {len(files)} file(s)")
    reports = []
    for path in files:
        if ".jsonl" in path:
            rep = profile_jsonl(path, args.max_rows)
        else:
            rep = profile(path, args.max_rows, args.uniq_cap)
        rep["roles"] = guess_roles(rep)
        reports.append(rep)
        if args.json:
            continue
        print(f"\n== {rep['path']}")
        if rep.get("format") == "jsonl":
            print(f"   JSONL size={rep['size_mb']} MB lines_scanned={rep['lines_scanned']}"
                  f"{' (CAP)' if rep['hit_row_cap'] else ''}")
            for c in rep["columns"]:
                extra = f"present={c['present_pct']}% e.g. {c['sample']}"[:120]
                print(f"   - {c['name'][:30]:<30} {c['kind']:<18} {extra}")
                if c.get("nested_keys"):
                    print(f"       nested: {c['nested_keys'][:12]}")
            continue
        if "error" in rep:
            print(f"   ERROR: {rep['error']}")
            continue
        print(f"   size={rep['size_mb']} MB  delim={rep['delimiter']}  cols={rep['n_cols']}"
              f"  rows_scanned={rep['rows_scanned']}{' (ROW CAP)' if rep['hit_row_cap'] else ''}")
        for c in rep["columns"]:
            if c["kind"] in ("numeric", "timestamp") and "min" in c:
                extra = f"min={c['min']} max={c['max']}"
                if "mean" in c:
                    extra += f" mean={c['mean']}"
            else:
                extra = f"e.g. {c['samples']}"
            print(f"   - {c['name']:<26} {c['kind']:<9} null={c['null_pct']:>5}% "
                  f"distinct={c['distinct']} {extra}"[:150])
        if rep["roles"]:
            print(f"   roles: {json.dumps(rep['roles'])}")

    print(f"\n# {len(reports)} file(s), "
          f"{round(sum(r.get('size_mb', 0) for r in reports), 2)} MB total")
    print("\n# --- paste everything below this line back ---")
    print(json.dumps(reports, indent=1, default=str))


if __name__ == "__main__":
    main()
