import gzip, io, json, os, re

ROOT     = "/cs2_betting_data"
MAX_ROWS = 100_000
UNIQ_CAP = 5_000

TS_RE  = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?|\d{10}(\.\d+)?|\d{13})$")
NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")

def _open(p):
    return gzip.open(p, "rt", errors="replace", newline="") if p.endswith(".gz") \
           else open(p, "r", errors="replace", newline="")

def _delim(hdr):
    c = {d: hdr.count(d) for d in [",", "\t", ";", "|"]}
    b = max(c, key=c.get)
    return b if c[b] else ","

def profile(p):
    with _open(p) as fh:
        hdr = fh.readline().rstrip("\r\n")
        if not hdr:
            return {"path": p, "error": "empty"}
        d    = _delim(hdr)
        cols = [c.strip() for c in hdr.split(d)]
        st   = [dict(n=0, num=0, txt=0, ts=0, uniq=set(), over=False,
                     lo=None, hi=None, tot=0.0, tslo=None, tshi=None, samp=[], mlen=0)
                for _ in cols]
        rows = 0
        for line in fh:
            if rows >= MAX_ROWS:
                break
            rows += 1
            parts = line.rstrip("\r\n").split(d)
            for i in range(len(cols)):
                s   = st[i]
                val = parts[i] if i < len(parts) else ""
                if val == "" or val.upper() in ("NA", "NAN", "NULL", "NONE", "-"):
                    s["n"] += 1
                    continue
                s["mlen"] = max(s["mlen"], len(val))
                if not s["over"]:
                    s["uniq"].add(val)
                    if len(s["uniq"]) > UNIQ_CAP:
                        s["over"], s["uniq"] = True, set()
                if TS_RE.match(val):
                    s["ts"] += 1
                    s["tslo"] = val if s["tslo"] is None else min(s["tslo"], val)
                    s["tshi"] = val if s["tshi"] is None else max(s["tshi"], val)
                if NUM_RE.match(val):
                    s["num"] += 1
                    v = float(val)
                    s["tot"] += v
                    s["lo"] = v if s["lo"] is None else min(s["lo"], v)
                    s["hi"] = v if s["hi"] is None else max(s["hi"], v)
                else:
                    s["txt"] += 1
                    if len(s["samp"]) < 3 and val not in s["samp"]:
                        s["samp"].append(val[:38])

    out = []
    for name, s in zip(cols, st):
        seen = max(s["n"] + s["num"] + s["txt"], 1)
        nn   = seen - s["n"]
        kind = ("timestamp" if nn and s["ts"] == nn else
                "numeric"   if nn and s["txt"] == 0 else
                "string"    if nn and s["num"] == 0 else "mixed")
        r = dict(name=name, kind=kind, null_pct=round(100 * s["n"] / seen, 1),
                 distinct=None if s["over"] else len(s["uniq"]), maxlen=s["mlen"])
        if kind == "timestamp" and s["tslo"]:
            r["min"], r["max"] = s["tslo"], s["tshi"]
        elif kind == "numeric" and s["lo"] is not None:
            r["min"], r["max"] = s["lo"], s["hi"]
            r["mean"] = round(s["tot"] / max(nn, 1), 4)
        else:
            r["samples"] = s["samp"]
        out.append(r)

    return dict(path=p, size_mb=round(os.path.getsize(p) / 1e6, 2),
                delim={" " * 0 + "\t": "TAB"}.get(d, d), n_cols=len(cols),
                rows_scanned=rows, hit_cap=rows >= MAX_ROWS, columns=out)

files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(ROOT)
               for f in fs if f.endswith((".gz", ".csv")))
print(f"{len(files)} file(s) under {ROOT}\n")
reports = []
for p in files:
    r = profile(p)
    reports.append(r)
    print(f"== {r['path']}  |  {r.get('size_mb')} MB  |  {r.get('n_cols')} cols  "
          f"|  {r.get('rows_scanned')} rows{' (CAPPED)' if r.get('hit_cap') else ''}")
    for c in r.get("columns", []):
        if "min" in c:
            extra = f"{c['min']} .. {c['max']}" + (f" mean={c['mean']}" if "mean" in c else "")
        else:
            extra = f"e.g. {c['samples']}"
        print(f"   {c['name'][:24]:<24} {c['kind']:<9} null={c['null_pct']:>5}% "
              f"distinct={c['distinct']} {extra}"[:140])
    print()

print("PLACEHOLDER_JSON")
