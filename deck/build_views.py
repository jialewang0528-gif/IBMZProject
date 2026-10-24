"""Write the Mermaid architecture page and the detect.py code view as HTML."""
import sys
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

out = sys.argv[1]

MERMAID = r"""flowchart LR
  G["<b>Game data</b><br/>pro demos → rounds,<br/>economy, kills"] --> AL
  M["<b>Market data</b><br/>Polymarket + OddsPapi,<br/>1-min prices, every fill"] --> AL
  AL["<b>Alignment</b><br/>one clock, 64-tick,<br/>anchored on settlement"]
  AL --> A["<b>A · Win-probability model</b><br/>logistic regression<br/>xgboost benchmark"]
  AL --> B["<b>B · Market probability</b><br/>map-winner price"]
  AL --> C["<b>C · Player self-deviation</b><br/>corroborating only"]
  AL --> D["<b>D · Market anomaly</b><br/>price steps, trade size,<br/>no-news filter"]
  F["<b>F · Injection harness</b><br/>labelled simulated prints"] -.-> B
  A --> E
  B --> E
  C --> E
  D --> E
  E["<b>E · Fusion</b><br/>residual ≥ 15 pts for 2+ rounds<br/>market moved first<br/>no game event explains it"] --> X["<b>Alert card</b><br/>exact top-3 factors"]
  X --> H(["Human analyst decides"])
  classDef alert stroke:#f85149,color:#ffdcd7
  class E,X alert
"""

page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.5.1/github-markdown-dark.css">
<style>body{{background:#0d1117;margin:0;padding:28px 36px}} .markdown-body{{max-width:none}} pre.mermaid{{background:#0d1117;border:0}}</style>
</head><body><article class="markdown-body">
<h2>Architecture</h2>
<p>Every component runs inside the s390x container. Data never enters the repo.</p>
<pre class="mermaid">{MERMAID}</pre>
</article>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad:true, theme:'dark', flowchart:{{curve:'basis', htmlLabels:true}},
  themeVariables:{{fontFamily:'-apple-system,Segoe UI,Helvetica,Arial,sans-serif', fontSize:'15px',
  primaryColor:'#161b22', primaryBorderColor:'#30363d', primaryTextColor:'#e6edf3', lineColor:'#8b949e'}}}});
</script></body></html>"""
open(f"{out}/architecture.html", "w").write(page)

CONSTS = """GAP_THRESHOLD = 0.15
PERSISTENCE_ROUNDS = 2
STEP_WINDOW_MIN = 10
STEP_THRESHOLD = 0.05
FLOOR_S = 60
"""
BODY = '''    if panel.empty or "model_p" not in panel.columns:
        return []
    panel = panel.reset_index(drop=True).copy()
    panel["gap"] = gap_series(panel)
    panel["residual_gap"] = residual_gap(panel)
    spans = persistence(panel["residual_gap"], thresh, k)
    steps = step_detector(series, thresh=step_thresh) if series is not None else []
    out = []
    for start, end in spans:
        row = panel.iloc[start]
        t_start = pd.to_datetime(row["t_start_utc"], utc=True)
        resid = float(row["residual_gap"])
        moved_first = any(
            (pd.Timestamp(st["ts_utc"]) <= t_start)
            and (pd.Timestamp(st["ts_utc"]) >= t_start - pd.Timedelta(minutes=STEP_WINDOW_MIN))
            and (np.sign(st["delta"]) == np.sign(resid))
            for st in steps
        )
        t_market = row.get("market_ts_utc")
        clean = no_news(panel, pd.Timestamp(t_market)) if pd.notna(t_market) else True
        if not (moved_first and clean):
            # a divergence without a market move, or one explained by a round
            # ending inside the floor, is not an alert under the stated rule
            continue
'''


def block(code, start):
    fmt = HtmlFormatter(style="github-dark", linenos="table", linenostart=start, nowrap=False)
    return highlight(code, PythonLexer(), fmt), fmt.get_style_defs(".highlight")


c1, css = block(CONSTS, 21)
c2, _ = block(BODY, 152)
code = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{background:#0d1117;margin:0;padding:24px;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6edf3}}
.box{{border:1px solid #30363d;border-radius:6px;overflow:hidden}}
.bar{{background:#161b22;border-bottom:1px solid #30363d;padding:10px 16px;color:#e6edf3;font:600 14px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
.bar span{{color:#8b949e;font-weight:400;margin-left:10px}}
.gap{{color:#8b949e;padding:4px 16px;border-top:1px solid #21262d;border-bottom:1px solid #21262d;background:#0d1117}}
{css}
.highlight{{background:#0d1117}} .highlight pre{{margin:0;padding:0 12px;font-size:13px;line-height:20px}}
.linenodiv pre, td.linenos pre{{color:#6e7681;text-align:right;padding:0 10px 0 16px;margin:0;line-height:20px;font-size:13px}}
table.highlighttable{{border-collapse:collapse;width:100%}} td{{padding:0;vertical-align:top}}
</style></head><body><div class="box">
<div class="bar">pipeline/detect.py<span>the alert rule, lines 21–25 and 152–175</span></div>
{c1}<div class="gap">⋯</div>{c2}
</div></body></html>"""
open(f"{out}/code_detect.html", "w").write(code)
print("ok")
