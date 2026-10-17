# CS2 Integrity Watch

Game-side integrity layer for Counter-Strike 2: it puts live market-implied
probability next to a game-state win-probability model on one clock, and ranks
matches where the market moved away from what the game said — for a human to
investigate. **An alert is not a verdict.**

| Document | Purpose |
|---|---|
| [`cs2-match-fixing-spec.md`](cs2-match-fixing-spec.md) | the product spec: scope, measurement rule, what the demo proves and does not |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | module spec, frozen JSON contracts, task list, deployment |
| [`deploy/deploy.md`](deploy/deploy.md) | runbook: ports, recovery after a container restart, security notes |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | measured environment facts, traps already hit, state at handoff, what to do first |

## What is in here

```
pipeline/   io.py  align.py  features.py  market.py  detect.py  inject.py  build_artifacts.py
models/     train.py  score.py            (logistic for alerting, xgboost as benchmark)
server/     app.py                        (stdlib http.server — no Flask on s390x)
site/       index.html  static/app.js  static/style.css   (vanilla JS, offline-capable)
deploy/     start_all.sh  Dockerfile  deploy.md
tools/      audit_data.py  audit_cell.py  schema_report.py  make_fixture.py
tests/      test_pipeline.py              (16 tests, `python -m unittest discover -s tests`)
```

## Run it

```bash
export CS2_DATA_ROOT=/home/ibm-user/cs2_betting_data

python -m pipeline.align --validate                 # clock + team orientation
python -m models.train --cv                         # fit + calibrate, write artifacts
python -m pipeline.build_artifacts --out site/data --inject-map <map_uid>
python -m server.app --port 38888                   # http://host:38888/
```

No data or model on hand? Generate a synthetic dataset that mirrors the real
schemas (this is what the tests use):

```bash
python tools/make_fixture.py --out /tmp/fixture --maps 40
export CS2_DATA_ROOT=/tmp/fixture CS2_ARTIFACT_DIR=/tmp/artifacts
python -m models.train --cv
python -m pipeline.build_artifacts --out site/data --inject-map 2392432-2
python -m server.app --port 38888
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Data sources

- Game data: `blanchon/opencs2_dataset_demo` (CC-BY-4.0), parsed with `demoparser2`.
- Market data: Polymarket public APIs (1-minute prices, every fill, in-play).
- Data lives outside this repo; `.gitignore` blocks `*.gz`, `*.dem`, `site/data/`,
  `models/artifacts/`.

## Limits, stated up front

- Market prices are 1-minute snapshots: ordering below ~60s plus map-start error
  is not measurable.
- No round-level markets exist, so analysis is at map level.
- Alert thresholds are provisional until calibrated on the real 297-map set.
- Part of the market feed in the demo is injected and labelled as such.
