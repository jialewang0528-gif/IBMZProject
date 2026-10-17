# Handoff — CS2 Integrity Watch

Written 17 Oct 2026 for an agent picking this up cold. Read this first, then
`cs2-match-fixing-spec.md` (the product contract) and
`docs/implementation-plan.md` (the build contract).

---

## 1. What this is

A game-side integrity layer for Counter-Strike 2: it puts live market-implied
probability next to a game-state win-probability model on one clock, and ranks
matches where the market moved away from what the game state said — for a human
to investigate. **An alert is never a verdict.**

Deliverables: a hardened spec, a working pipeline + model + site, a public URL,
and a video. Time budget was 10 hours; it is largely spent.

---

## 2. Environment facts (measured, do not re-derive)

| Fact | Value |
|---|---|
| Host | LinuxONE instance `tylerinstance1`, public IP `148.100.79.57` |
| Container | `notebook`, image `registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x`, user `ibm-user` |
| Architecture | **s390x**, Python 3.12.15 |
| Installed | pandas 3.0.6, numpy 2.0.2, scikit-learn 1.9.0, **xgboost 3.4.1**, scipy, matplotlib, jinja2 |
| **Not** installed | flask, pyarrow, streamlit, shap, pytest, kubectl |
| Reachable ports from internet | **only 22 and 38888** (8000/8080/5000/9000 time out; verified from an outside machine) |
| Tunnels | impossible — cloudflared and ngrok ship no s390x build |
| Data | `/home/ibm-user/cs2_betting_data` **inside the container**, 485 MB, in the writable layer (lost if the container is recreated) |
| Repo on container | `/home/ibm-user/IBMZProject` |
| Disk | 35 GB free |

Consequences baked into the code: stdlib `http.server` (no Flask), no parquet
(use csv/gz), **the app owns 38888 and Jupyter moves to 127.0.0.1:8888**.

---

## 3. Data facts (measured)

- **Game data** provenance: HF `blanchon/opencs2_dataset_demo` (CC-BY-4.0),
  parsed with `demoparser2 0.42.0`. Tables: `round_table` 42,009 rows,
  `round_detail` 42,009, `kills` 286,671, `player_round_economy` 420,090,
  `util_bought` 341,258, `player_map_stats` 61,707, `maps_meta` 1,988 maps /
  **860 series**, `pistol_loss_opps` 3,960.
- **Market data**: Polymarket public APIs. 12,489 events / 117,318 markets;
  scope 860 Bo3 series Mar 23–Apr 20 2026; 1-minute price snapshots including
  in-play; every trade fill. `odds_table.csv.gz` ~963k Polymarket rows + 13k
  OddsPapi rows. **Polymarket prices are already probabilities — no margin.**
- **Join**: 860 game-side series ∩ **665** market-side = **665 series**;
  **1,284 maps across 654 series have both halves.** That is the demo pool.
  407 series have market data on every map played. A series-level join is not
  enough: 78 market `map_uid`s have no demo counterpart, so 11 series have no
  alignable map at all.
- **The old "297" was an artifact, not a measurement.**
  `tools/schema_report.py` unioned `match_id` (860) + `match_date` (860) + score
  tokens (3) into one set and printed its size as "key values" = 1,723; and it
  capped the odds scan at 400k rows, so its "intersection" came from a truncated
  slice = 297. Never quote 297 or 1,723. See spec §3.3.
- **No round-level markets exist** → analysis is at map level.
- `maps_meta.match_date` is the **series start**, identical for every map of a
  match. Map *k* start = series start + prior map durations + 600s breaks.
- It is ALSO **event-local time mislabelled as UTC**: measured against each
  map's market settlement, the game clock sits 1–3 whole hours ahead, clustering
  at −1 h (CET), −2 h (CEST), −3 h (MSK). Uncorrected the feeds never overlap and
  coverage is 0.00 on every map. The clock is now anchored per map on its market
  settlement (last print strictly between 0.02 and 0.98). This also makes the
  600s break assumption cancel out.
- `round_detail` clock columns are **64-tick** in this dataset, verified against
  the modal freezetime (1,280 = 20 s) and the 115 s + 40 s round ceiling. Mean
  round length cannot discriminate the rate: 7,000 units is 109 s at 64-tick and
  55 s at 128-tick. The rate is inferred dataset-wide, never assumed.

---

## 4. Measurement rule (changed during implementation — know this)

```
gap[r]      = market_p[r] − model_p[r]
baseline    = median(gap[first 3 rounds])          # this map's own offset
residual[r] = gap[r] − baseline                    # THIS is what alerts

alert requires: |residual| ≥ 0.15 for ≥ 2 consecutive rounds
                AND a market step ≥ 0.05 in the divergence direction
                    within the preceding 10 minutes
                AND no round ending within the 60s floor before the print
```

Why: a globally trained model carries a per-map offset (~20 points in fixtures),
so thresholding the absolute gap alerts on every round of every map. Residual
measures the actual claim: the market moving *away* from the game state.

**Thresholds (0.15 / 2 rounds) are provisional.** They were never calibrated on
the real 1,284 maps. Calibrating them is the most valuable remaining task; do not
quote a false-positive rate until it is done.

---

## 5. Code map

```
pipeline/io.py              table/odds resolution (root or tables/), caching
pipeline/align.py           clock unit inference, map→round timeline, as-of market
                            join, --validate (tickrate, orientation, market coverage)
pipeline/features.py        start-of-round features; asserts no leakage;
                            map_name deliberately excluded
pipeline/market.py          series extraction + quality gate (with reasons)
pipeline/detect.py          residual-gap alerting, persistence, step detector,
                            no-news filter, player within-match z-scores, --ablation
pipeline/inject.py          injection of panel values AND market prints, both flagged
pipeline/build_artifacts.py JSON contract (matches/series/summary), validation
models/train.py             split by match_id, logistic + xgboost, isotonic calibration
models/score.py             scoring + exact logit attribution (no SHAP on s390x)
server/app.py               stdlib ThreadingHTTPServer; /api/{matches,series,score,
                            health,summary}, /data/* static fallback, noindex headers
site/                       vanilla JS SPA, inline SVG chart, offline-capable
deploy/start_all.sh         takes 38888, moves Jupyter to 8888, idempotent
deploy/{Dockerfile,deploy.md}
tools/{audit_data,audit_cell,schema_report,make_fixture}.py
tests/test_pipeline.py      20 tests: pure rules, CLI acceptance, live server, packaging
```

Run everything from **inside the container**:

```bash
cd /home/ibm-user/IBMZProject
export CS2_DATA_ROOT=/home/ibm-user/cs2_betting_data
python3 -m unittest discover -s tests -v          # expect 20 OK
python3 -m pipeline.align --validate               # gate: coverage median ≈ 1.0
python3 -m models.train --cv                       # target Brier ≤ 0.21
python3 -m pipeline.build_artifacts --out site/data --inject-map <map_uid>
bash deploy/start_all.sh                           # http://148.100.79.57:38888/
```

There is **no docker CLI inside the container**; `docker exec` only works from
the host.

---

## 6. Traps already hit (do not repeat)

1. **A blanket `models/` line in `.gitignore` excluded the models package** from
   git; a fresh clone failed with `No module named models`. Now only
   `models/artifacts/` is ignored, and `TestRepoPackaging` fails if a promised
   file is untracked.
2. **`maps_meta.csv.gz` lives in `tables/`**, not the data root. `io.py` resolves
   candidate paths and reports what it looked for.
3. **`match_date` is the series start.** Using it as the map start misaligns map
   2+ by an entire map and the market join coverage collapses to ~0.
4. **Absolute-gap alerting produces constant false positives** (see §4).
5. **`map_name` as a feature memorises the per-map label.** Brier went 0.38 → 0.04
   after dropping it.
6. **Timezone trap:** writing local-time timestamps labelled `Z` shifted every
   market print by 7 hours. The fixture got it wrong once; the coverage check
   now catches it.
7. **Nested quoting** in shell one-liners left the terminal in a heredoc state
   and swallowed commands. Use scripts, not nested quotes.

---

## 7. State at handoff

- 20 tests pass **on the container** (`Ran 20 tests ... OK`).
- Pipeline, model, server, site, deploy scripts all verified locally against
  `tools/make_fixture.py` (40 synthetic maps): Brier 0.041, attribution
  reconstructs the logit to 1e-9, injected divergence detected end to end,
  server serves all routes, static fallback works.
- **Not yet run against the real 1,284 maps.** The blocking input is the output of
  `python3 -m pipeline.align --validate` (tickrate resolved / orientation ok /
  market coverage median). Without it, nothing downstream on real data is proven.

### What to do first

1. Get `--validate` output. If coverage median ≈ 1.0 and all maps resolve, proceed.
   If a map reports `clock unit unresolved`, the message names all three
   interpretations — fix the inference, don't tune thresholds.
2. `models.train --cv` on real data. If Brier > 0.21, reduce features (economy +
   score only) rather than shipping an uncalibrated model.
3. `build_artifacts` over every aligned map, pick a demo map, inject rounds 14–16.
4. `deploy/start_all.sh`, then verify from outside: `curl 148.100.79.57:38888/api/health`
   must show `"arch": "s390x"` and an `n_maps` equal to the number of maps in
   `site/data/matches.json` (not a hardcoded pool size).
5. Record the video (4 min: problem → architecture → replay with injection firing
   → alert with contributions → limits).

---

## 8. Open items

| Item | Owner | Note |
|---|---|---|
| Threshold calibration on 1,284 real maps | next agent | highest-value task; provisional 0.15/2-rounds |
| Video format/length | organiser rules | unknown; plan assumes 4 min |
| `/home/ibm-user/.git-credentials` holds a plaintext GitHub PAT | human | delete before serving publicly; the app is unauthenticated by design |
| Data is in the container's writable layer | human | copy to `/home/linux1/shared` (bind mount) before hour 6 or lose it |
| NOMERCY retrospective (Mar 28 2026, 4 series + 33 controls) | later | not on the critical path |
| Polymarket terms of use | nobody | not reviewed; blocks publishing raw derived series |
| Second market source (OddsPapi) | later | history only from ~Apr 10 2026 |

---

## 9. Decisions the human made (do not re-litigate)

- Real player names are permitted on the public site, with **no access gate**.
  Mitigated in the UI: no allegation language anywhere ("divergence score",
  never "fixed"), verbatim disclaimer on every alert card, `noindex`, and a
  banner stating part of the feed is simulated and no finding is made about any
  person. `test_no_allegation_language_in_site` enforces the vocabulary.
- Deliverable is a spec plus a demo video; the site is the artifact judges click.
- Submission is a video; the site must survive without a presenter.
- Solo builder, parallel agents — hence the file-scoped task list and frozen
  JSON contract in `docs/implementation-plan.md`.

---

## 10. Honest limitations to state in any pitch

- 1,284 maps across 654 series have both data halves; ~80% of them clock-align.
- 60-second market resolution plus map-start error: ordering below that is not
  measurable.
- No round-level markets → map-level analysis only.
- One retrospective case (NOMERCY) is not a validation study.
- Part of the demo's market feed is injected and labelled as such.
- An alert is not a verdict.
