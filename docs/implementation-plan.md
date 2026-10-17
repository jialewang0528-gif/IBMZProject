# CS2 Integrity Watch — Implementation Plan

Oct 17, 2026 · companion to `cs2-match-fixing-spec.md` v2

**What this document is:** the module specification the build follows, written so that independent agents (or one person switching hats) can work in parallel without merging into each other. Every column name below is taken from the real tables; every task has an acceptance command that can fail.

**What it is not:** a schedule. Hours are estimates for ordering, not commitments.

**Frozen-decision notice:** §1.3 fixes the deployment shape and §3 freezes the JSON contracts. Those two sections must not be edited while tasks are in flight — every other section can.

---

## 1. System shape

### 1.1 Request path

```
browser ──► 0.0.0.0:38888 (Flask, public)
                ├── /                     static SPA shell (index.html + app.js)
                ├── /api/matches          precomputed list of 297 maps        [cache]
                ├── /api/series/<map_uid> precomputed round panel + alerts    [cache]
                ├── /api/score            POST round state → live model inference
                └── /api/health           arch, uname, models loaded, uptime
```

### 1.2 Build path (offline, in the container)

```
cs2_betting_data/{tables/*.csv.gz, odds_table.csv.gz, maps_meta}
        │
        ├─ pipeline/align.py      → panel: one row per (map_uid, round_num) with wall clock
        ├─ pipeline/features.py   → start-of-round feature matrix + label
        ├─ models/train.py        → logistic (alerting) + xgboost (benchmark), calibrated
        ├─ models/score.py        → P(team1 wins map) per round
        ├─ pipeline/market.py     → market series per map with as-of join
        ├─ pipeline/detect.py     → gap series, persistence, D detectors, C z-scores
        ├─ pipeline/inject.py     → synthetic prints at chosen rounds, flagged
        └─ pipeline/build_artifacts.py → site/data/*.json  (also the static fallback bundle)
```

### 1.3 Deployment (frozen)

| Item | Decision | Reason |
|---|---|---|
| Substrate | plain Docker container `notebook` on `148.100.79.57` | `kubectl` absent; K8s adds nothing visible |
| HTTP stack | Python stdlib `http.server` (ThreadingHTTPServer) | no Flask wheel for s390x, no PyArrow; zero dependencies beats one apt install |
| Public port | **`0.0.0.0:38888` — the app owns it** | probe: only 22 and 38888 are reachable from the internet; 8000/8080/5000/9000 timed out |
| Jupyter | moved to `127.0.0.1:8888`, reached via `ssh -L 8888:localhost:8888` | frees 38888; avoids unauthenticated JupyterLab, which would expose `/home/ibm-user/.git-credentials` |
| Fallback | `site/` is a complete static bundle; if the API is unreachable the page renders from `site/data/*.json` with a "static mode" banner | covers container death; also the artifact for Cloudflare Pages later |
| Public posture | real names, no gate (decided) | mitigate in UI: no allegation language, simulation banner, `noindex` |
| Auth | none on public routes; organizer mode is not gated (decided) | recorded as accepted risk |

**Port ownership risk and its mitigation:** Jupyter is started by the image entrypoint. If the container restarts, Jupyter reclaims 38888 and the site dies. `deploy/start_all.sh` is the recovery path and is run after any restart:

```bash
docker exec -d notebook /home/ibm-user/IBMZProject/deploy/start_all.sh
```

`start_all.sh` kills any listener on 38888, starts Jupyter on `127.0.0.1:8888`, starts Flask on `0.0.0.0:38888`, then polls `/api/health` until it answers. It is idempotent.

---

## 2. Data contracts (inputs)

### 2.1 Tables consumed (exact columns)

| Table | Columns used | Used by |
|---|---|---|
| `tables/maps_meta.csv.gz` | `map_uid, match_id, map_index, event, team1, team2, map_name, match_date, series_score1, series_score2, map_winner_team_resolved, rounds_played, hltv_stars, analysis_eligible_v1, qc_status` | align, features, scoring |
| `tables/round_table.csv.gz` | `map_uid, match_id, map_index, round_num, winner_side, winner_team, bomb_planted, ct_team_name_resolved, t_team_name_resolved, ct_team_balance, t_team_balance, ct_team_spend, t_team_spend, ct_team_current_equip_value, t_team_current_equip_value, ct_armor_players, ct_helmet_players, ct_defuser_players, t_armor_players, t_helmet_players, t_defuser_players, ct_buy, t_buy, equip_diff_ct_minus_t` | features |
| `tables/round_detail.csv.gz` | `map_uid, round_num, start, freeze_end, end, official_end, winner, reason, bomb_plant, duration_ticks` | align (clock) |
| `tables/player_round_economy.csv.gz` | `map_uid, round_num, steamid, player_name, side, roster_uid, team_name_resolved, start_balance, balance, cash_spent_this_round, current_equip_value, armor_value` | C |
| `tables/kills.csv.gz` | `map_uid, tick, round_num, attacker_steamid, victim_steamid, attacker_side, weapon, headshot, dmg_health` | C |
| `tables/player_map_stats.csv.gz` | `map_uid, steamid, name, side, n_rounds, rating, kast, adr, kills, deaths, roster_uid` | C |
| `odds_table.csv.gz` | `match_id, map_index, market, selection, fair_prob, timestamp_utc, is_live, map_uid, source, line` | market |

### 2.2 Orientation (must be verified once, in T1)

`maps_meta.team1` / `team2` are the reference for "team1 wins the map". `round_table` names sides `ct`/`t` by team name. Therefore:

```
team1_is_ct(map, round) = (round_table.ct_team_name_resolved == maps_meta.team1)
label(map, round)       = 1 if team1 wins the map else 0     # from map_winner_team_resolved
```

The market side already guarantees `selection=team1` means `maps_meta.team1` (verified upstream: 664/664 match winners agree with HLTV results). **T1 must assert these agree on at least 20 maps, or every downstream number is wrong.**

### 2.3 Clock model

```
map_start_utc(map)  = maps_meta.match_date                      # tz-aware, anchor
round_start_utc(r)  = map_start_utc + round_detail.start / tickrate
tickrate           ∈ {64, 128}                                  # detect in T1
map_start(map k)   = series_start + Σ(duration + 600s) for maps 1..k-1
                     # match_date is the SERIES start (all maps share it)
resolution_floor_s  = 60 + map_start_error_s                    # displayed on every alert
```

`tickrate` is detected, not assumed: pick the value whose implied round durations match `round_detail.duration_ticks` when compared against the median observed live-market movement rate. If neither is consistent within 15%, the map is excluded from the demo pool and logged.

---

## 3. Frozen JSON contracts

These three shapes are the only interface between the build half and the serving half. **Do not change them mid-build.** Additive optional fields are allowed; renames are not.

### 3.1 `site/data/matches.json`

```json
[{
  "map_uid": "2392432-1",
  "match_id": 2392432,
  "match_date": "2026-03-23T11:04:09-07:00",
  "event": "PGL Astana 2026 Europe Open Qualifier 1",
  "map_name": "de_overpass",
  "team1": "magic",
  "team2": "Inner Circle Academy",
  "n_rounds": 22,
  "hltv_stars": 0,
  "max_gap": 0.29,
  "severity": "high",
  "n_alerts": 1,
  "has_injection": true,
  "injected_rounds": [14, 15, 16],
  "resolution_floor_s": 60
}]
```

### 3.2 `site/data/series/<map_uid>.json`

```json
{
  "meta": { "map_uid": "2392432-1", "match_id": 2392432, "event": "...", "map_name": "de_overpass",
            "team1": "magic", "team2": "Inner Circle Academy", "map_start_utc": "...",
            "tickrate": 64, "resolution_floor_s": 60, "model": "logit-v1", "generated_utc": "..." },
  "rounds": [{
      "round_num": 14,
      "t_start_utc": "2026-03-23T18:31:10Z",
      "score_team1": 7, "score_team2": 6,
      "team1_side": "ct",
      "alive_team1": 5, "alive_team2": 5,
      "equip_diff_team1_minus_team2": 1250.0,
      "buy_team1": "full", "buy_team2": "semi",
      "model_p": 0.43,
      "market_p": 0.71,
      "gap": 0.28,
      "market_ts_utc": "2026-03-23T18:29:44Z",
      "market_ts_delta_s": -86,
      "injected": false
  }],
  "alerts": [{
      "round_num": 14, "span": [14, 16], "gap": 0.28, "severity": "high",
      "criteria": { "gap_ge_15": true, "persistence_rounds": 3,
                    "market_moved_first": true, "no_explaining_event": true },
      "contributions": [ { "name": "equip_diff", "value": -1250.0, "contribution": -0.081 },
                         { "name": "alive_diff", "value": 0, "contribution": 0.0 } ],
      "floor_s": 60,
      "disclaimer": "Divergence signal only. Not a finding about any person."
  }],
  "players": [ { "rank": 1, "steamid": "76561198...", "name": "MaSvAl", "side": "ct",
                 "rounds_flagged": 3, "z_mean": -2.4, "note": "within-match self-deviation" } ]
}
```

### 3.3 `POST /api/score`

Request:

```json
{ "round_state": { "score_team1": 7, "score_team2": 6, "round_num": 14, "team1_side": "ct",
                   "alive_team1": 5, "alive_team2": 5, "equip_team1": 12500, "equip_team2": 11250,
                   "buy_team1": "full", "buy_team2": "semi", "map_name": "de_overpass" } }
```

Response:

```json
{ "p_logit": 0.43, "p_xgb": 0.39, "model": "logit-v1",
  "contributions": [ { "name": "equip_diff", "value": -1250.0, "contribution": -0.081 } ],
  "latency_ms": 3.1, "arch": "s390x", "hostname": "tylerinstance1", "python": "3.12.15" }
```

### 3.4 `GET /api/health`

```json
{ "ok": true, "arch": "s390x", "uname": "Linux ... s390x", "python": "3.12.15",
  "models_loaded": ["logit-v1", "xgb-v1"], "n_maps": 297, "uptime_s": 412 }
```

---

## 4. Module specification

Rules for every module: no I/O at import time, no global mutable state, no printing (return values or log), every function accepts an explicit path/DataFrame, no hidden reads of `/home/ibm-user/cs2_betting_data` outside `pipeline/io.py`.

### 4.1 `pipeline/io.py`

| Function | Signature | Notes |
|---|---|---|
| `read_table` | `read_table(name: str, root: Path = DATA_ROOT) -> pd.DataFrame` | caches; `name` is `round_table`, `round_detail`, `maps_meta`, `kills`, `player_round_economy`, `player_map_stats` |
| `read_odds` | `read_odds(root: Path = DATA_ROOT) -> pd.DataFrame` | parses `timestamp_utc` to UTC |
| `eligible_maps` | `eligible_maps() -> pd.DataFrame` | `maps_meta` ∩ `odds_table.match_id`, `analysis_eligible_v1 == True`, `qc_status == "pass"` |
| `DATA_ROOT` | `Path("/home/ibm-user/cs2_betting_data")`, overridable by env `CS2_DATA_ROOT` | |

**Acceptance:** `python -m pipeline.io` prints table row counts + eligible map count, and `eligible_maps()` returns ≥ 200 map rows.

### 4.2 `pipeline/align.py`

| Function | Signature | Notes |
|---|---|---|
| `detect_tickrate` | `detect_tickrate(map_uid: str) -> int` | 64 or 128, see §2.3 |
| `map_panel` | `map_panel(map_uid: str) -> pd.DataFrame` | one row per round: `round_num, t_start_utc, t_end_utc, tickrate, map_start_utc, floor_s` |
| `attach_market` | `attach_market(panel, odds) -> pd.DataFrame` | as-of join: for each round, the last market print with `ts <= t_start_utc` **and** `ts > t_start_utc - 600s`; adds `market_p, market_ts_utc, market_ts_delta_s` |

**Acceptance:** `python -m pipeline.align --map-uid <uid>` prints the panel with strictly increasing `t_start_utc`, no duplicate `round_num`, and `market_ts_delta_s` in `[-600, 0]` for every row with a market value. Fails loudly if tickrate detection is ambiguous.

### 4.3 `pipeline/features.py`

Features are **start-of-round only** — leakage is the main risk here.

```
score_team1, score_team2, round_num, is_pistol (round in {1,13}), is_second_half (>=13)
team1_side_is_ct (bool), map_name (categorical)
equip_team1, equip_team2, equip_diff = equip_team1 - equip_team2
balance_diff, spend_diff
armor_diff, helmet_diff, defuser_diff  (team1 - team2 counts)
buy_team1, buy_team2 (categorical: eco|semi|full|force)
hltv_stars (0..2+), map_index
```

| Function | Signature |
|---|---|
| `feature_matrix` | `feature_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]` → (X, y), `y` = 1 if `maps_meta.team1` wins the map |
| `FEATURE_NAMES` | list, order stable, asserted in tests |

**Acceptance:** `python -m pipeline.features --map-uid <uid>` prints `X.shape`, `y.mean()`, null counts (all zero), and asserts that no column derived from `winner_side`, `winner_team`, `end_reason`, or `duration_ticks` is in `X`.

### 4.4 `models/train.py`

| Function | Signature | Notes |
|---|---|---|
| `split` | `split(maps: list[str], seed=7) -> tuple[list, list]` | split **by match_id**, never by round |
| `fit_logit` | `fit_logit(X, y) -> sklearn Pipeline` | `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| `fit_xgb` | `fit_xgb(X, y) -> xgb.XGBClassifier` | `max_depth=4, n_estimators=300, learning_rate=0.05` |
| `calibrate` | `calibrate(model, X, y) -> model` | `CalibratedClassifierCV(method="isotonic", cv=3)` on the training split only |
| `evaluate` | `evaluate(model, X, y) -> dict` | `{"brier": float, "logloss": float, "n": int}` |
| `save` / `load` | `save(model, path)` / `load(path)` | `joblib` to `models/artifacts/` |

**Acceptance:** `python -m models.train --cv` prints holdout Brier for logit and xgb, both ≤ 0.21, and writes `models/artifacts/logit-v1.joblib` + `xgb-v1.joblib` + `metrics.json`. Baseline to beat is 0.25 (coin flip) and the market's own 0.202 closing Brier.

### 4.5 `models/score.py`

| Function | Signature | Notes |
|---|---|---|
| `score_panel` | `score_panel(X: pd.DataFrame) -> np.ndarray` | loads models once (module-level cache, lazily) |
| `contributions` | `contributions(X_row: pd.Series) -> list[dict]` | exact logit attribution: `coef[i] * scaled_x[i]`, sorted by `abs()` desc, top 5 |
| `score_round_state` | `score_round_state(dict) -> dict` | builds a one-row frame matching `FEATURE_NAMES`; used by `/api/score` |

**Acceptance:** `python -m models.score --map-uid <uid>` prints model `p` for the first 5 rounds; `contributions()` on one row sums to within 1e-9 of `logit - intercept`.

### 4.6 `pipeline/market.py`

| Function | Signature | Notes |
|---|---|---|
| `market_series` | `market_series(map_uid: str, odds: pd.DataFrame) -> pd.DataFrame` | `ts_utc, market_p, source, is_live`; `market_p` = `fair_prob` for `selection=team1` on `market=map_winner` for that `map_index` |
| `series_quality` | `series_quality(series) -> dict` | `n_prints, n_live_prints, span_hours, max_gap_seconds, usable: bool` |

**Acceptance:** `python -m pipeline.market --map-uid <uid>` prints the quality dict; a map with `usable: false` is excluded from artifacts with a reason.

### 4.7 `pipeline/detect.py`

| Function | Signature | Notes |
|---|---|---|
| `gap_series` | `gap_series(panel) -> pd.DataFrame` | `gap = market_p - model_p` |
| `persistence` | `persistence(gap: pd.Series, thresh=0.15, k=2) -> list[tuple[int,int]]` | runs of ≥ k consecutive rounds above threshold |
| `step_detector` | `step_detector(series, window_min=10, thresh=0.05) -> list[dict]` | changes ≥ 0.05 within 10 min |
| `size_anomaly` | `size_anomaly(series) -> list[dict]` | print size z-score ≥ 3 vs the map's own distribution |
| `no_news` | `no_news(alert_round, panel) -> bool` | no round win/plant/multi-kill within `floor_s` before the market move |
| `player_deviation` | `player_deviation(map_uid) -> pd.DataFrame` | per-player per-round z of `kills`, `dmg`, `equip` vs own match mean; only rounds inside alert spans are reported |
| `alerts` | `alerts(panel) -> list[dict]` | applies §4.2 of the spec; emits contract §3.2 `alerts[]` |

**Acceptance:** `python -m pipeline.detect --map-uid <uid>` prints alert count and, for each alert, the four criteria booleans. On the injected map it must produce ≥ 1 alert; on 10 random non-injected maps it must produce 0 alerts (assert this, it is the false-positive guard).

### 4.8 `pipeline/inject.py`

| Function | Signature | Notes |
|---|---|---|
| `inject` | `inject(panel, rounds: list[int], delta=-0.25, rng=None) -> pd.DataFrame` | adds rows to the market series with `injected=True`, sizes sampled from that map's own print-size distribution, timestamps placed 60–180s before the round boundary |
| `INJECTION_NOTE` | constant string used verbatim in artifacts and UI | "injected prints R14–R16 (simulated)" |

**Acceptance:** `python -m pipeline.inject --map-uid <uid> --rounds 14,15,16` writes a CSV where injected rows are marked, and the un-injected rows are byte-identical to the input.

### 4.9 `pipeline/build_artifacts.py`

| Function | Signature | Notes |
|---|---|---|
| `build` | `build(out_dir: Path, limit: int | None = None, inject_map: str | None = None) -> dict` | writes `matches.json`, `series/<map_uid>.json`, `summary.json` (counts, model metrics, floor) |
| `main` | CLI: `--out site/data --limit 297 --inject-map <map_uid>` | |

**Acceptance:** `python -m pipeline.build_artifacts --out site/data` produces a valid bundle: `site/data/matches.json` parses, has ≥ 200 entries, every `series/<map_uid>.json` referenced exists and matches the contract, and total size < 20 MB. `python -m json.tool` on each file is part of the check.

### 4.10 `server/app.py`

| Route | Behaviour |
|---|---|
| `GET /` | serves `site/index.html` |
| `GET /api/matches` | `matches.json` (from memory, loaded at boot) |
| `GET /api/series/<map_uid>` | `series/<map_uid>.json`; 404 with JSON error if absent |
| `POST /api/score` | live inference via `models.score.score_round_state`; returns `p_logit`, `p_xgb`, `contributions`, `latency_ms`, `arch`, `hostname`, `python` |
| `GET /api/health` | `{ok, arch, uname, python, models_loaded, n_maps, uptime_s}` |
| `GET /static/*` | assets |
| `GET /data/*` | the static fallback bundle (same files the API serves) |

Rules: bind `0.0.0.0:38888`; `threaded=True`; no debug in demo mode; 5s timeout on model load at boot; if artifacts are missing, serve a JSON 503 explaining how to build them (never a stack trace).

**Acceptance:** with the server running, `curl -s localhost:38888/api/health | python -m json.tool` shows `"arch": "s390x"` and `"n_maps": 297`; `curl -s -X POST localhost:38888/api/score -d '<§3.3 body>' -H 'Content-Type: application/json'` returns `p_logit` in `(0,1)` and `latency_ms` < 50.

### 4.11 `site/index.html` + `site/app.js`

Vanilla JS, no framework, no build step, no CDN dependency (offline-capable). Panels: clock/floor badge, round state, dual-curve chart (inline SVG), gap bars with ±15 lines, player ranks, alert card, health footer showing `arch` from `/api/health`.

Rules: every `fetch` has a 3s timeout and falls back to the bundled JSON with a "static mode" banner; the string "fixed" never appears in the UI; the disclaimer from §3.2 is rendered verbatim on every alert card; `<meta name="robots" content="noindex,nofollow">`; a top banner states that part of the market feed is simulated and no finding is made about any person.

**Acceptance:** open the URL, click the injected map, see the alert fire with contributions; kill the API (`docker exec notebook pkill -f "server.app"`) and reload — page still renders with the static banner.

### 4.12 `deploy/`

| File | Purpose |
|---|---|
| `start_all.sh` | idempotent: free 38888, start Jupyter on 127.0.0.1:8888, start Flask on 0.0.0.0:38888, poll `/api/health` 30×1s |
| `Dockerfile` | `FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest`, `pip install flask joblib`, copy `server/`, `site/`, `models/`, `pipeline/`, `CMD ["deploy/start_all.sh"]` |
| `deploy.md` | the three commands to run, the SSH tunnel command, the recovery command, and the security note (no `.git-credentials` in the image; use a build arg or a mounted key) |

**Acceptance:** from a clean shell, `docker exec -d notebook /home/ibm-user/IBMZProject/deploy/start_all.sh` then `curl -s 148.100.79.57:38888/api/health` from an outside machine returns 200 within 30s.

---

## 5. Task list (agent-parallel)

Rules: one agent per task; **a task may only touch files listed in its scope**; contracts in §3 are frozen; a task is done only when its acceptance command passes and the output is pasted; merge order is by task id.

| # | Task | Files (scope) | Depends on | Hours | Acceptance |
|---|---|---|---|---|---|
| T1 | Clock + orientation validation | `pipeline/io.py`, `pipeline/align.py`, `tests/test_align.py` | — | 1.5 | `python -m pipeline.align --validate` prints ≥20 maps with tickrate resolved and team1/CT agreement 100% |
| T2 | Feature matrix | `pipeline/features.py`, `tests/test_features.py` | T1 | 1.0 | no-leakage assert passes; `X.shape`, `y.mean()` printed |
| T3 | Train + calibrate | `models/train.py` | T2 | 1.5 | `--cv` prints logit & xgb Brier ≤ 0.21, artifacts saved |
| T4 | Scoring + attribution | `models/score.py` | T3 | 0.75 | contribution sum ≈ logit − intercept within 1e-9 |
| T5 | Market series + quality gate | `pipeline/market.py` | T1 | 1.0 | quality dict printed; unusable maps excluded with reason |
| T6 | Detection (residual gap, D, C) | `pipeline/detect.py` | T4, T5 | 1.5 | paired ablation passes: injected map alerts and an alert span covers the injected rounds; control rate reported, not asserted (thresholds are provisional) |
| T7 | Injection | `pipeline/inject.py` | T5 | 0.5 | injected rows flagged; originals unchanged |
| T8 | Artifact build | `pipeline/build_artifacts.py` | T6, T7 | 1.0 | bundle validates, < 20 MB, ≥ 200 maps |
| T9 | API server | `server/app.py` | T8 (contract only) | 1.0 | `/api/health` shows s390x; `/api/score` < 50 ms |
| T10 | Front-end | `site/index.html`, `site/app.js`, `site/style.css` | §3 contract only | 2.0 | alert renders; static fallback works |
| T11 | Deploy | `deploy/*` | T9 | 0.75 | public URL answers `/api/health` from outside |
| T12 | Fallback bundle | `pipeline/build_artifacts.py --static-only` | T8 | 0.25 | `python -m http.server` over `site/` renders the alert |

**Parallelism map (who can start immediately):**

```
T1 ──► T2 ──► T3 ──► T4 ──┐
 └───► T5 ────────────────┴──► T6 ──┐
        └──► T7 ───────────────────┴──► T8 ──► T12
                              §3 contract ──► T9 ──► T11
                              §3 contract ──► T10
```

`T9` and `T10` start on day one against the frozen contract with a hand-written fixture in `site/data/` — they must not wait for T8. That's the entire point of §3.

**Triage, by measured cost** (this is the part that decides whether it fits):

| Path | Tasks | Wall clock | What you get |
|---|---|---|---|
| Full | all 12 | 9.0 h | live inference on Z + C/D + fallback |
| Lean | drop C + `size_anomaly` from T6, drop xgb from T3 | 8.5 h | live inference, gap + persistence only |
| Static-only | T1,T2,T3,T5,T7,T8,T10,T12 | 8.75 h | precomputed site, no public URL |

The static-only path is **not cheaper** than the lean path, because the public URL costs only T9 (1.0 h) + T11 (0.75 h) while the front-end costs 2.0 h and clock validation costs 1.5 h. So if hours run short, cut detection richness and front-end polish — never cut the live endpoint, since it is the one thing that demonstrates s390x inference.

**Critical path:** T1 → T2 → T3 → T4 → T6 → T8 → T9 → T11 = **9.0 h**. Sum of all tasks is **12.75 h**, so parallelism is not optional — wall clock equals the critical path only if independent tasks run concurrently. Cut order if behind: C part of T6, then `size_anomaly`, then xgb in T3, then T12. Never cut: T1 validation, calibration in T3, persistence in T6, the disclaimer in T10.

---

## 6. Definition of done

1. `curl 148.100.79.57:38888/api/health` from outside returns `"arch": "s390x"` and `n_maps: 297`.
2. The site lists ≥ 200 maps; clicking the injected map fires exactly one alert with contributions and the verbatim disclaimer.
3. `POST /api/score` returns a probability and a latency in milliseconds, computed at request time (proved by changing the input and seeing the output change).
4. Killing the API leaves the page rendering from the static bundle with a visible banner.
5. `python -m json.tool` passes on every artifact; no artifact over 20 MB.
6. `grep -ri "fixed\|cheat\|banned" site/` returns nothing.
7. `docs/implementation-plan.md` and `cs2-match-fixing-spec.md` agree on the numbers in §1.3 and §3.

---

## 7. Risks specific to this build

| Risk | Trigger | Response |
|---|---|---|
| Jupyter reclaims 38888 | container restart | run `deploy/start_all.sh`; it is idempotent and takes ~5s |
| `tickrate` detection ambiguous | odd maps | exclude the map from artifacts with `reason: "clock_unresolved"`; never guess |
| Only ~297 maps join both halves | fixed by data | already reflected: `n_maps: 297`, no coverage claim beyond it |
| Logistic model underperforms (Brier > 0.25) | weak features | fall back to fewer features + isotonic only; do not ship an uncalibrated model |
| Public names + injected data | decided, not a risk to manage technically | UI disclaimer, `noindex`, no allegation language |
| Flask missing in image | `pip install flask` allowed? | verified Jinja2 present; Flask is pure Python and installs on s390x or falls back to `http.server` + manual routing |
| Agent collisions | two agents editing one file | §5 scopes are disjoint; any task needing an out-of-scope edit must stop and request a contract change |

---

## 8. What stays open

| Item | Owner | Note |
|---|---|---|
| Video format/length | organiser rules | plan assumes 4 min, recorded from the site |
| Polymarket terms review | nobody yet | blocks publishing raw derived series beyond the demo page |
| Second market feed (OddsPapi coverage) | after demo | only from ~Apr 10 2026 |
| NOMERCY retrospective as acceptance test | after demo | data exists; not on the critical path |
