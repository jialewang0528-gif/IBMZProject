# Handoff — CS2 Integrity Watch

Written 17 Oct 2026, at the end of a long session, for an agent or human picking
this up cold. Read this first, then `cs2-match-fixing-spec.md` (the product
contract) and `docs/implementation-plan.md` (the build contract).

---

## 1. What this is

A game-side integrity layer for Counter-Strike 2. It puts market-implied
probability next to a game-state win-probability model on one clock, and ranks
maps where the market moved away from what the game state said — for a human to
investigate. **An alert is never a verdict.**

**It is live: <http://148.100.79.57:38888/>**

| Deliverable | State |
|---|---|
| Hardened spec | `cs2-match-fixing-spec.md`, current, with the calibration result in §4.3a |
| Pipeline + model | Working on real data, 33 tests pass |
| Public site | Live and verified from outside; 1,014 maps, 756 alerts |
| Video | **NOT RECORDED** — the only missing deliverable |

---

## 2. Environment facts (measured — do not re-derive)

| Fact | Value |
|---|---|
| Host | LinuxONE instance `tylerinstance1`, public IP `148.100.79.57` |
| Container | now **`cs2site`** (see §8). The old `notebook` container is stopped, not deleted |
| Architecture | **s390x**, Python 3.12.15, **3 vCPUs**, 8 GB RAM, 33 GB free disk |
| Installed | pandas 3.0.6, numpy 2.0.2, scikit-learn 1.9.0, xgboost 3.4.1, scipy, jinja2 |
| **Not** installed | flask, pyarrow, streamlit, shap, pytest, kubectl, `ps`, `pgrep`, `timeout`, `free`, `uptime` |
| Container network | **`NetworkMode=host`** — port 38888 *is* the host's port |
| Container PID 1 | `tini -g -- jupyter notebook … --port=38888` (on the old container) |
| Restart policy | **`no`** — if the container stops, nothing brings it back |
| Reachable ports | only **22 and 38888** |
| Tunnels | impossible — no s390x `cloudflared`/`ngrok` |
| Data | `/home/ibm-user/cs2_betting_data`, **in the container's writable layer** (no volume, no bind mount) |
| Repo in container | `/home/ibm-user/IBMZProject` |

Consequences baked into the code: stdlib `http.server` (no Flask), no parquet
(use csv/gz), the app owns 38888, and the deploy is a container swap from the
host (§8).

Missing shell utilities matter: check liveness by scanning `/proc/*/cmdline`,
not `ps`/`pgrep`.

---

## 3. Data facts (measured)

**Game side.** HF `blanchon/opencs2_dataset_demo` (CC-BY-4.0), parsed with
`demoparser2 0.42.0`. `round_table` 42,009 rows · `round_detail` 42,009 ·
`kills` 286,671 · `player_round_economy` 420,090 · `maps_meta` **1,988 maps
across 860 series** · `pistol_loss_opps` 3,960.

**Market side.** Polymarket public APIs. 12,489 events / 117,318 markets; scope
860 Bo3 series, Mar 23 – Apr 20 2026; 1-minute snapshots including in-play;
every fill. `odds_table.csv.gz` ~963k Polymarket rows + 13k OddsPapi.
**Polymarket prices are already probabilities — no margin.**

**The pool, end to end** — quote whichever line the claim needs, and no more:

```
1,988  maps in maps_meta                       (860 series)
1,284  maps have a map-level market            (665 series = 77% of 860)
1,014  maps survive the whole pipeline         (269 skipped, reasons in summary.json)
  756  alerts, all of which satisfy the stated rule
```

> **"297" and "1,723" were never measurements.** Both came from
> `tools/schema_report.py`: 1,723 was its column selector *unioning* `match_id`
> (860) + `match_date` (860) + score tokens (3) and printing the set's size as
> "key values"; 297 came from a hardcoded 400k-row cap on the odds scan. The
> missed tell: 1,988 maps / 1,723 series = 1.15 maps per series, impossible for
> a Bo3. The tool is fixed and a test fails if the numbers reappear outside the
> post-mortem in the spec.

---

## 4. Measured results

| Gate | Result |
|---|---|
| Clock | **64-tick** everywhere (verified two independent ways, §9.2) |
| Orientation | 94% of maps resolve to exactly one side |
| Clock alignment | 80% of maps; median market coverage **1.00** |
| Model | Brier **0.1775** logit / **0.1768** xgb on a 188-match holdout (target ≤ 0.21; coin flip 0.25) |
| Live inference | `/api/score` ≈ **9 ms** on s390x |
| Tests | 33 pass in ~11 s |
| Build | 1,014 maps in **6.5 min** with `--jobs 3` |

---

## 5. The central limitation — read this before claiming anything

The alert thresholds have been calibrated, and **they do not work.** Null = a
circular shift of each map's market series relative to its rounds (real price
path, alignment destroyed). Anything the null produces is a false positive by
construction.

| threshold | alerts/map | maps alerting | null/map | null share |
|---|---|---|---|---|
| 0.10 | 0.96 | 65% | 0.92 | 95% |
| **0.15 (shipped)** | **0.77** | **56%** | **0.79** | **102%** |
| 0.20 | 0.54 | 39% | 0.66 | 122% |
| 0.30 | 0.21 | 17% | 0.44 | 209% |

Real data produces **no more alerts than a misaligned copy of itself**, and it
gets worse as the threshold rises. The cause is structural, not numeric: the
statistic is a *level* gap against a per-map baseline, and both feeds drift
slowly, so a shifted market has a similar profile. `market_moved_first` does not
rescue it — a 0.05 step inside 10 minutes is satisfied by one-minute noise on
nearly every map.

**Therefore: quote no detection rate, no false-positive rate, and never imply an
alert is a finding.** The demo's alert is *injected and labelled*; that is the
only alert attributable with certainty.

**The fix** is a statistic sensitive to *alignment*, not level: does the
market move with the model's round-to-round changes (the model moves when rounds
are won), and where does it stop doing so? Require departure from that
co-movement rather than a level offset. `tools/calibrate.py` already implements
the right null for testing it. This is the highest-value remaining work.

---

## 6. Code map

```
pipeline/io.py              table/odds resolution, caching (odds parse and the
                            map_winner filter are cached — asking per map is free)
pipeline/align.py           clock: 64-tick inference, per-map timezone correction,
                            map→round timeline, as-of market join, --validate
pipeline/features.py        start-of-round features; no leakage; map_name excluded
pipeline/market.py          series extraction + quality gate over the map's own
                            trading window (with reasons)
pipeline/detect.py          residual-gap alerting (all four criteria enforced),
                            sign-aware persistence, player deviation, --ablation
pipeline/inject.py          injection of panel values AND market prints, both flagged
pipeline/build_artifacts.py JSON contract (matches/series/summary), fork-pool build
models/train.py             split by match_id (30% of matches), logit + xgb + isotonic
models/score.py             scoring + exact logit attribution (no SHAP on s390x)
server/app.py               stdlib ThreadingHTTPServer, path-component containment
site/                       vanilla JS SPA, inline SVG chart, offline-capable
deploy/deploy_host.sh       HOST-side port swap (§8)
deploy/start_all.sh         what the container runs; does NOT touch Jupyter
tools/{calibrate,audit_data,audit_cell,schema_report,make_fixture}.py
tests/test_pipeline.py      33 tests: pure rules, CLI acceptance, live server, packaging
```

Run everything **inside the container**:

```bash
docker exec -it cs2site bash
cd /home/ibm-user/IBMZProject
export CS2_DATA_ROOT=/home/ibm-user/cs2_betting_data

python3 -m unittest discover -s tests -v        # expect 33 OK
python3 -m pipeline.align --validate --n-maps 60
python3 -m models.train --cv                    # writes models/artifacts/
python3 -m pipeline.build_artifacts --out site/data \
    --inject-map 2393088-2 --inject-rounds 14,15,16      # ~6.5 min
python3 tools/calibrate.py --maps 150           # ~5 min; see §5
docker restart cs2site                          # reload artifacts
```

Demo map: **`2393088-2`, Legacy vs MOUZ, IEM Rio 2026, de_inferno.** 23 rounds,
zero organic alerts, injected 14–16 → exactly one high alert at round 14
spanning `[14,16]`, residual −0.4046, five attribution contributions, 10 players
in the panel. If you change the detector, re-check that the injection still
produces its own alert — the first pick (Sangal vs Nemiga) silently stopped
being a clean demo when the rules tightened.

---

## 7. What to do in the first 30 minutes

1. `curl http://148.100.79.57:38888/api/health` — expect `s390x`, both models,
   `n_maps: 1014`.
2. `python3 -m unittest discover -s tests` — expect 33 OK.
3. Read spec §4.3a and §5 above. **Do not skip this.**
4. `python3 tools/calibrate.py --maps 60` to see the null result for yourself
   before touching the detector.
5. Then: the detector redesign, or the video.

---

## 8. Deploy topology (read `deploy/deploy.md` before touching ports)

The container shares the **host** network namespace, so 38888 is the host's
port — there is no mapping to change. Jupyter was the container's PID 1 child,
so killing it stops the container, and `RestartPolicy=no` means nothing restarts
it. **Do not run `pkill -f jupyter`.**

The swap runs from the host:

```bash
ssh linux1@148.100.79.57 'bash /home/ibm-user/IBMZProject/deploy/deploy_host.sh'
```

It commits the container (carrying the data out of the writable layer), stops
it, starts `cs2site` whose *process* is the site, waits for `/api/health`, and
rolls itself back on failure.

**Rollback:** `docker stop cs2site && docker start notebook`

After a data rebuild, `docker restart cs2site` is enough; to keep the image in
sync with what is running, `docker commit cs2site cs2site:snapshot`.

Consequence to know: **Jupyter is offline.** `docker exec notebook` no longer
works. Use `docker exec cs2site`. Jupyter is still reachable inside the stopped
container if you start it (on 8888 only if you edit its config; its entrypoint
hardcodes 38888).

---

## 9. Traps already hit (each cost real time; do not repeat)

**9.1 Data and joining**

1. **A blanket `models/` line in `.gitignore` excluded the models package.** A
   fresh clone failed with `No module named models`. Only `models/artifacts/` is
   ignored now, and `TestRepoPackaging` fails if a promised file is untracked.
2. **`maps_meta.csv.gz` lives in `tables/`**, not the data root. `io.py` tries
   both and says what it looked for.
3. **`match_date` is the series start,** identical for every map of a match.
   Using it as the map start misaligns map 2+ by an entire map.
4. **`match_date` is also event-local time mislabelled as UTC.** Measured
   against each map's market settlement, the game clock sits 1–3 whole hours
   ahead, clustering at −1 h (CET), −2 h (CEST), −3 h (MSK). Uncorrected the
   feeds never overlap and coverage is 0.00 on every map. The clock is anchored
   per map on its own market settlement (last print strictly between 0.02 and
   0.98) — a known shared event, not a price level. This also makes the
   inter-map break assumption cancel out.
5. **The clock columns are 64-tick, not 128.** Mean round length cannot tell you
   (≈7,000 units is 109 s at 64-tick and 55 s at 128-tick — that ambiguity is
   what made an earlier version resolve maps to 128 and put every timestamp out
   by 2×). Pin it on rules the server enforces: the modal freezetime (1,280 =
   20 s; at 128-tick it would be 10 s, not a standard value) and the 115 s round
   clock + 40 s bomb ceiling. At 128-tick **no round in 42,009 ever reaches the
   round limit**, which is impossible.
6. **`eligible_maps()` claimed "has market rows" but tested only that the
   match_id appeared somewhere in the odds table** — admitting 254 maps with no
   map_winner print. It now requires the map's own market.
7. **Steam IDs exceed 2^53.** `player_round_economy.steamid` is int64 (exact),
   `kills.attacker_steamid` is float64 (rounded). Joining on them matched
   nothing, kills stayed 0, variance was 0, and the player panel shipped **empty
   on all 1,014 maps without an error**. Joined on player name now.
8. **pandas 3 raises `MergeError` on `datetime64[ns]` vs `[us]`.** Pin both
   sides of every `merge_asof` to ns. A bare `except` in `validate()` converted
   this into "coverage 0.00" and hid it for a whole session.

**9.2 Measurement logic**

9. **Do not measure max round length over `end - start`.** That includes the
   freezetime and any pause inside it. Use `end - freeze_end` (live play) — that
   is the quantity with a hard ceiling.
10. **`baseline_gap` must be positional.** Taking the first three *non-null*
    gaps pulls the baseline from whichever rounds have prints, which can be the
    divergent rounds themselves — silently cancelling the gap being measured.
11. **A persistence span must break when the sign flips.** Otherwise the run
    borrows its first round's direction, and "the market moved in the divergence
    direction" looks for a downward step inside an upward move.
12. **`alerts()` must enforce its own rule.** It once computed
    `market_moved_first` / `no_explaining_event` and emitted anyway: 47% of
    shipped alerts failed a mandatory criterion.
13. **Calibration needs the right null.** A permutation of prices destroys their
    serial correlation and manufactures runs (it reported 110% when the truth
    was 102%); a circular shift is the correct null. And check the arithmetic of
    the ratio — the first version divided by the shuffle count twice.
14. **A 16-map fixture is too small to calibrate on.** Isotonic saturates to
    exactly 1.0, so the market cannot diverge and the injection test becomes
    vacuous. Tests use 40 maps.

**9.3 Infrastructure**

15. **`net=host` + `RestartPolicy=no` + Jupyter as PID 1** ⇒ killing Jupyter
    stops the container permanently. See §8.
16. **`docker exec` output has no `ps`/`pgrep`/`free`/`uptime`/`timeout`.**
    Scan `/proc/*/cmdline`. When killing by pattern, exclude your own shell —
    a `case *build_artifacts*` string kills the shell running it.
17. **Nested shell quoting** through `ssh → docker exec → bash -lc` will eat a
    command, and `$PWD` expands *before* `cd`. Write script files and pipe them
    in (`ssh … 'docker exec -i cs2site bash -lc "cat > /tmp/x.py"' < x.py`).
18. **`io.read_odds()` re-parsed 1.04 M rows per call.** At one call per map,
    `validate --n-maps 250` took 800+ s. Cached now; do not undo that.

---

## 10. State and next steps, in order

1. **Detector redesign** (§5). Everything else is cosmetic next to this.
2. **Record the video** — the only missing deliverable.
3. Re-run `tools/calibrate.py` after the redesign and put the number in the spec.
4. Optional: NOMERCY retrospective (Mar 28 2026, 4 series + 33 controls) — never
   started, not on any critical path.

## 11. Open items

| Item | Owner | Note |
|---|---|---|
| `/home/ibm-user/.git-credentials` holds a plaintext GitHub PAT | **human** | Delete before anyone looks hard at this box; the app is unauthenticated by design |
| Data lives in the container's writable layer | **human** | It is now also inside `cs2site:snapshot`, but copy to `/home/linux1/shared` for a real backup |
| Polymarket terms of use | nobody | Not reviewed; blocks publishing raw derived series |
| Video format / length | organiser rules | Unknown; plan assumes 4 min |
| Jupyter is offline | human | Deliberate: it held 38888. Start the old container to get it back |
| Second market source (OddsPapi) | later | History only from ~Apr 10 2026 |

## 12. Decisions already made (do not re-litigate)

- **Real player names are permitted on the public site, with no access gate.**
  Mitigated in the UI: no allegation vocabulary ("divergence score", never
  "fixed"), verbatim disclaimer on every alert card, `noindex`, and a banner
  saying part of the feed is simulated and no finding is made about any person.
  `test_no_allegation_language_in_site` enforces the vocabulary.
- **The deliverable is a spec, a video, and a site that survives without a
  presenter.**
- **An alert is a review candidate, never a verdict.**

## 13. Honest limitations to state in any pitch

- 1,284 maps have both data halves; **1,014** survive the pipeline.
- 60-second market resolution plus map-start error: ordering below that is not
  measurable.
- No round-level markets exist → map-level analysis only.
- **The alert rule has not been shown to beat a misaligned null** (§5). Say this
  plainly; it is more credible than hiding it.
- The demo's market feed is injected and labelled.
- One retrospective case (NOMERCY) is not a validation study.
