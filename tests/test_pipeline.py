"""Acceptance tests. Run with:

    python -m unittest discover -s tests -v

Design: pure-function tests for the rules, and subprocess tests for the
acceptance commands in docs/implementation-plan.md §4, because those commands
are what the build is judged on. No pytest dependency (the s390x image has
none), no network.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str], env: dict, expect: int = 0) -> subprocess.CompletedProcess:
    proc = subprocess.run([PY, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != expect:
        raise AssertionError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
        )
    return proc


# ---------------------------------------------------------------- pure logic
class TestRepoPackaging(unittest.TestCase):
    """A blanket .gitignore rule once excluded the whole models/ package.

    That failure mode is invisible locally and fatal on a fresh clone, so it is
    checked here: every package the docs promise must be tracked by git.
    """

    REQUIRED = [
        "pipeline/io.py", "pipeline/align.py", "pipeline/features.py",
        "pipeline/market.py", "pipeline/detect.py", "pipeline/inject.py",
        "pipeline/build_artifacts.py", "pipeline/__init__.py",
        "models/__init__.py", "models/train.py", "models/score.py",
        "server/app.py", "site/index.html", "site/static/app.js",
        "site/static/style.css", "deploy/start_all.sh", "deploy/deploy.md",
        "tests/test_pipeline.py", "tools/make_fixture.py",
        "cs2-match-fixing-spec.md", "docs/implementation-plan.md",
    ]

    def test_all_required_files_are_tracked(self):
        if not (ROOT / ".git").exists():
            self.skipTest("not a git checkout")
        tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.split())
        missing = [f for f in self.REQUIRED if f not in tracked]
        self.assertFalse(missing, f"not tracked by git (ignored?): {missing}")

    def test_packages_are_importable(self):
        for pkg in ("pipeline", "models"):
            proc = subprocess.run([PY, "-c", f"import {pkg}; print('ok')"], cwd=ROOT,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"cannot import {pkg}: {proc.stderr[-400:]}")


class TestClockUnits(unittest.TestCase):
    """The clock columns may be ticks at 64/128 or already seconds."""

    def _write(self, tmp: Path, unit: str) -> None:
        import csv, gzip
        rows = []
        start = 0
        for r in range(1, 11):
            if unit == "seconds":
                dur = 75
                rows.append([r, start, start + 20, start + 95, start + 98, 75])
                start += 95 + 7
            elif unit == "ticks64":
                rows.append([r, start, start + 20 * 64, start + 95 * 64,
                             start + 98 * 64, 75 * 64])
                start += (95 + 7) * 64
            else:  # ticks128
                rows.append([r, start, start + 20 * 128, start + 95 * 128,
                             start + 98 * 128, 75 * 128])
                start += (95 + 7) * 128
        (tmp / "tables").mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp / "tables" / "round_detail.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["map_uid", "round_num", "start", "freeze_end", "end",
                        "official_end", "winner", "reason", "bomb_plant", "bomb_site",
                        "duration_ticks"])
            for row in rows:
                w.writerow(["m-1", *row, "ct", "ct_killed", "FALSE", "", row[-1]])

    def test_unit_inference(self):
        from pipeline import align
        expected = {"seconds": 1, "ticks64": 64, "ticks128": 128}
        for unit, want in expected.items():
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                self._write(root, unit)
                det = pd.read_csv(root / "tables" / "round_detail.csv.gz")
                mean = det["duration_ticks"].mean()
                if unit == "seconds":
                    self.assertEqual(want, 1)
                else:
                    self.assertAlmostEqual(mean / want, 75)  # what the detector relies on
        # the ambiguous case must raise, not guess
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root, "seconds")
            det = pd.read_csv(root / "tables" / "round_detail.csv.gz")
            det["duration_ticks"] = det["duration_ticks"] * 1000  # nonsense units
            det.to_csv(root / "tables" / "round_detail.csv.gz", index=False)
            vals = det["duration_ticks"]
            self.assertFalse(45 <= vals.mean() <= 115)
            self.assertFalse(any(45 <= vals.mean() / tr <= 115 for tr in (64, 128)))


class TestDetectionRules(unittest.TestCase):
    def _panel(self, market, model):
        n = len(market)
        return pd.DataFrame({
            "round_num": np.arange(1, n + 1),
            "market_p": market,
            "model_p": model,
            "t_start_utc": pd.date_range("2026-03-24T02:00:00Z", periods=n, freq="2min"),
            "t_end_utc": pd.date_range("2026-03-24T02:00:30Z", periods=n, freq="2min"),
        })

    def test_persistence_needs_consecutive_rounds(self):
        from pipeline import detect
        gap = pd.Series([0.01, 0.20, 0.01, 0.20, 0.20, 0.20, 0.01])
        spans = detect.persistence(gap, thresh=0.15, k=2)
        self.assertEqual(spans, [(3, 5)], "a single-round spike must not alert")

    def test_residual_removes_per_map_offset(self):
        from pipeline import detect
        # market is a constant 20 points above the model: that is a bias, not a divergence
        model = [0.4] * 10
        market = [0.6] * 10
        panel = self._panel(market, model)
        self.assertAlmostEqual(detect.baseline_gap(panel), 0.2, places=6)
        self.assertTrue((detect.residual_gap(panel).abs() < 1e-9).all(),
                        "a constant offset must produce zero residual")

    def test_severity_thresholds(self):
        from pipeline import detect
        self.assertEqual(detect.severity(0.30, 3, False), "high")
        self.assertEqual(detect.severity(0.22, 2, False), "medium")
        self.assertEqual(detect.severity(0.15, 2, True), "medium")
        self.assertEqual(detect.severity(0.16, 2, False), "low")
        self.assertEqual(detect.severity(0.10, 1, False), "low")

    def test_injection_moves_market_away_from_model(self):
        from pipeline import inject
        panel = self._panel([0.40] * 8, [0.60] * 8)
        out = inject.inject(panel, [3, 4, 5])
        self.assertEqual(inject.injected_rounds(out), [3, 4, 5])
        # market below model -> injection pushes it further below
        self.assertLess(out.loc[out["round_num"] == 4, "market_p"].iloc[0], 0.40)
        untouched = out[~out["injected"]]
        self.assertTrue(untouched["market_p"].equals(panel.loc[untouched.index, "market_p"]))

    def test_series_quality_rejects_sparse(self):
        from pipeline import market
        ts = pd.to_datetime(["2026-03-24T02:00:00Z", "2026-03-24T02:01:00Z"], utc=True)
        sparse = pd.DataFrame({"ts_utc": ts, "market_p": [0.5, 0.51], "is_live": [True, True]})
        q = market.series_quality(sparse)
        self.assertFalse(q["usable"])
        self.assertIn("too_few_prints", q["reason"])


# ------------------------------------------------------------ acceptance CLIs
class TestPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.data = base / "data"
        cls.artifacts = base / "artifacts"
        cls.site = base / "site"
        cls.env = {
            **os.environ,
            "CS2_DATA_ROOT": str(cls.data),
            "CS2_ARTIFACT_DIR": str(cls.artifacts),
            "PYTHONPATH": str(ROOT),
        }
        run(["tools/make_fixture.py", "--out", str(cls.data), "--maps", "16"], cls.env)
        # the real dataset keeps maps_meta inside tables/; mirror that here so the
        # path-resolution code is exercised rather than assumed
        (cls.data / "tables").mkdir(exist_ok=True)
        (cls.data / "maps_meta.csv.gz").rename(cls.data / "tables" / "maps_meta.csv.gz")
        meta = pd.read_csv(cls.data / "tables" / "maps_meta.csv.gz")
        cls.map_uid = str(meta["map_uid"].iloc[1])

        run(["-m", "models.train"], cls.env)
        run(["-m", "pipeline.build_artifacts", "--out", str(cls.site),
             "--inject-map", cls.map_uid], cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_align_validates_clock(self):
        out = run(["-m", "pipeline.align", "--validate"], self.env).stdout
        self.assertIn("tickrate resolved", out)
        self.assertIn("orientation ok", out)
        self.assertNotIn("None", out.split("tickrate resolved")[1].split("\n")[0])

    def test_market_coverage_across_maps(self):
        """Map 2 shares match_date with map 1; if the anchor were the series
        start, map 2 would have ~zero market coverage."""
        out = run(["-m", "pipeline.align", "--validate"], self.env).stdout
        self.assertIn("market coverage median", out)
        median = float(out.split("market coverage median:")[1].split()[0])
        self.assertGreater(median, 0.5, "market join failed for most rounds: "
                                        "check the map-start anchor")

    def test_no_leakage_in_features(self):
        out = run(["-m", "pipeline.features", "--map-uid", self.map_uid], self.env).stdout
        self.assertIn("no nulls, no leakage columns", out)

    def test_attribution_reconstructs_logit(self):
        out = run(["-m", "models.score", "--map-uid", self.map_uid], self.env).stdout
        self.assertIn("attribution check", out)

    def test_injection_ablation(self):
        out = run(["-m", "pipeline.detect", "--map-uid", self.map_uid, "--ablation"],
                  self.env).stdout
        self.assertIn("PASS: injection is detected end-to-end", out)

    def test_artifacts_match_contract(self):
        matches = json.loads((self.site / "matches.json").read_text())
        summary = json.loads((self.site / "summary.json").read_text())
        self.assertGreater(len(matches), 0)
        self.assertEqual(summary["n_maps"], len(matches))
        for m in matches:
            for key in ("map_uid", "team1", "team2", "map_name", "n_alerts", "severity",
                        "resolution_floor_s"):
                self.assertIn(key, m)
        uid = self.map_uid
        doc = json.loads((self.site / "series" / f"{uid}.json").read_text())
        for key in ("meta", "rounds", "alerts", "players"):
            self.assertIn(key, doc)
        self.assertTrue(doc["meta"]["resolution_floor_s"] >= 60)
        for r in doc["rounds"]:
            self.assertGreaterEqual(r["model_p"], 0.0)
            self.assertLessEqual(r["model_p"], 1.0)
        injected = [r for r in doc["rounds"] if r["injected"]]
        self.assertTrue(injected, "the injected map must carry flagged rounds")
        for a in doc["alerts"]:
            self.assertIn("not a finding", a["disclaimer"].lower())

    def test_no_allegation_language_in_site(self):
        site = ROOT / "site"
        text = "".join(p.read_text(errors="ignore") for p in site.rglob("*")
                       if p.suffix in {".html", ".js", ".css"})
        lowered = text.lower()
        for banned in ("match-fixing", "match fixing", "they cheated", "is fixed", "banned"):
            self.assertNotIn(banned, lowered, f"allegation language present: {banned}")


# ------------------------------------------------------------------- server
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        data, artifacts, site = base / "data", base / "artifacts", base / "site"
        cls.env = {
            **os.environ,
            "CS2_DATA_ROOT": str(data),
            "CS2_ARTIFACT_DIR": str(artifacts),
            "CS2_SITE_DIR": str(site),
            "PYTHONPATH": str(ROOT),
        }
        run(["tools/make_fixture.py", "--out", str(data), "--maps", "8"], cls.env)
        (data / "tables").mkdir(exist_ok=True)
        (data / "maps_meta.csv.gz").rename(data / "tables" / "maps_meta.csv.gz")
        meta = pd.read_csv(data / "tables" / "maps_meta.csv.gz")
        cls.map_uid = str(meta["map_uid"].iloc[0])
        run(["-m", "models.train"], cls.env)
        run(["-m", "pipeline.build_artifacts", "--out", str(site / "data")], cls.env)

        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [PY, "-m", "server.app", "--host", "127.0.0.1", "--port", str(cls.port)],
            cwd=ROOT, env=cls.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/api/health", timeout=1)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.3)
        cls.proc.terminate()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.tmp.cleanup()

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return json.loads(r.read())

    def test_health(self):
        h = self.get("/api/health")
        self.assertTrue(h["ok"])
        self.assertIn("arch", h)
        self.assertGreater(h["n_maps"], 0)
        self.assertEqual(h["resolution_floor_s"], 60)

    def test_matches_and_series(self):
        matches = self.get("/api/matches")
        self.assertIsInstance(matches, list)
        doc = self.get(f"/api/series/{self.map_uid}")
        self.assertIn("rounds", doc)

    def test_score_is_live_inference(self):
        def post(state: dict) -> dict:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/api/score",
                data=json.dumps({"round_state": state}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())

        a = post({"score_team1": 1, "score_team2": 10, "round_num": 15,
                  "equip_team1": 800, "equip_team2": 4800, "buy_team1": "eco",
                  "buy_team2": "full", "map_name": "de_mirage"})
        b = post({"score_team1": 10, "score_team2": 1, "round_num": 15,
                  "equip_team1": 4800, "equip_team2": 800, "buy_team1": "full",
                  "buy_team2": "eco", "map_name": "de_mirage"})
        self.assertNotAlmostEqual(a["p_logit"], b["p_logit"], places=3,
                                  msg="output did not change with input: not live inference")
        self.assertLess(a["latency_ms"], 500)
        self.assertIn("contributions", a)

    def test_static_fallback_route(self):
        doc = self.get("/data/matches.json")
        self.assertIsInstance(doc, list)

    def test_bad_series_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/series/nope-1")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
