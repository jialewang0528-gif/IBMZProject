# Deploy runbook

Facts this runbook is built on (all measured 17 Oct 2026, not assumed):

| Fact | Value | How it was verified |
|---|---|---|
| Public IP | `148.100.79.57` | `curl ifconfig.me` on the host |
| Reachable ports | **22 and 38888 only** | probed from an outside machine: 38888 → 302, 8000/8080/5000/9000 → timeout |
| Architecture | `s390x`, Python 3.12.15 | `uname -m`, `platform.machine()` inside the container |
| **Container network** | **`NetworkMode=host`** | `docker inspect`: `PortBindings={}` |
| **Restart policy** | **`no`** | `docker inspect` |
| **PID 1 of the container** | **`tini -g -- jupyter notebook ... --port=38888`** | `/proc/1/cmdline` |
| Jupyter server | PID 6, child of tini | `/proc/*/cmdline` |
| Data | `/home/ibm-user/cs2_betting_data` in the container's **writable layer** — no bind mount, no volume | | 
| Disk | 33 GB free after the deploy snapshot | `df -h` |

Two consequences drive everything below:

1. The container shares the **host** network namespace, so `38888` is literally
   the host's port. There is no port mapping to change.
2. **Jupyter is the container's process.** tini forwards signals to its only
   child, so killing Jupyter stops the container — and `RestartPolicy=no` means
   nothing brings it back.

> **Do not run `pkill -f jupyter`.** An earlier version of `deploy/start_all.sh`
> did, which would have taken the box down and left it down.

---

## 1. Build the artifacts (inside the container)

Artifacts live in the writable layer, so they must be built in the container
that will serve them.

```bash
docker exec -it cs2site bash        # or `notebook` before the first swap
cd /home/ibm-user/IBMZProject
export CS2_DATA_ROOT=/home/ibm-user/cs2_betting_data

python3 -m models.train --cv
python3 -m pipeline.build_artifacts --out site/data \
    --inject-map 2393088-2 --inject-rounds 14,15,16
```

The demo map is **Legacy vs MOUZ, IEM Rio 2026, de_inferno**: 23 rounds, no
alerts before injection, and injecting rounds 14–16 yields exactly one high
alert at round 14 spanning `[14,16]`. Pick a different one only if it still has
zero organic alerts, or the injection is no longer the thing the viewer sees.

Expect roughly 20 minutes: ~1,014 maps, ~756 alerts, 269 skipped. The skips are
reported with reasons in `site/data/summary.json` (`skipped`), never silently
dropped.

## 2. Take port 38888 (from the HOST, not inside the container)

There is **no docker CLI inside the container** and the container cannot swap
its own PID 1, so this step runs on the host:

```bash
ssh linux1@148.100.79.57 'bash /home/ibm-user/IBMZProject/deploy/deploy_host.sh'
```

It does four things, and it is safe to re-run:

1. `docker commit notebook cs2site:snapshot` — carries the data, which only
   exists in the container's writable layer.
2. `docker stop notebook` — frees host port 38888.
3. Runs a replacement container `cs2site` whose *process* is the site.
4. Waits for `/api/health`; on failure it rolls itself back to Jupyter.

Verify from **outside** the box:

```bash
curl -s http://148.100.79.57:38888/api/health
```

Must show `"arch": "s390x"`, both models loaded, and an `n_maps` equal to the
`n_maps` in `site/data/summary.json`. It is not a fixed number; 1,014 at the
time of writing.

## 3. Rollback

```bash
ssh linux1@148.100.79.57 'docker stop cs2site && docker start notebook'
```

The old container is never removed, so this restores Jupyter on 38888.

## 4. After a rebuild, refresh the image too

The running container reads `site/data` at startup, so `docker restart cs2site`
is enough for a data-only change. But if the container is ever *recreated*, it
comes from the image — so fold the new data back in:

```bash
docker commit cs2site cs2site:snapshot
```

Then reach Jupyter again on 8888 (only while `notebook` is the running one):

```bash
ssh -L 8888:localhost:8888 linux1@148.100.79.57
```

## 5. Security notes

- The public app is unauthenticated by design (judges must not need a token).
  It serves only derived JSON and static assets. Static routes resolve within
  `site/static` and `site/data` and reject anything that escapes by path
  component; 404s do not echo the server's filesystem layout.
- **Delete the stored GitHub token before publishing**: `/home/ibm-user/.git-credentials`
  is a plaintext PAT:
  ```bash
  rm -f ~/.git-credentials && git config --global credential.helper ''
  ```
- `X-Robots-Tag: noindex, nofollow` is on every response, and the page carries
  `<meta name="robots">`.
- Real player names appear on the page (product decision). The UI never uses
  allegation language, and every alert card carries the "not a finding about any
  person" disclaimer.

## 6. Optional: standalone image

```bash
docker build -f deploy/Dockerfile -t cs2-integrity:latest .
docker run -d --name cs2-site --network host cs2-integrity:latest
```

Do **not** run this alongside the other container: with `NetworkMode=host` they
fight over 38888.
