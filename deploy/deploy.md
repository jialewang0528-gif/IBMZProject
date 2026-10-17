# Deploy runbook

Three facts this runbook is built on (measured 17 Oct 2026):

| Fact | Value | How it was verified |
|---|---|---|
| Public IP | `148.100.79.57` | `curl ifconfig.me` on the host |
| Reachable ports | **22 and 38888 only** | probed from an outside machine: 38888 → 302, 8000/8080/5000/9000 → timeout |
| Architecture | `s390x`, Python 3.12.15 | `uname -m`, `platform.machine()` inside the container |

Consequences: **the app owns 38888**, Jupyter moves to `127.0.0.1:8888`, and
outbound tunnels are impossible (no `cloudflared`/`ngrok` build for s390x).

---

## 1. Build artifacts and models (inside the container)

```bash
docker exec -it notebook bash
cd /home/ibm-user/IBMZProject
git pull

export CS2_DATA_ROOT=/home/ibm-user/cs2_betting_data

python3 -m models.train --cv                        # writes models/artifacts/
python3 -m pipeline.build_artifacts \
    --out site/data --inject-map <map_uid>          # writes site/data/
```

Pick `<map_uid>` from the output of `python3 -m pipeline.align --validate`
or `python3 -m tools.schema_report` — any map that has both halves.

Sanity check before deploying:

```bash
python3 -m json.tool site/data/summary.json | head -20
```

## 2. Start the site

```bash
docker exec -d notebook /home/ibm-user/IBMZProject/deploy/start_all.sh
sleep 5
curl -s http://127.0.0.1:38888/api/health
```

From an outside machine: <http://148.100.79.57:38888/>

## 3. Recover after a container restart

A restart brings Jupyter back on 38888, so the site goes dark. Re-running the
same script fixes it (it is idempotent):

```bash
docker exec -d notebook /home/ibm-user/IBMZProject/deploy/start_all.sh
```

## 4. Reach Jupyter again (now on 8888)

```bash
ssh -L 8888:localhost:8888 linux1@148.100.79.57
# then open http://localhost:8888
```

## 5. Security notes

- The public app is unauthenticated by design (judges must not need a token).
  It serves only derived JSON and static assets; it never exposes the data dir
  or a shell.
- **Delete the stored GitHub token before publishing**: `/home/ibm-user/.git-credentials`
  is a plaintext PAT. Use an SSH key or a short-lived token instead:
  ```bash
  rm -f ~/.git-credentials
  git config --global credential.helper ''
  ```
- `X-Robots-Tag: noindex, nofollow` and `<meta name="robots">` are set on every
  response and in the page, so the site cannot be indexed.
- Real player names appear on the page (product decision). The UI never uses
  allegation language, and every alert card carries the "not a finding about
  any person" disclaimer.

## 6. Optional: image build for reproducibility

```bash
docker build -f deploy/Dockerfile -t cs2-integrity:latest .
docker run -d --name cs2-site -p 38888:38888 \
  -v /home/ibm-user/cs2_betting_data:/data:ro cs2-integrity:latest
```

Do **not** run this alongside the dev container on the same port — they fight
over 38888.
