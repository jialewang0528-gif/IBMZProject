#!/usr/bin/env bash
# Take port 38888 for the site. RUN THIS ON THE HOST, NOT INSIDE THE CONTAINER.
#
# There is no docker CLI inside the container, and the container cannot swap its
# own PID 1 (tini owns it and runs Jupyter). So the only correct move is:
#
#   1. snapshot the container, which carries its writable layer
#      (the 485 MB of data and site/data live there - there is no bind mount)
#   2. stop it, freeing 38888 on the host (the container uses NetworkMode=host)
#   3. start a replacement container whose process is the site
#
# The old container is left stopped, never removed, so rollback is one command.
#
# Usage:   ssh linux1@148.100.79.57 'bash /home/ibm-user/IBMZProject/deploy/deploy_host.sh'
# Rollback: ssh linux1@148.100.79.57 'docker stop cs2site && docker start notebook'
set -euo pipefail

CONTAINER="${CONTAINER:-notebook}"          # the Jupyter container
SITE_CONTAINER="${SITE_CONTAINER:-cs2site}" # the replacement
IMAGE="${IMAGE:-cs2site:snapshot}"
PORT="${PORT:-38888}"

echo "[1/4] snapshotting $CONTAINER -> $IMAGE"
echo "      (this carries the data: it lives in the writable layer, not a volume)"
docker commit "$CONTAINER" "$IMAGE"

echo "[2/4] stopping $CONTAINER (frees host port $PORT)"
docker stop "$CONTAINER" || true

echo "[3/4] starting $SITE_CONTAINER"
docker rm -f "$SITE_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$SITE_CONTAINER" --network host "$IMAGE" \
  bash -lc "cd /home/ibm-user/IBMZProject && exec bash deploy/start_all.sh $PORT" \
  >/dev/null

echo "[4/4] waiting for /api/health"
for _ in $(seq 1 30); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/api/health"; then
    echo
    echo "OK. From outside:  curl http://148.100.79.57:$PORT/api/health"
    echo "Rollback:          docker stop $SITE_CONTAINER && docker start $CONTAINER"
    exit 0
  fi
  sleep 1
done

echo "FAILED - last 20 lines of the container log:"
docker logs --tail 20 "$SITE_CONTAINER" 2>&1 || true
echo "Rolling back to Jupyter on $PORT ..."
docker stop "$SITE_CONTAINER" >/dev/null 2>&1 || true
docker start "$CONTAINER" >/dev/null 2>&1 || true
exit 1
