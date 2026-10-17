#!/usr/bin/env bash
# Start the site on the one port the platform exposes (38888) and move Jupyter
# to localhost:8888 so it no longer owns the public port.
#
# Idempotent: safe to re-run after a container restart, which is exactly when
# Jupyter steals 38888 back.
set -uo pipefail

APP_DIR="${APP_DIR:-/home/ibm-user/IBMZProject}"
PUBLIC_PORT="${PUBLIC_PORT:-38888}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
LOG_DIR="${LOG_DIR:-/tmp/cs2-site}"
mkdir -p "$LOG_DIR"

cd "$APP_DIR" || { echo "no app dir: $APP_DIR"; exit 1; }

echo "[1/4] freeing port $PUBLIC_PORT"
PIDS=$( (command -v ss >/dev/null && ss -tlnp 2>/dev/null || true) \
        | awk -v p=":$PUBLIC_PORT" '$4 ~ p {print $6}' \
        | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u )
for pid in $PIDS; do
  echo "      killing pid $pid on $PUBLIC_PORT"
  kill "$pid" 2>/dev/null || true
done
# also catch anything that still refuses to die by name
pkill -f "jupyter.*--port $PUBLIC_PORT" 2>/dev/null || true
sleep 1

echo "[2/4] starting Jupyter on 127.0.0.1:$JUPYTER_PORT (optional)"
if ! (command -v ss >/dev/null && ss -tln 2>/dev/null | grep -q ":$JUPYTER_PORT"); then
  nohup jupyter lab --no-browser --ip 127.0.0.1 --port "$JUPYTER_PORT" \
    > "$LOG_DIR/jupyter.log" 2>&1 &
  echo "      jupyter log: $LOG_DIR/jupyter.log"
else
  echo "      already listening on $JUPYTER_PORT"
fi

echo "[3/4] starting site on 0.0.0.0:$PUBLIC_PORT"
nohup python3 -m server.app --host 0.0.0.0 --port "$PUBLIC_PORT" \
  > "$LOG_DIR/site.log" 2>&1 &
echo "      site log: $LOG_DIR/site.log"

echo "[4/4] waiting for /api/health"
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PUBLIC_PORT/api/health" >/dev/null 2>&1; then
    echo "      OK after ${i}s"
    curl -s "http://127.0.0.1:$PUBLIC_PORT/api/health"
    echo
    exit 0
  fi
  sleep 1
done

echo "      FAILED — last 20 lines of $LOG_DIR/site.log:"
tail -20 "$LOG_DIR/site.log" 2>/dev/null || true
exit 1
