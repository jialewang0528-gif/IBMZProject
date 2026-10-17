#!/usr/bin/env bash
# Run the site as the container's PROCESS.
#
# It does NOT stop Jupyter, and that is deliberate.
#
# Measured topology on tylerinstance1 (17 Oct 2026):
#   PID 1        /usr/bin/tini -g -- jupyter notebook ... --port=38888
#   PID 6        jupyter-notebook        (child of tini)
#   container    NetworkMode=host, RestartPolicy=no
#
# Because the container uses the host network namespace, 38888 IS the host's
# port - there is no port mapping to change. Because tini forwards SIGTERM to
# its only child, `pkill -f jupyter` kills Jupyter, tini exits, and the
# container STOPS. With RestartPolicy=no nothing brings it back, so the old
# version of this script took the whole box down and left it down.
#
# The port swap therefore has to happen from the host, where the container can
# be stopped and replaced: see deploy/deploy_host.sh.
#
# This script is the command the replacement container runs. It is also useful
# inside the Jupyter container on a spare port, to check the server without
# disturbing 38888:
#
#   bash deploy/start_all.sh 38999
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ibm-user/IBMZProject}"
PUBLIC_PORT="${PUBLIC_PORT:-${1:-38888}}"

cd "$APP_DIR"

if [ -z "$(ls -A site/data 2>/dev/null)" ]; then
  echo "site/data is empty - build it first:" >&2
  echo "  python3 -m pipeline.build_artifacts --out site/data" >&2
  exit 1
fi

echo "serving $APP_DIR/site on 0.0.0.0:$PUBLIC_PORT"
exec python3 -m server.app --host 0.0.0.0 --port "$PUBLIC_PORT"
