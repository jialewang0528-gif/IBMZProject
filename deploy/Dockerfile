# Reproducible image for the site. Built on the platform's own s390x base so
# the "runs on the mainframe" claim is not just a runtime accident.
#
# Build (on the LinuxONE host, no registry push needed):
#   docker build -f deploy/Dockerfile -t cs2-integrity:latest .
#
# Run (data is mounted read-only; artifacts are prebuilt into the image):
#   docker run -d --name cs2-site -p 38888:38888 \
#     -v /home/ibm-user/cs2_betting_data:/data:ro cs2-integrity:latest
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

USER root
WORKDIR /app

# stdlib-only server: no flask, no pyarrow (neither has an s390x wheel)
RUN pip install --no-cache-dir --upgrade "pandas==3.0.6" "scikit-learn==1.9.0" \
    "xgboost==3.4.1" "joblib" 2>/dev/null || true

COPY pipeline/ /app/pipeline/
COPY models/ /app/models/
COPY server/ /app/server/
COPY site/ /app/site/
COPY deploy/ /app/deploy/

ENV PYTHONUNBUFFERED=1 \
    CS2_DATA_ROOT=/data \
    CS2_ARTIFACT_DIR=/app/models/artifacts \
    PORT=38888

# artifacts must be built before the image is published, or the page has no data;
# the server starts anyway and reports 503 with the build hint.
EXPOSE 38888
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -sf http://127.0.0.1:38888/api/health >/dev/null || exit 1

CMD ["python3", "-m", "server.app", "--host", "0.0.0.0", "--port", "38888"]
