#!/usr/bin/env bash
# Runs once per container start, before the app. Fetches the data that lives
# in volume-mounted directories (Seed-VC repo under models/, Piper base
# checkpoint under piper_checkpoints/) — anything baked into those paths at
# image-build time would be shadowed by the bind mounts. Idempotent, and
# download failures only warn, so a network blip can't stop the app starting.
set -euo pipefail

bash /app/scripts/setup_linux.sh data

exec "$@"
