#!/usr/bin/env bash
# GPU 3 — remaining: magic (clf) + real_estate (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 3 magic real_estate
