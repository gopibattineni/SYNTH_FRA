#!/usr/bin/env bash
# GPU 4 — magic (clf) + energy_efficiency, real_estate (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 4 magic energy_efficiency real_estate
