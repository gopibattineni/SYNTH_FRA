#!/usr/bin/env bash
# GPU 2 — bank, wine (clf) + air_quality (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 2 bank wine air_quality
