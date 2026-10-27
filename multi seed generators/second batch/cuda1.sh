#!/usr/bin/env bash
# GPU 1 — adult, forest_cover (clf) + online_shopping (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 1 adult forest_cover online_shopping
