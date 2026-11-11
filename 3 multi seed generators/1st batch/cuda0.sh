#!/usr/bin/env bash
# GPU 0 — remaining: forest_cover, wine (clf) + metro (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 0 forest_cover wine metro
