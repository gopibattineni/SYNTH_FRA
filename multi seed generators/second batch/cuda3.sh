#!/usr/bin/env bash
# GPU 3 — cdc_diabetes, mushroom (clf) + concrete (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 3 cdc_diabetes mushroom concrete
