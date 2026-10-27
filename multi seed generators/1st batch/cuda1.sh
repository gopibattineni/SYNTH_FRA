#!/usr/bin/env bash
# GPU 1 — remaining: cdc_diabetes (clf) + online_shopping, air_quality (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 1 cdc_diabetes online_shopping air_quality
