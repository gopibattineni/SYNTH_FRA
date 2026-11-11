#!/usr/bin/env bash
# GPU 2 — remaining: mushroom (clf) + concrete, energy_efficiency (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 2 mushroom concrete energy_efficiency
