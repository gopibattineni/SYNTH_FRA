#!/usr/bin/env bash
# GPU 0 — cancer, alzhimers (clf) + metro (reg)
exec "$(cd "$(dirname "$0")" && pwd)/run_cuda_worker.sh" 0 cancer alzhimers metro
