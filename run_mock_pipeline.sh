#!/usr/bin/env bash
set -e
python -m pipeline.run_direct_generation_pipeline --config configs/env001_deepseek_rag.yaml --run-name mock_run --mock
echo ""
echo "Mock direct pipeline finished."
echo "核心输出：runs/env_001/mock_run/final_outputs/"
