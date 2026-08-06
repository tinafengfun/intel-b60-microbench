#!/usr/bin/env bash
# Profiling inner script: 2-layer DeepSeek-V4-Flash NVFP4, TP=2, no MTP spec
set +B
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
  --served-model-name DeepSeek-V4-Flash --host 0.0.0.0 --port "${PORT:-8100}" \
  --kv-cache-dtype fp8 --block-size 256 --load-format safetensors \
  --tensor-parallel-size "${TP:-2}" --moe-backend b12x --linear-backend b12x \
  --gpu-memory-utilization "${UTIL:-0.85}" --max-model-len "${MAXLEN:-4096}" --max-num-seqs 1 \
  --max-num-batched-tokens "${MNBT:-512}" --max-cudagraph-capture-size 1 \
  --attention-backend B12X_MLA_SPARSE \
  --compilation-config="${COMPILATION:-{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"]}}" \
  --tokenizer-mode deepseek_v4 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_record_shapes":true,"torch_profiler_with_stack":true}' \
  --enable-flashinfer-autotune
