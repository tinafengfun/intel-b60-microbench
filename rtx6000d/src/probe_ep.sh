#!/usr/bin/env bash
# Probe EP-capable MoE backends: try each, report which loads.
set -u
TP=2
GPUS="0,1"
for BACKEND in flashinfer_cutlass cutlass marlin flashinfer_cutedsl emulation; do
  echo "=== probing backend=$BACKEND $(date) ==="
  docker rm -f dsv4-probe >/dev/null 2>&1
  docker run -d --name dsv4-probe --gpus "\"device=$GPUS\"" --network host --ipc=host \
    -v /root/models/DeepSeek-V4-Flash-2L-NVFP4:/model:ro \
    -v /root/dsv4-prof/patches/workspace.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/workspace.py:ro \
    -v /root/dsv4-prof/patches/nvfp4.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:ro \
    -v /root/dsv4-prof/patches/quant_config.py:/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/quant_config.py:ro \
    -v /root/dsv4-prof/inner_prof.sh:/inner_prof.sh:ro \
    -v /root/dsv4-prof/traces:/traces \
    -v /root/dsv4-prof/cache:/cache \
    -e PORT=8101 -e MAXLEN=4096 -e UTIL=0.85 -e TP=$TP -e MAXSEQS=64 -e MNBT=2048 -e CGCAP=64 \
    -e EXTRA_ARGS="--enable-expert-parallel --moe-backend $BACKEND" \
    -e VLLM_CACHE_DIR=/cache/jit/vllm \
    -e USES_B12X=True -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_MHC=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e B12X_MLA_SM120_UNIFIED=1 -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKSPACE_MAX_MB=512 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
    voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132 bash /inner_prof.sh >/dev/null
  STATUS=TIMEOUT
  for i in $(seq 1 42); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8101/health 2>/dev/null | grep -q 200; then STATUS=OK; break; fi
    if ! docker ps -q -f name=dsv4-probe | grep -q .; then
      STATUS=$(docker logs dsv4-probe 2>&1 | grep -oE "ValueError: [^\"]{0,160}|RuntimeError: [^\"]{0,160}|AssertionError[^\"]{0,120}" | head -1)
      [ -z "$STATUS" ] && STATUS="DIED(no error line)"
      break
    fi
    sleep 10
  done
  echo "RESULT backend=$BACKEND status=$STATUS"
  docker rm -f dsv4-probe >/dev/null 2>&1
  [ "$STATUS" = "OK" ] && break
done
echo PROBE_DONE
