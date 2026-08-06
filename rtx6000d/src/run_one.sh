#!/usr/bin/env bash
# Run one TP/EP config: restart server, wait ready, profile bs1 + bs64, stash traces.
# usage: run_one.sh <TP> <EP:0|1> <LABEL>
set -u
TP=$1; EP=$2; LABEL=$3
GPUS=$(seq -s, 0 $((TP-1)))
EXTRA=""
[ "$EP" = "1" ] && EXTRA="--enable-expert-parallel"

echo "=== config $LABEL: TP=$TP EP=$EP gpus=$GPUS $(date) ==="
docker rm -f dsv4-prof >/dev/null 2>&1
docker run -d --name dsv4-prof --gpus "\"device=$GPUS\"" --network host --ipc=host \
  -v /root/models/DeepSeek-V4-Flash-2L-NVFP4:/model:ro \
  -v /root/dsv4-prof/patches/workspace.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/workspace.py:ro \
  -v /root/dsv4-prof/patches/nvfp4.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:ro \
  -v /root/dsv4-prof/patches/quant_config.py:/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/quant_config.py:ro \
  -v /root/dsv4-prof/inner_prof.sh:/inner_prof.sh:ro \
  -v /root/dsv4-prof/traces:/traces \
  -v /root/dsv4-prof/cache:/cache \
  -e PORT=8100 -e MAXLEN=4096 -e UTIL=0.85 -e TP=$TP -e MAXSEQS=64 -e MNBT=2048 -e CGCAP=64 \
  -e EXTRA_ARGS="$EXTRA" \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e USES_B12X=True -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e B12X_MLA_SM120_UNIFIED=1 -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKSPACE_MAX_MB=512 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132 bash /inner_prof.sh >/dev/null

for i in $(seq 1 90); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health 2>/dev/null | grep -q 200 && break
  docker ps -q -f name=dsv4-prof | grep -q . || { echo "CONTAINER_DIED $LABEL"; docker logs dsv4-prof 2>&1 | grep -E "ERROR|Assertion" | tail -5; exit 1; }
  sleep 10
done
echo "ready $LABEL"

rm -f /root/dsv4-prof/traces/*
python3 /root/dsv4-prof/prof_client.py || { echo "BS1_FAILED $LABEL"; exit 1; }
sleep 5
python3 /root/dsv4-prof/prof_client_b64.py || { echo "BS64_FAILED $LABEL"; exit 1; }
mkdir -p /root/dsv4-prof/traces_$LABEL
rm -rf /root/dsv4-prof/traces_$LABEL/*
mv /root/dsv4-prof/traces/* /root/dsv4-prof/traces_$LABEL/
echo "DONE $LABEL $(ls /root/dsv4-prof/traces_$LABEL | wc -l) trace files"
