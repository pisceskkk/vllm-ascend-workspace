export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export VLLM_VERSION=0.20.2
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE="AIV"
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export HCCL_DETERMINISTIC=true
vllm serve /path/to/weight \
    --port 8091 \
    --served-model-name model \
    --safetensors-load-strategy 'prefetch' \
    --max_model_len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.95 \
    --api-server-count 1 \
    --max-num-seqs 256 \
    -dp 1 -tp 16 -pcp 1 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --quantization ascend \
    --enable-expert-parallel \
    2>&1 | tee server.log
    # --speculative-config '{"num_speculative_tokens": 3,"method": "deepseek_mtp", "enforce_eager": "true"}' \
    # --enforce-eager \
