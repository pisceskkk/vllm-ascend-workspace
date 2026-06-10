rm ~/ascend/log/debug/ -rf
# export VLLM_VERSION=0.21.0

export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=2048
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
# export VLLM_PP_LAYER_PARTITION="41,37"

#export ASCEND_BUFFER_POOL=4:8
export ASCEND_ENABLE_USE_FABRIC_MEM=1 #HDK>=26.0

export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
# export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1

export PYTHONHASHSEED=0

export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export HCCL_DETERMINISTIC=true
vllm serve /path/to/weight \
    --seed 1024 \
    --host 0.0.0.0 \
    --port 9000 \
    --served-model-name model \
    --max_model_len 66000 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.85 \
    --api-server-count 1 \
    --max-num-seqs 24 \
    -dp 1 -pp 2 -tp 1 -pcp 16 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --quantization ascend \
    --enable-expert-parallel \
    --additional-config '{"enable_dsa_cp": false, "ascend_compilation_config":{"enable_npugraph_ex": true, "enable_static_kernel": false},"fuse_muls_add":true, "multistream_overlap_shared_expert": false, "enable_mc2_hierarchy_comm":false, "enable_sparse_c8": false, "enable_cpu_binding":true}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
    --enforce-eager \
    2>&1 | tee server.log
    # --hf-overrides '{"use_index_cache": true, "index_topk_freq": 2}' \
    # --safetensors-load-strategy 'prefetch' \
    # , "eplb_config":{"dynamic_eplb":false, "expert_heat_collection_interval":600,"algorithm_execution_interval":50, "eplb_policy_type":2, "num_redundant_experts":16}
    # --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager": "true"}' \
