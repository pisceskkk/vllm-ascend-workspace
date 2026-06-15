rm ~/ascend/log/debug/ -rf
# export VLLM_VERSION=0.21.0

export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=4096
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
# export VLLM_PP_LAYER_PARTITION="41,37"

#export ASCEND_BUFFER_POOL=4:8
export ASCEND_ENABLE_USE_FABRIC_MEM=1 #HDK>=26.0

export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
# export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1

rm -rf ~/ascend/log/debug/plog/*
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
# --data-parallel-backend ray \
    # --distributed-executor-backend ray \
vllm serve /mnt/sfs_turbo/wuhu-bucket-infer-1/psg/models/models/GLM-5.1-w4a8 \
    --seed 1024 \
    --host 0.0.0.0 \
    --port 9000 \
    --served-model-name glm-5 \
    --max_model_len 6000 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.95 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    -dp 1 -pp 1 -tp 1 -pcp 16 \
    --cp-kv-cache-interleave-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 16, 64]}' \
    --additional-config '{"enable_dsa_cp": false, "ascend_compilation_config":{"enable_npugraph_ex": true, "enable_static_kernel": false},"fuse_muls_add":true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm":false, "enable_sparse_c8": false, "enable_cpu_binding":true, "recompute_scheduler_enable": false}' \
    --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager": "true"}' \
    --quantization ascend \
    --enable-expert-parallel \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    2>&1 | tee server.log
    # --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
    # --safetensors-load-strategy 'prefetch' \
    # , "eplb_config":{"dynamic_eplb":false, "expert_heat_collection_interval":600,"algorithm_execution_interval":50, "eplb_policy_type":2, "num_redundant_experts":16}
    # --hf-overrides '{"use_index_cache": false, "index_topk_freq": 4}' \
    # --kv-transfer-config \
    # '{"kv_connector": "MooncakeConnectorV1",
    # "kv_role": "kv_producer",
    # "kv_port": "30000",
    # "engine_id": "0",
    # "kv_connector_extra_config": {
    #             "prefill": {
    #                     "dp_size": 1,
    #                     "tp_size": 1,
    #     		"pp_size": 2
    #             },
    #             "decode": {
    #                     "dp_size": 32,
    #                     "tp_size": 1
    #             }
    #     }
    # }' \
