export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=1
nic_name="enp23s0f3"
local_ip="7.244.18.37"
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
#export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_BASE_PORT=50000
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512


export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export LD_LIBRARY_PAggTH=$LD_LIBRARY_PATH:/usr/local/lib
#export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTER_HCCS_DISABLE=TRUE
export VLLM_TORCH_PROFILER_DIR="./profile"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /mnt/sfs_turbo/wuhu-bucket-infer-1/psg/models/models/GLM-5.1-w8a8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --profiler-config \
    '{"profiler": "torch",
    "torch_profiler_dir": "./vllm_profile",
    "torch_profiler_with_stack": false}' \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 65536 \
    --additional-config '{"enable_dsa_cp": true, "fuse_muls_add":true, "recompute_scheduler_enable" : false, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": true}' \
    --max-num-batched-tokens 4096 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --max-num-seqs 32 \
    --quantization ascend \
    --gpu-memory-utilization 0.95 \
    --pipeline-parallel-size 1 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45
    # --kv-transfer-config \
    # '{"kv_connector": "MooncakeConnectorV1",
    # "kv_role": "kv_producer",
    # "kv_port": "30000",
    # "engine_id": "0",
    # "kv_connector_extra_config": {
    #             "prefill": {
    #                     "dp_size": 4,
    #                     "tp_size": 8
    #             },
    #             "decode": {
    #                     "dp_size": 32,
    #                     "tp_size": 1
    #             }
    #     }
    # }'
