#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

# IB环境按需开启
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# 避免共享内存不足
export NCCL_SHM_DISABLE=0

ray stop -f

ray start \
    --head \
    --node-ip-address=10.0.0.1 \
    --port=6379 \
    --dashboard-host=0.0.0.0

echo "Ray head started"
