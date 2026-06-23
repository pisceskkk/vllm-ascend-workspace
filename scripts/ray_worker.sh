#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

ray stop -f

ray start \
    --address='10.0.0.1:6379' \
    --node-ip-address=10.0.0.2

echo "Ray worker joined"
