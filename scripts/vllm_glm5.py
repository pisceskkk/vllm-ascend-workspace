#!/usr/bin/env python3
"""Run GLM-5 offline inference with the vLLM Python API on Linux/Ascend.

This is a pure Python offline script: it sets the environment from
vllm_glm5.sh, constructs vllm.LLM directly, runs one or more chat requests,
and exits. It does not start an OpenAI API server and does not call curl.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


DEFAULT_MODEL_PATH = "/mnt/share/glm5-w4a8-new"
DEFAULT_PROMPT = "你是谁？"

DEFAULT_ENV = {
    "VLLM_ASCEND_ENABLE_NZ": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "OMP_PROC_BIND": "false",
    "OMP_NUM_THREADS": "20",
    "HCCL_BUFFSIZE": "768",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_SERVER_DEV_MODE": "1",
    "ASCEND_ENABLE_USE_FABRIC_MEM": "1",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
    "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
    "PYTHONHASHSEED": "0",
    "VLLM_ENGINE_READY_TIMEOUT_S": "10000",
    "VLLM_RPC_TIMEOUT": "3600000",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
    "TASK_QUEUE_ENABLE": "1",
    "CPU_AFFINITY_CONF": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}

# Disabled environment options copied from vllm_glm5.sh for later use:
# DEFAULT_ENV["VLLM_VERSION"] = "0.21.0"
# DEFAULT_ENV["VLLM_PP_LAYER_PARTITION"] = "41,37"
# DEFAULT_ENV["ASCEND_BUFFER_POOL"] = "4:8"
# DEFAULT_ENV["VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE"] = "1"
DEFAULT_ENV["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7,8,9,10,11"
#DEFAULT_ENV["ASCEND_RT_VISIBLE_DEVICES"] = "12,13,14,15"
# DEFAULT_ENV["ASCEND_RT_VISIBLE_DEVICES"] = "8,9,10,11,12,13,14,15"
# DEFAULT_ENV["HCCL_DETERMINISTIC"] = "true"

COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [4, 16, 64],
}

ADDITIONAL_CONFIG = {
    "sfa_dcp_replicate_k": True,
    "enable_dsa_cp": False,
    "ascend_compilation_config": {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
    },
    "fuse_muls_add": True,
    "multistream_overlap_shared_expert": True,
    "enable_mc2_hierarchy_comm": False,
    "enable_sparse_c8": False,
    "enable_cpu_binding": True,
    "recompute_scheduler_enable": False,
    # Disabled additional config copied from vllm_glm5.sh for later use:
    # "eplb_config": {
    #     "dynamic_eplb": False,
    #     "expert_heat_collection_interval": 600,
    #     "algorithm_execution_interval": 50,
    #     "eplb_policy_type": 2,
    #     "num_redundant_experts": 16,
    # },
}

# Disabled LLM options copied from vllm_glm5.sh for later use:
# ALT_MODEL_PATH = "/mnt/sfs_turbo/wuhu-bucket-infer-1/psg/models/models/GLM-5.1-w4a8"
# LLM_KWARGS["data_parallel_backend"] = "ray"
# LLM_KWARGS["distributed_executor_backend"] = "ray"
# LLM_KWARGS["speculative_config"] = {
#     "num_speculative_tokens": 1,
#     "method": "deepseek_mtp",
#     "enforce_eager": "true",
# }
# LLM_KWARGS["profiler_config"] = {
#     "profiler": "torch",
#     "torch_profiler_dir": "./vllm_profile",
#     "torch_profiler_with_stack": False,
# }
# LLM_KWARGS["hf_overrides"] = {"use_index_cache": False, "index_topk_freq": 4}
# LLM_KWARGS["kv_transfer_config"] = {
#     "kv_connector": "MooncakeConnectorV1",
#     "kv_role": "kv_producer",
#     "kv_port": "30000",
#     "engine_id": "0",
#     "kv_connector_extra_config": {
#         "prefill": {"dp_size": 1, "tp_size": 1, "pp_size": 2},
#         "decode": {"dp_size": 32, "tp_size": 1},
#     },
# }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure Python offline GLM-5 inference with vLLM.",
    )
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="User prompt. Repeat this option for batch inference.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Text file with one prompt per line, or a JSON list of strings.",
    )
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=6000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument("--cp-kv-cache-interleave-size", type=int, default=128)
    parser.add_argument("--quantization", default="ascend")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=False)
    parser.add_argument("--disable-expert-parallel", action="store_true")
    parser.add_argument("--disable-eager", action="store_true")
    parser.add_argument("--enable-async-scheduling", action="store_true")
    parser.add_argument("--safetensors-load-strategy", default="prefetch")
    parser.add_argument("--skip-ascend-log-cleanup", action="store_true")
    parser.add_argument("--skip-cpu-governor", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def apply_runtime_env() -> None:
    for name, value in DEFAULT_ENV.items():
        os.environ[name] = value


def prepare_ascend_logs(skip_cleanup: bool) -> None:
    if skip_cleanup:
        return
    shutil.rmtree(Path.home() / "ascend" / "log" / "debug", ignore_errors=True)


def set_cpu_governor(skip_governor: bool) -> None:
    if skip_governor:
        return

    cpu_root = Path("/sys/devices/system/cpu")
    for governor_path in cpu_root.glob("cpu*/cpufreq/scaling_governor"):
        try:
            governor_path.write_text("performance\n", encoding="utf-8")
        except OSError as exc:
            print(
                f"Warning: failed to set {governor_path} to performance: {exc}",
                file=sys.stderr,
            )


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        content = prompt_path.read_text(encoding="utf-8")
        try:
            loaded = json.loads(content)
        except json.JSONDecodeError:
            loaded = [line.strip() for line in content.splitlines() if line.strip()]

        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("--prompt-file must contain text lines or a JSON list of strings")
        prompts.extend(loaded)

    return prompts or [DEFAULT_PROMPT]


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "data_parallel_size": args.dp,
        "pipeline_parallel_size": args.pp,
        "tensor_parallel_size": args.tp,
        "prefill_context_parallel_size": args.pcp,
        "decode_context_parallel_size": args.dcp,
        "cp_kv_cache_interleave_size": args.cp_kv_cache_interleave_size,
        "compilation_config": COMPILATION_CONFIG,
        "additional_config": ADDITIONAL_CONFIG,
        "quantization": args.quantization,
        "enable_expert_parallel": not args.disable_expert_parallel,
        "enforce_eager": not args.disable_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "async_scheduling": args.enable_async_scheduling,
        "safetensors_load_strategy": args.safetensors_load_strategy,
    }
    return llm_kwargs


def build_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=None,
    )


def run_offline_inference(args: argparse.Namespace, prompts: list[str]) -> list[dict[str, Any]]:
    # Import vLLM after applying env vars so plugin/runtime flags are visible.
    from vllm import LLM

    llm_kwargs = build_llm_kwargs(args)
    sampling_params = build_sampling_params(args)
    llm = LLM(**llm_kwargs)

    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)

    results = []
    for prompt, output in zip(prompts, outputs):
        completion = output.outputs[0]
        results.append(
            {
                "prompt": prompt,
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "token_ids": completion.token_ids,
            }
        )

    del llm
    cleanup_vllm_runtime()
    return results


def cleanup_vllm_runtime() -> None:
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()

    with contextlib.suppress(Exception):
        import torch

        torch.npu.empty_cache()


def format_results(results: list[dict[str, Any]], pretty: bool) -> str:
    if pretty:
        return json.dumps(results, ensure_ascii=False, indent=2)
    return json.dumps(results, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    apply_runtime_env()
    prepare_ascend_logs(args.skip_ascend_log_cleanup)
    set_cpu_governor(args.skip_cpu_governor)

    prompts = load_prompts(args)
    results = run_offline_inference(args, prompts)
    output_text = format_results(results, args.pretty)

    print(output_text)
    if args.output_file:
        Path(args.output_file).write_text(output_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
