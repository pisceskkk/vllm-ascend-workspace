#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL_PATH_DEFAULT = "/mnt/share/glm5-w4a8-new"
ASCEND_ENV_DEFAULT = "/usr/local/Ascend/ascend-toolkit/set_env.sh"


def log(msg: str) -> None:
    print(f"[offline-profile] {msg}", flush=True)


def source_ascend_env(env_script: str) -> None:
    env_path = Path(env_script).expanduser()
    if not env_path.exists():
        log(f"WARN: Ascend env script not found: {env_path}")
        return

    cmd = f"source {shlex.quote(str(env_path))} >/dev/null 2>&1 && env -0"
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        log(f"WARN: failed to source Ascend env: {env_path}")
        log(proc.stderr.decode(errors="ignore"))
        return

    for item in proc.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        os.environ[k.decode(errors="ignore")] = v.decode(errors="ignore")

    log(f"loaded Ascend env: {env_path}")


def setup_env(devices: str) -> None:
    # 复刻你的在线脚本环境变量；必须在 import vllm 前设置。
    envs = {
        "VLLM_ASCEND_ENABLE_NZ": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "20",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "ASCEND_ENABLE_USE_FABRIC_MEM": "1",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "0",
        "PYTHONHASHSEED": "0",
        "VLLM_ENGINE_READY_TIMEOUT_S": "10000",
        "VLLM_RPC_TIMEOUT": "3600000",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
        "TASK_QUEUE_ENABLE": "1",
        "CPU_AFFINITY_CONF": "1",
        "ASCEND_RT_VISIBLE_DEVICES": devices,
    }

    for k, v in envs.items():
        os.environ[k] = v

    log(f"ASCEND_RT_VISIBLE_DEVICES={devices}")


def clean_logs() -> None:
    debug_dir = Path.home() / "ascend/log/debug"
    shutil.rmtree(debug_dir, ignore_errors=True)
    (debug_dir / "plog").mkdir(parents=True, exist_ok=True)
    log(f"cleaned {debug_dir}")


def set_cpu_governor() -> None:
    ok = 0
    for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            Path(path).write_text("performance\n")
            ok += 1
        except Exception:
            pass

    if ok:
        log(f"set CPU governor to performance for {ok} CPUs")
    else:
        log("WARN: failed to set CPU governor; continue")


def check_msprof() -> None:
    msprof = shutil.which("msprof")
    if not msprof:
        raise RuntimeError(
            "msprof not found. 请先 source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        )
    log(f"msprof={msprof}")


def build_prompts(num_prompts: int) -> list[str]:
    prompts = [
        "请用一句话介绍北京。",
        "请用一句话介绍上海。",
        "请用一句话介绍广州。",
        "请用一句话介绍深圳。",
        "把 hello 翻译成中文。",
        "把 apple 翻译成中文。",
        "写一句关于星星的话。",
        "写一句关于大海的话。",
        "给出一个简单数学事实。",
        "给出一个简单物理事实。",
        "用一句话解释机器学习。",
        "用一句话解释数据库。",
        "列出一种常见水果。",
        "列出一种常见动物。",
        "说一句鼓励的话。",
        "说一句天气相关的话。",
    ]

    if num_prompts <= len(prompts):
        return prompts[:num_prompts]

    return [prompts[i % len(prompts)] + f" 编号{i}" for i in range(num_prompts)]


def compute_output_tokens(
    decode_rounds: int,
    speculative_tokens: int,
    output_tokens: int | None,
) -> int:
    if output_tokens is not None:
        return output_tokens

    # 开启 deepseek_mtp 且 num_speculative_tokens=3 时，
    # 单个 decode worker step 可能产生多个 token。
    #
    # 为了确保至少能推进：
    #   delay + warmup + 20 active + RECORD_AND_SAVE + NONE/auto-stop
    # 这里给足 token budget。
    #
    # 默认：decode_rounds=20, speculative_tokens=3
    # output_tokens = (20 + 4) * 4 + 8 = 104
    return (decode_rounds + 4) * (speculative_tokens + 1) + 8


def find_profile_dirs(profile_dir: Path) -> list[Path]:
    dirs = [p.resolve() for p in profile_dir.rglob("*_ascend_pt") if p.is_dir()]
    return sorted(set(dirs), key=lambda p: p.stat().st_mtime)


def analyse_profile_dirs(profile_dirs: list[Path]) -> None:
    for profile_dir in profile_dirs:
        log(f"analyse profile dir: {profile_dir}")
        code = (
            "from torch_npu.profiler.profiler import analyse\n"
            f"analyse({str(profile_dir)!r})\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def print_profile_markers(profile_dirs: list[Path]) -> None:
    for profile_dir in profile_dirs:
        log(f"profile dir: {profile_dir}")

        markers: list[Path] = []
        markers.extend(profile_dir.rglob("*start_info*"))
        markers.extend(profile_dir.rglob("info.json"))
        markers.extend(profile_dir.rglob("host_start.log"))

        if not markers:
            log("WARN: no start_info/info.json/host_start.log found")
        else:
            log("marker files:")
            for marker in markers[:30]:
                log(f"  {marker}")
            if len(markers) > 30:
                log(f"  ... {len(markers) - 30} more")

        output_dir = profile_dir / "ASCEND_PROFILER_OUTPUT"
        if output_dir.exists():
            log(f"ASCEND_PROFILER_OUTPUT={output_dir}")
            for name in [
                "analysis.db",
                "api_statistic.csv",
                "kernel_details.csv",
                "operator_details.csv",
                "op_statistic.csv",
                "step_trace_time.csv",
                "trace_view.json",
            ]:
                for f in output_dir.rglob(name):
                    log(f"  {f}")

            for f in output_dir.rglob("ascend_pytorch_profiler_*.db"):
                log(f"  {f}")
        else:
            log("WARN: ASCEND_PROFILER_OUTPUT not found yet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--devices", default="4,5,6,7,8,9,10,11")
    parser.add_argument("--ascend-env", default=ASCEND_ENV_DEFAULT)
    parser.add_argument("--profile-dir", default="./vllm_profile_offline")

    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--decode-rounds", type=int, default=20)
    parser.add_argument("--output-tokens", type=int, default=None)

    parser.add_argument("--speculative-tokens", type=int, default=3)

    # 默认复刻你的在线脚本：torch_profiler_with_stack=true。
    # 如果 CANN 数据太大或解析慢，临时加 --no-stack。
    parser.add_argument("--no-stack", action="store_true")

    parser.add_argument("--skip-analyse", action="store_true")
    parser.add_argument("--no-clean-profile", action="store_true")
    parser.add_argument("--flush-wait-sec", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_ascend_env(args.ascend_env)
    setup_env(args.devices)
    clean_logs()
    set_cpu_governor()
    check_msprof()

    profile_root = Path(args.profile_dir).expanduser().resolve()
    if profile_root.exists() and not args.no_clean_profile:
        shutil.rmtree(profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)

    log(f"model={args.model}")
    log(f"profile_root={profile_root}")

    # vLLM 必须在环境变量设置后 import。
    from vllm import LLM, SamplingParams

    output_tokens = compute_output_tokens(
        decode_rounds=args.decode_rounds,
        speculative_tokens=args.speculative_tokens,
        output_tokens=args.output_tokens,
    )

    prompts = build_prompts(args.num_prompts)

    # 关键修复：
    #   1. warmup_iterations > 0，强制启用 torch.profiler.schedule。
    #   2. active_iterations=20，真正采 20 个 active decode worker step。
    #   3. max_iterations=20，让 vLLM 在 active 计数超过 20 后自动 stop。
    #   4. 由于 PyTorch schedule 最后一个 active step 是 RECORD_AND_SAVE，
    #      再多给 token budget，让 profiler 有机会推进到 NONE/auto-stop，
    #      避免 stop 时仍处于 RECORD。
    profiler_config: dict[str, Any] = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_root),
        "torch_profiler_with_stack": not args.no_stack,

        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_with_flops": False,

        "ignore_frontend": True,

        "delay_iterations": 1,

        # schedule-based profiling
        "wait_iterations": 0,
        "warmup_iterations": 1,
        "active_iterations": args.decode_rounds,

        # auto-stop after the active profiling window
        "max_iterations": args.decode_rounds,
    }

    compilation_config: dict[str, Any] = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [4, 16, 64, 128],
    }

    additional_config: dict[str, Any] = {
        "sfa_dcp_replicate_k": True,
        "enable_dsa_cp": False,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": False,
        },
        "fuse_muls_add": True,
        "multistream_overlap_shared_expert": True,
        "enable_mc2_hierarchy_comm": False,
        "enable_sparse_c8": True,
        "enable_cpu_binding": True,
        "recompute_scheduler_enable": False,
    }

    speculative_config: dict[str, Any] = {
        "num_speculative_tokens": args.speculative_tokens,
        "method": "deepseek_mtp",
        "enforce_eager": True,
    }

    llm_kwargs: dict[str, Any] = {
        # vllm serve /mnt/share/glm5-w4a8-new
        "model": args.model,

        # --seed 1024
        "seed": 1024,

        # --max_model_len 8192
        "max_model_len": 8192,

        # --max-num-batched-tokens 2048
        "max_num_batched_tokens": 2048,

        # --gpu-memory-utilization 0.95
        "gpu_memory_utilization": 0.95,

        # --max-num-seqs 32
        "max_num_seqs": 32,

        # -dp 1 -pp 1 -tp 8 -pcp 1 -dcp 8
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 8,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 8,

        # --cp-kv-cache-interleave-size 1
        "cp_kv_cache_interleave_size": 1,

        # --compilation-config
        "compilation_config": compilation_config,

        # --additional-config
        "additional_config": additional_config,

        # --quantization ascend
        "quantization": "ascend",

        # --enable-expert-parallel
        "enable_expert_parallel": True,

        # --safetensors-load-strategy prefetch
        "safetensors_load_strategy": "prefetch",

        # --profiler-config
        "profiler_config": profiler_config,

        # --speculative-config
        "speculative_config": speculative_config,
    }

    log("LLM kwargs:")
    log(json.dumps(llm_kwargs, ensure_ascii=False, indent=2))

    log("creating offline LLM...")
    llm = LLM(**llm_kwargs)

    try:
        # 预热，不进 profiling。
        warmup_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4,
            min_tokens=4,
            ignore_eos=True,
        )
        log("warmup generate...")
        llm.generate(prompts, warmup_sampling)

        # 16 条短请求。
        # min_tokens + ignore_eos 保证每条请求都跑够 output_tokens，
        # 从而覆盖 20 轮 decode active step。
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=output_tokens,
            min_tokens=output_tokens,
            ignore_eos=True,
        )

        log(
            "start profiling: "
            f"num_prompts={len(prompts)}, "
            f"decode_rounds={args.decode_rounds}, "
            f"output_tokens_per_prompt={output_tokens}, "
            f"with_stack={not args.no_stack}"
        )

        started = False
        outputs = None

        try:
            llm.start_profile()
            started = True

            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - t0

            log(f"profiled generate done, elapsed={elapsed:.3f}s")

        finally:
            # 如果 max_iterations 已经自动 stop，这里只清 active 状态；
            # 如果仍未自动 stop，这里会暴露问题，但 output_tokens 默认已经留足余量。
            if started:
                log("stop_profile")
                llm.stop_profile()

        if outputs is not None:
            log("generated token counts:")
            for i, out in enumerate(outputs):
                token_ids = out.outputs[0].token_ids
                text = out.outputs[0].text.replace("\n", "\\n")
                log(f"  req[{i:02d}] tokens={len(token_ids)}, text_prefix={text[:80]!r}")

    finally:
        log("shutdown LLM")
        try:
            if hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception as e:
            log(f"WARN: llm.shutdown failed: {repr(e)}")

        del llm
        gc.collect()

    log(f"wait {args.flush_wait_sec}s for profiler files flushing...")
    time.sleep(args.flush_wait_sec)

    profile_dirs = find_profile_dirs(profile_root)
    if not profile_dirs:
        raise RuntimeError(f"No *_ascend_pt profile directory found under {profile_root}")

    log("found profile dirs:")
    for d in profile_dirs:
        log(f"  {d}")

    print_profile_markers(profile_dirs)

    if not args.skip_analyse:
        analyse_profile_dirs(profile_dirs)
        print_profile_markers(profile_dirs)
    else:
        log("skip analyse")

    log(f"done. profile_root={profile_root}")


if __name__ == "__main__":
    main()
