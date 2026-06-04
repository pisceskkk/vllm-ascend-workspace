#!/usr/bin/env python3
"""
Offline vLLM launcher + concurrency tester using LLM API.
Builds on vllm_flash.sh + multicurl.sh patterns.

Usage:
  python vllm_launcher.py -m /mnt/share/z00919641/v4_w8a8
  python vllm_launcher.py -m /mnt/share/z00919641/v4_w8a8 --eager -c 20
  python vllm_launcher.py -m /mnt/share/z00919641/v4_w8a8 -p "你好" --max-tokens 128
"""

import argparse
import atexit
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime

# Env vars must be set BEFORE importing vllm
# os.environ.setdefault("HCCL_DETERMINISTIC", "true")
# os.environ.setdefault("VLLM_VERSION", "0.20.2")
# os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")

# --------------- defaults (mirror vllm_flash.sh) ---------------
DEFAULTS = {
    "model": "/path/to/weights",
    "tp_size": 8,
    "dp_size": 1,
    "enable_ep": True,
    "max_model_len": 1024,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 16,
    "gpu_memory_utilization": 0.95,
    "block_size": 128,
    "quantization": "ascend",
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
    "additional_config": {
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": False,
        },
        "enable_cpu_binding": "true",
        "multistream_overlap_shared_expert": False,
    },
    "enforce_eager": False,
    "visible_devices": "",
}

# ---------------- concurrency test defaults -----------------
TEST_PROMPT = "Who are you?"
TEST_MAX_TOKENS = 100
TEST_TEMPERATURE = 0.0
TEST_TOP_P = 0.95
DEFAULT_CONCURRENCY = 16


_llm = None  # tracked for cleanup


def _cleanup():
    """Best-effort cleanup on exit / signal."""
    global _llm
    # 1. Shutdown LLM gracefully
    if _llm is not None:
        try:
            del _llm
        except Exception:
            pass
        _llm = None
    # 2. Destroy torch distributed group if initialized
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    # 3. Kill leftover multiprocessing children
    try:
        import multiprocessing as mp
        for p in mp.active_children():
            try:
                p.terminate()
            except Exception:
                pass
    except Exception:
        pass


def _signal_handler(signum, frame):
    print(f"\n[launcher] Received signal {signum}, cleaning up...", flush=True)
    _cleanup()
    sys.exit(128 + signum)


atexit.register(_cleanup)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main():
    global _llm
    parser = argparse.ArgumentParser(
        description="Offline vLLM launcher + concurrency tester (LLM API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vllm_launcher.py -m /path/to/weight
  python vllm_launcher.py -m /path/to/weight --eager -c 20
  python vllm_launcher.py -m /path/to/weight -p "你好" --max-tokens 128
        """,
    )
    # ---- model / engine options ----
    parser.add_argument("-m", "--model", default=DEFAULTS["model"])
    parser.add_argument("-tp", "--tp-size", type=int, default=DEFAULTS["tp_size"])
    parser.add_argument("-dp", "--dp-size", type=int, default=DEFAULTS["dp_size"])
    parser.add_argument("-ep", "--enable-ep", type=bool, default=DEFAULTS["enable_ep"])
    parser.add_argument("--max-model-len", type=int, default=DEFAULTS["max_model_len"])
    parser.add_argument("--max-num-batched-tokens", type=int, default=DEFAULTS["max_num_batched_tokens"])
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULTS["max_num_seqs"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULTS["gpu_memory_utilization"])
    parser.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    parser.add_argument("--quantization", default=DEFAULTS["quantization"])
    parser.add_argument("--compilation-config", default=json.dumps(DEFAULTS["compilation_config"]))
    parser.add_argument("--additional-config", default=json.dumps(DEFAULTS["additional_config"]))
    parser.add_argument("-e", "--eager", dest="enforce_eager", action="store_true",
                        help="Enable enforce-eager mode")
    parser.add_argument('-v', "--visible-devices", default=DEFAULTS["visible_devices"],
                        help="ASCEND_RT_VISIBLE_DEVICES (e.g. 8,9,10,11,12,13,14,15)")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER,
                        help="Extra key=value pairs forwarded to LLM constructor")

    # ---- test options ----
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of concurrent prompts")
    parser.add_argument("-p", "--prompt", default=TEST_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=TEST_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEST_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TEST_TOP_P)

    args = parser.parse_args()

    # ---- apply late env vars ----
    if args.visible_devices:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
    if args.dbg_dir:
        args.dbg_dir = os.path.join(args.dbg_dir, "eager" if args.enforce_eager else "graph")
        os.environ["VLLM_ASCEND_DBG_DIR"] = args.dbg_dir

    # ---- backup old debug logs ----
    dbg_dir = os.environ.get("VLLM_ASCEND_DBG_DIR", "")
    if dbg_dir and os.path.isdir(dbg_dir):
        entries = os.listdir(dbg_dir)
        if entries:
            backup_root = os.path.join(os.path.dirname(dbg_dir), "backup")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(backup_root, f"{ts}_{os.path.basename(dbg_dir)}")
            os.makedirs(backup_dir, exist_ok=True)
            for name in entries:
                src = os.path.join(dbg_dir, name)
                dst = os.path.join(backup_dir, name)
                shutil.move(src, dst)
            print(f"[launcher] Backed up {len(entries)} old log(s) to {backup_dir}")
    # ---- end backup ----
    compilation_config = json.loads(args.compilation_config) if isinstance(args.compilation_config, str) else args.compilation_config
    additional_config = json.loads(args.additional_config) if isinstance(args.additional_config, str) else args.additional_config

    # ---- extra key=value args ----
    extra_kwargs = {}
    if args.extra_args:
        for kv in args.extra_args:
            key, _, value = kv.partition("=")
            if not key:
                continue
            # Try to infer types
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            extra_kwargs[key] = value

    # ---- late import: vllm after env is fully set up ----
    from vllm import LLM, SamplingParams

    print("[launcher] Initializing LLM...")
    print(f"[launcher] model={args.model}")
    print(f"[launcher] tp_size={args.tp_size}, dp_size={args.dp_size}")
    print(f"[launcher] max_model_len={args.max_model_len}, max_num_seqs={args.max_num_seqs}")
    print(f"[launcher] enforce_eager={args.enforce_eager}")

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        data_parallel_size=args.dp_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        enable_expert_parallel=args.enable_ep,
        compilation_config=compilation_config,
        additional_config=additional_config,
        **extra_kwargs,
    )
    _llm = llm
    init_elapsed = time.perf_counter() - t0
    print(f"[launcher] LLM ready, init took {init_elapsed:.1f}s")

    # ---- concurrency test ----
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    messages_list = [
        [{"role": "user", "content": args.prompt}]
        for _ in range(args.concurrency)
    ]

    # Tokenize first, then batch all prompts as TokensInput before engine runs
    tokenizer = llm.get_tokenizer()
    prompts = []
    for m in messages_list:
        token_ids = tokenizer.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True)
        prompts.append({"prompt_token_ids": token_ids, "type": "token"})

    print(f"\n[launcher] Running {args.concurrency} concurrent prompts "
          f"(debug queue → single batch)...")
    t1 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_elapsed = time.perf_counter() - t1

    # ---- results ----
    ok = fail = 0
    for i, output in enumerate(outputs):
        if output.outputs and output.outputs[0].text:
            ok += 1
        else:
            fail += 1
        content = output.outputs[0].text if output.outputs else "<empty>"
        print(f"  [{i+1:02d}] {content}")

    total_time = time.perf_counter() - t0
    print(f"\n[launcher] {ok}/{args.concurrency} succeeded, {fail} failed")
    print(f"[launcher] init={init_elapsed:.1f}s  generate={gen_elapsed:.1f}s  total={total_time:.1f}s")

    _llm = None
    _cleanup()


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
