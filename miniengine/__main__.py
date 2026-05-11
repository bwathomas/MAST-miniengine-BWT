"""
CLI entry point — launch the MiniEngine server.

Usage:
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --port 8080 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging

import torch
import uvicorn

from miniengine.engine import Engine
from miniengine.scheduler import Scheduler
from miniengine import server as srv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="miniengine",
        description="Minimal LLM serving engine",
    )
    p.add_argument(
        "--model", type=str, required=True, help="HuggingFace model id or local path"
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--device", type=str, default="cuda", help="Device to load model on")
    p.add_argument(
        "--max-running",
        type=int,
        default=16,
        help="Max concurrent requests in the scheduler",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="batched",
        choices=["baseline", "batched", "paged"],
        help="Scheduling mode: baseline (one request at a time), "
        "batched (iteration-level batching, milestone 1), or "
        "paged (M2 Part A — pre-allocated paged KV pool)",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Fraction of total GPU memory pre-allocated for static "
        "tensors (model weights + KV pool). Only used when --mode paged.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="Tokens per KV page. Only used when --mode paged. "
        "Must be a positive multiple of 256 (flash-attn 2.8+ requirement).",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Apply torch.compile to the MLP sub-region (M2 Part C required).",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture & replay paged-decode forward via CUDA graphs "
        "(M2 Part C extra credit). Requires --mode paged.",
    )
    p.add_argument(
        "--cuda-graph-batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated bucket batch sizes to capture when "
        "--cuda-graph is set; live batches are rounded UP to the nearest.",
    )
    # ── Iteration-loop knobs (default to no-op) ────────────────────────
    p.add_argument(
        "--flash-num-splits",
        type=int,
        default=0,
        help="Split-KV depth passed to flash_attn_with_kvcache "
        "(FlashAttention §4 IO-aware). 0 = let the kernel pick. On L4 "
        "the auto-heuristic often under-uses the 58 SMs; try 2/4/8 to "
        "boost paged-decode occupancy.",
    )
    p.add_argument(
        "--lpt-reorder",
        action="store_true",
        help="LPT virtual reorder of the live decode batch by descending "
        "KV length before the kernel call (FA4 §3.3 Longest-Processing-"
        "Time). Marginal on bandwidth-bound decode but free.",
    )
    p.add_argument(
        "--rope-cache-cap",
        type=int,
        default=Engine.DEFAULT_ROPE_CACHE_CAP,
        help="Pre-populate RoPE cos/sin cache to this length and lock "
        "further growth, removing one host-sync per forward "
        "(FlashInfer §D.1-derived). 0 = keep dynamic growth.",
    )
    p.add_argument(
        "--prefill-token-budget",
        type=int,
        default=0,
        help="Packed-prefill token budget τ (DistServe §3.1 / "
        "Sarathi-Serve §4.3). When >0, the scheduler caps the packed "
        "prefill at this many prompt tokens per step; spillover defers "
        "to the next step. 0 = no cap.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("miniengine")

    dtype = getattr(torch, args.dtype)
    logger.info(
        "Initializing engine  model=%s  dtype=%s  mode=%s",
        args.model,
        args.dtype,
        args.mode,
    )

    cuda_graph_batch_sizes = [
        int(x) for x in args.cuda_graph_batch_sizes.split(",") if x.strip()
    ]
    engine = Engine(
        model_path=args.model,
        dtype=dtype,
        device=args.device,
        mode=args.mode,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        torch_compile=args.torch_compile,
        cuda_graph=args.cuda_graph,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
        flash_num_splits=args.flash_num_splits,
        lpt_reorder=args.lpt_reorder,
        rope_cache_cap=args.rope_cache_cap,
        prefill_token_budget=args.prefill_token_budget,
    )
    sched = Scheduler(engine=engine, max_running=args.max_running, mode=args.mode)

    # Wire up the server module globals
    srv.engine = engine
    srv.scheduler = sched
    srv.model_id = args.model

    # Start scheduler background thread
    sched.start()

    logger.info("Starting server on %s:%d", args.host, args.port)
    try:
        uvicorn.run(srv.app, host=args.host, port=args.port, log_level="info")
    finally:
        sched.stop()


if __name__ == "__main__":
    main()
