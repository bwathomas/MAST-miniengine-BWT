"""Mooncake open-trace replay driver (Milestone 2 improvement item 9).

Replays the Mooncake LLM serving trace (Mooncake §4 + the open dataset at
https://github.com/kvcache-ai/Mooncake/tree/main/mooncake_trace) against a
running MiniEngine server's OpenAI-compatible ``/v1/chat/completions``
endpoint. Reports per-request TTFT/TPOT and aggregate throughput.

Why we want this
----------------
Our B-03 bottleneck (paged at conc=8 = 1.60× M1, below the 2× target) is
measured on the ShareGPT bench which has near-zero prompt-prefix overlap.
Mooncake's real-world trace contains ~40% shared-prefix sessions, which
is exactly the regime where the batch-2 ``--shared-prefix-cache`` work
(once we wire it in fully) and the ``--recompute-recovery`` admission
path are expected to dominate. Without a trace like this, B-03 numbers
underestimate paged's real-world advantage.

Trace format (lazy schema, just what we need)
---------------------------------------------
The Mooncake open trace is a CSV-ish JSONL with one record per request::

    {
      "timestamp": <unix-ms or relative-ms>,
      "input_length": <int>,
      "output_length": <int>,
      "hash_ids": [<int>, ...]   # optional, encodes shared-prefix structure
    }

We don't need the actual prompt text — synthesizing N tokens of dummy
text via the model's BOS-padded vocabulary is sufficient for the
benchmark. ``--hash-ids-to-prompt`` re-injects shared-prefix structure
by mapping each unique hash to a fixed token block; concatenating gives
prompts that share prefixes in lockstep with the trace.

Usage
-----
::

    # 1. start the engine
    python -m miniengine --model Qwen/Qwen3-8B --mode paged --port 8000 &

    # 2. replay (assumes trace is unpacked to ./mooncake.jsonl)
    python colab/mooncake_trace_replay.py \\
        --trace mooncake.jsonl \\
        --server http://localhost:8000 \\
        --limit 1000 \\
        --speedup 10.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover - optional, only needed at runtime
    aiohttp = None  # type: ignore


@dataclass
class TraceRecord:
    timestamp_s: float
    input_length: int
    output_length: int
    hash_ids: list[int] = field(default_factory=list)


@dataclass
class ResultRecord:
    request_id: int
    input_length: int
    requested_output: int
    actual_output: int
    arrival_s: float
    first_token_s: float | None
    last_token_s: float | None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.arrival_s) * 1000.0

    @property
    def total_s(self) -> float | None:
        if self.last_token_s is None:
            return None
        return self.last_token_s - self.arrival_s

    @property
    def tpot_ms(self) -> float | None:
        if (
            self.last_token_s is None
            or self.first_token_s is None
            or self.actual_output <= 1
        ):
            return None
        return (
            (self.last_token_s - self.first_token_s)
            / (self.actual_output - 1)
            * 1000.0
        )


def load_trace(path: Path, limit: int | None = None) -> list[TraceRecord]:
    """Load JSONL trace records, normalize timestamps to start at 0."""
    records: list[TraceRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            ts_raw = obj.get("timestamp", obj.get("arrival_time", i))
            ts_s = float(ts_raw) / 1000.0 if ts_raw > 1e5 else float(ts_raw)
            records.append(
                TraceRecord(
                    timestamp_s=ts_s,
                    input_length=int(obj.get("input_length", 0)),
                    output_length=int(obj.get("output_length", 0)),
                    hash_ids=list(obj.get("hash_ids", [])),
                )
            )
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"no records loaded from {path}")
    t0 = records[0].timestamp_s
    for r in records:
        r.timestamp_s -= t0
    return records


def synthesize_prompt(
    record: TraceRecord, vocab_block_size: int = 64
) -> str:
    """Build a deterministic prompt of approximately ``input_length`` tokens.

    Uses simple repeated ASCII to keep tokenization cheap. Real Mooncake
    replays should ideally restore prefix structure via ``hash_ids``;
    this stub just emits length-correct filler. With a 1:1 mapping from
    word to BPE token, the prompt length in tokens is approximately
    ``input_length`` — the actual count is close enough for throughput
    benchmarking since the engine handles whatever length comes in.
    """
    n_words = max(1, record.input_length)
    if record.hash_ids:
        # Reconstruct shared-prefix structure: each hash → ``vocab_block_size``
        # repetitions of a short literal that's unique to that hash.
        parts: list[str] = []
        n = 0
        for h in record.hash_ids:
            piece = f"hash{h:08x} " * vocab_block_size
            parts.append(piece)
            n += vocab_block_size
            if n >= n_words:
                break
        return ("".join(parts))[: n_words * 8]
    return ("the quick brown fox jumps over the lazy dog ") * (n_words // 9 + 1)


async def _replay_one(
    session: Any,
    server: str,
    record: TraceRecord,
    request_id: int,
    arrival_s: float,
    model: str,
) -> ResultRecord:
    prompt = synthesize_prompt(record)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, record.output_length),
        "stream": True,
        "temperature": 0.6,
    }
    first_token_s: float | None = None
    last_token_s: float | None = None
    actual_output = 0
    async with session.post(f"{server}/v1/chat/completions", json=payload) as resp:
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body in ("[DONE]", ""):
                continue
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta", {}) or {}
            if not delta.get("content"):
                continue
            now = time.perf_counter()
            if first_token_s is None:
                first_token_s = now
            last_token_s = now
            actual_output += 1
    return ResultRecord(
        request_id=request_id,
        input_length=record.input_length,
        requested_output=record.output_length,
        actual_output=actual_output,
        arrival_s=arrival_s,
        first_token_s=first_token_s,
        last_token_s=last_token_s,
    )


async def run_replay(
    trace: list[TraceRecord],
    server: str,
    model: str,
    speedup: float,
    max_concurrent: int,
) -> list[ResultRecord]:
    if aiohttp is None:
        raise RuntimeError(
            "aiohttp is required: pip install aiohttp"
        )
    sem = asyncio.Semaphore(max_concurrent)
    timeout = aiohttp.ClientTimeout(total=None)
    start = time.perf_counter()

    async def launch(idx: int, rec: TraceRecord, session: Any) -> ResultRecord:
        delay = rec.timestamp_s / max(speedup, 1e-9)
        wait = (start + delay) - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        async with sem:
            arrival = time.perf_counter()
            return await _replay_one(session, server, rec, idx, arrival, model)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(launch(i, rec, session))
            for i, rec in enumerate(trace)
        ]
        return await asyncio.gather(*tasks)


def _summarize(results: list[ResultRecord]) -> dict[str, Any]:
    completed = [r for r in results if r.last_token_s is not None]
    if not completed:
        return {"completed": 0, "total": len(results)}
    ttfts = [r.ttft_ms for r in completed if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in completed if r.tpot_ms is not None]
    total_tokens = sum(r.actual_output for r in completed)
    wallclock = max(r.last_token_s for r in completed) - min(  # type: ignore[type-var]
        r.arrival_s for r in completed
    )
    return {
        "completed": len(completed),
        "total": len(results),
        "total_output_tokens": total_tokens,
        "throughput_tok_per_s": total_tokens / wallclock if wallclock > 0 else 0.0,
        "ttft_ms_p50": statistics.median(ttfts) if ttfts else None,
        "ttft_ms_p95": (
            statistics.quantiles(ttfts, n=20)[-1] if len(ttfts) >= 20 else None
        ),
        "tpot_ms_p50": statistics.median(tpots) if tpots else None,
        "tpot_ms_p95": (
            statistics.quantiles(tpots, n=20)[-1] if len(tpots) >= 20 else None
        ),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Replay a Mooncake-format trace against a MiniEngine server."
    )
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--server", type=str, default="http://localhost:8000")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Compress real time by this factor (e.g. 10.0 makes the "
        "trace replay 10× faster than wall-clock).",
    )
    p.add_argument("--max-concurrent", type=int, default=64)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    trace = load_trace(args.trace, limit=args.limit)
    print(f"Loaded {len(trace)} trace records from {args.trace}")
    results = asyncio.run(
        run_replay(
            trace,
            args.server,
            args.model,
            speedup=args.speedup,
            max_concurrent=args.max_concurrent,
        )
    )
    summary = _summarize(results)
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    if args.out:
        args.out.write_text(json.dumps({"results": [r.__dict__ for r in results], "summary": summary}))
        print(f"Wrote detailed results to {args.out}")


if __name__ == "__main__":
    main()
