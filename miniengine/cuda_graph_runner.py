"""Manual CUDA graph capture and replay for paged decode (M2 Part C extra credit).

The runner owns one ``torch.cuda.CUDAGraph`` per *bucket batch size*. A live
decode batch of size ``B_live`` is rounded UP to the smallest captured bucket
``B >= B_live``; its graph is replayed after fresh data is ``copy_``-ed into
stable per-bucket input buffers. Sampling and per-step list/tensor building
all stay outside the captured region.

What gets captured: ``CausalLM.forward(input_ids, position_ids,
kv_caches=<pool views>, paged_metadata=<stable meta>)`` — the same call the
M2 Part B paged decode (``flash_attn_with_kvcache`` via PagedAttentionMetadata)
already issues. The pool's per-layer K/V tensors and the metadata's
``block_table`` / ``cache_seqlens`` are the stable identities the graph
records; we ``.copy_`` fresh contents in on every step.

The ``RotaryEmbedding`` cos/sin cache is pre-populated and its forward is
replaced with a sync-free lookup so the captured region contains no
``.item()`` (which would be illegal during graph capture).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from miniengine.model import PagedAttentionMetadata, RotaryEmbedding

if TYPE_CHECKING:
    from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class CudaGraphRunner:
    DEFAULT_MAX_PAGES_PER_SEQ = 128
    DEFAULT_ROPE_CAP = 8192

    def __init__(
        self,
        engine: "Engine",
        batch_sizes: list[int],
        max_pages_per_seq: int = DEFAULT_MAX_PAGES_PER_SEQ,
        rope_cache_cap: int = DEFAULT_ROPE_CAP,
        warmup_iters: int = 3,
    ) -> None:
        self.engine = engine
        sizes = sorted({int(b) for b in batch_sizes if int(b) > 0})
        if not sizes:
            raise ValueError("CudaGraphRunner requires at least one positive bucket")
        self.batch_sizes: list[int] = sizes
        self.max_pages_per_seq: int = int(max_pages_per_seq)
        self.rope_cache_cap: int = int(rope_cache_cap)
        self.warmup_iters: int = int(warmup_iters)

        self._captured: bool = False
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._inputs: dict[int, dict[str, torch.Tensor]] = {}
        # Persistent pinned host scratch tensors per bucket. Replay stages
        # per-step inputs here in bulk (no per-element GPU writes) and then
        # async-copies the whole staging block into the captured GPU buffers.
        # Pinning enables a true non-blocking H2D and avoids a pageable-memory
        # fallback that would silently force a sync. See ``replay_paged_decode``.
        self._host_inputs: dict[int, dict[str, torch.Tensor]] = {}
        # Highest row index that was actually written into a bucket's host
        # scratch on the previous replay. The next replay only needs to wipe
        # rows ``[live_b : prev_live_b)`` back to safe init values; rows that
        # have never been live since capture are still at their init state.
        self._prev_live_b: dict[int, int] = {}
        self._meta: dict[int, PagedAttentionMetadata] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        # Reserve a single pool page that all dummy rows (and the warmup /
        # capture forwards themselves) read and write to, so the kernel's
        # in-place K/V writes never corrupt pages that real requests later
        # acquire. The page stays "allocated" for the engine's lifetime.
        if engine.pool is None:
            raise RuntimeError("CudaGraphRunner requires a paged engine pool")
        self._scratch_page: int = engine.pool.allocate(1)[0]

    @property
    def max_batch_size(self) -> int:
        return self.batch_sizes[-1]

    @property
    def is_captured(self) -> bool:
        return self._captured

    @property
    def scratch_page(self) -> int:
        return self._scratch_page

    def bucket_for(self, live_batch_size: int) -> int:
        for b in self.batch_sizes:
            if b >= live_batch_size:
                return b
        raise ValueError(
            f"live batch size {live_batch_size} exceeds max bucket "
            f"{self.max_batch_size}"
        )

    def covers(self, live_batch_size: int, max_pages: int, max_position: int) -> bool:
        return (
            live_batch_size <= self.max_batch_size
            and max_pages <= self.max_pages_per_seq
            and max_position < self.rope_cache_cap
        )

    def capture_all(self) -> None:
        if self._captured:
            return
        if not str(self.engine.device).startswith("cuda"):
            raise RuntimeError("CUDA graphs require a CUDA device")
        if self.engine.pool is None:
            raise RuntimeError("CUDA graph capture requires --mode paged")

        self._prepare_rotary_for_capture()
        torch.cuda.synchronize()
        for b in self.batch_sizes:
            logger.info(
                "Capturing CUDA graph  bucket=%d  max_pages_per_seq=%d",
                b,
                self.max_pages_per_seq,
            )
            self._capture_bucket(b)
        self._captured = True
        logger.info(
            "CUDA graphs captured for %d bucket(s): %s",
            len(self.batch_sizes),
            self.batch_sizes,
        )

    def _prepare_rotary_for_capture(self) -> None:
        # Default RotaryEmbedding.forward calls ``int(position_ids.max().item())``,
        # a device→host sync that's illegal inside a captured region. Pre-populate
        # the cos/sin cache to ``rope_cache_cap`` and replace forward with a
        # sync-free closure that only indexes into the cache.
        for module in self.engine.model.modules():
            if not isinstance(module, RotaryEmbedding):
                continue
            device = module.inv_freq.device
            seed = torch.tensor([[self.rope_cache_cap - 1]], device=device, dtype=torch.long)
            with torch.inference_mode():
                _ = module(seed)
            cos_cache = module._cos
            sin_cache = module._sin

            def static_forward(
                position_ids: torch.Tensor,
                _cos: torch.Tensor = cos_cache,
                _sin: torch.Tensor = sin_cache,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                cos = _cos[position_ids].unsqueeze(2)
                sin = _sin[position_ids].unsqueeze(2)
                return cos, sin

            module.forward = static_forward
        logger.info(
            "Patched RotaryEmbedding for capture  rope_cap=%d  (no host sync)",
            self.rope_cache_cap,
        )

    def _capture_bucket(self, b: int) -> None:
        engine = self.engine
        device = engine.device
        pool = engine.pool
        assert pool is not None

        input_ids = torch.zeros((b, 1), dtype=torch.long, device=device)
        position_ids = torch.zeros((b, 1), dtype=torch.long, device=device)
        block_table = torch.full(
            (b, self.max_pages_per_seq),
            self._scratch_page,
            dtype=torch.int32,
            device=device,
        )
        # cache_seqlens=1 ensures every captured forward attends to at least
        # one valid position even for unused dummy rows in the bucket.
        cache_seqlens = torch.ones((b,), dtype=torch.int32, device=device)

        meta = PagedAttentionMetadata(
            is_prefill=False,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
        )
        kv_caches = [
            (pool.k_cache(i), pool.v_cache(i)) for i in range(pool.num_layers)
        ]

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = engine.model(
                    input_ids,
                    position_ids,
                    kv_caches=kv_caches,
                    paged_metadata=meta,
                )
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            logits, _ = engine.model(
                input_ids,
                position_ids,
                kv_caches=kv_caches,
                paged_metadata=meta,
            )

        self._graphs[b] = graph
        self._inputs[b] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
        }
        # Persistent pinned host scratch initialised to the same safe values
        # used at capture time, so any row this bucket never touches (its
        # "dummy" rows) implicitly carries valid inputs for the replay
        # without per-step writes. ``replay_paged_decode`` only resets the
        # rows that were *previously* live and are now dummy.
        self._host_inputs[b] = {
            "input_ids": torch.zeros((b, 1), dtype=torch.long, pin_memory=True),
            "position_ids": torch.zeros((b, 1), dtype=torch.long, pin_memory=True),
            "cache_seqlens": torch.ones((b,), dtype=torch.int32, pin_memory=True),
            "block_table": torch.full(
                (b, self.max_pages_per_seq),
                self._scratch_page,
                dtype=torch.int32,
                pin_memory=True,
            ),
        }
        self._meta[b] = meta
        self._outputs[b] = logits
        self._prev_live_b[b] = 0

    @torch.inference_mode()
    def replay_paged_decode(
        self,
        input_ids_list: list[int],
        cache_lens_list: list[int],
        page_tables: list[list[int]],
    ) -> tuple[torch.Tensor, int]:
        """Stage live-batch inputs into pinned host scratch in bulk, async
        H2D into the captured bucket's stable buffers, replay.

        This is the single hot path for paged decode under cuda-graphs. The
        critical-path constraint on smaller GPUs (e.g. L4) is that the
        per-step input prep must not issue per-row H2D micro-syncs — that
        completely defeats the captured graph. Here every per-step write
        lands first on a pinned host tensor (CPU memcpy, no GPU sync), and
        only four batched ``copy_(non_blocking=True)`` calls cross the PCIe
        bus before ``graph.replay()``.

        Dummy rows ``[live_b : bucket)`` already carry safe values from
        capture (cache_seqlens=1, block_table=scratch_page, ids=0); rows
        that were live on a *previous* replay but are dummy on this one are
        wiped back to those values in ``[live_b : prev_live_b)`` only.

        Returns ``(logits, bucket)``. ``logits`` aliases graph-internal
        memory and is overwritten on the next replay of the same bucket;
        callers that need it past the next replay must copy it.
        """
        if not self._captured:
            raise RuntimeError(
                "CudaGraphRunner.replay_paged_decode called before capture_all()"
            )
        live_b = len(input_ids_list)
        if live_b == 0:
            raise ValueError("replay_paged_decode requires at least one request")
        b = self.bucket_for(live_b)
        host = self._host_inputs[b]
        dev = self._inputs[b]
        max_pages = self.max_pages_per_seq

        # ── Stage live rows on the pinned host scratch (CPU memcpys) ───
        # Build small CPU tensors via ``torch.tensor(list)`` (C-fast) and
        # slice-assign into the persistent pinned buffers in one shot per
        # field. No per-element loops touching GPU memory.
        ids_cpu = torch.tensor(input_ids_list, dtype=torch.long)
        lens_long_cpu = torch.tensor(cache_lens_list, dtype=torch.long)
        lens_int_cpu = lens_long_cpu.to(torch.int32)
        host["input_ids"][:live_b, 0] = ids_cpu
        host["position_ids"][:live_b, 0] = lens_long_cpu
        host["cache_seqlens"][:live_b] = lens_int_cpu

        # Block tables vary in length per request; pad to ``max_pages`` on
        # CPU then bulk-assign. flash-attn only reads up to
        # ``ceil(cache_seqlens[b]/page_size)`` entries per row, so padding
        # with 0 is safe for live rows just like in the no-graph path.
        padded = [pt + [0] * (max_pages - len(pt)) for pt in page_tables]
        bt_cpu = torch.tensor(padded, dtype=torch.int32)
        host["block_table"][:live_b] = bt_cpu

        # ── Restore dummy values only for rows that were live last time
        # but are dummy this time (incremental reset, not full sweep) ───
        prev_live_b = self._prev_live_b.get(b, 0)
        if live_b < prev_live_b:
            host["input_ids"][live_b:prev_live_b].zero_()
            host["position_ids"][live_b:prev_live_b].zero_()
            host["cache_seqlens"][live_b:prev_live_b].fill_(1)
            host["block_table"][live_b:prev_live_b].fill_(self._scratch_page)
        self._prev_live_b[b] = live_b

        # ── Four batched async H2D copies (pinned source ⇒ truly async) ─
        dev["input_ids"].copy_(host["input_ids"], non_blocking=True)
        dev["position_ids"].copy_(host["position_ids"], non_blocking=True)
        dev["cache_seqlens"].copy_(host["cache_seqlens"], non_blocking=True)
        dev["block_table"].copy_(host["block_table"], non_blocking=True)

        self._graphs[b].replay()
        return self._outputs[b], b
