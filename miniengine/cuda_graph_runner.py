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
        self._meta[b] = meta
        self._outputs[b] = logits

    @torch.inference_mode()
    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Copy fresh inputs into the captured bucket's stable buffers and
        replay. All input tensors must already be shaped for the bucket
        (leading dim == ``self.bucket_for(live_B)``); dummy rows must carry
        safe values (the engine fills them with cache_seqlens=1 + a valid
        block_table[0] entry).

        Returns the (logits, bucket) from the graph's own output buffer —
        this aliases graph-internal memory and is overwritten on the next
        replay of the same bucket.
        """
        if not self._captured:
            raise RuntimeError("CudaGraphRunner.replay called before capture_all()")
        b = input_ids.shape[0]
        if b not in self._graphs:
            raise ValueError(
                f"no captured graph for bucket batch size {b}; "
                f"captured buckets are {self.batch_sizes}"
            )
        bufs = self._inputs[b]
        bufs["input_ids"].copy_(input_ids)
        bufs["position_ids"].copy_(position_ids)
        bufs["block_table"].copy_(block_table)
        bufs["cache_seqlens"].copy_(cache_seqlens)

        self._graphs[b].replay()
        return self._outputs[b], b
