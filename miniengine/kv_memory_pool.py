"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout: per layer, K and V are kept as separate tensors of
shape ``(num_pages, page_size, num_kv_heads, head_dim)``. This matches
the layout that ``flash_attn_with_kvcache`` (Part B) expects, so
swapping in a real paged-attention kernel later only requires changing
the indexing path — not reshuffling memory. The free list is a simple
``deque[int]`` of physical page indices (FIFO acquire / FIFO release).
"""

from __future__ import annotations

from collections import deque

import torch


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).
        page_size:    Tokens per page. Tunable knob — exposed as
                      `--page-size` on the CLI. Smaller = less
                      fragmentation, bigger page tables; larger = the
                      opposite.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. "cuda".
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self._k_caches: list[torch.Tensor] = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self._v_caches: list[torch.Tensor] = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

        self._free: deque[int] = deque(range(num_pages))
        # Per-page refcount (PagedAttention §4.4). 0 = free, 1 = owned by
        # one request, >1 = shared (e.g. a prefix-cache hit aliasing the
        # same physical page to multiple sequences). ``free()`` decrements;
        # a page returns to the free list only when it hits 0. With no
        # explicit acquire() calls every allocation behaves exactly like
        # the prior unique-owner pool, so existing call sites are unchanged.
        self._refcount: list[int] = [0] * num_pages

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Each returned page has refcount=1. Raises ``RuntimeError`` if the
        free list is too short to satisfy the request — callers (the
        scheduler) are expected to check :pyattr:`num_free` first when
        they want a deterministic admission decision rather than an
        exception.
        """
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}")
        if num_pages > len(self._free):
            raise RuntimeError(
                f"KV pool exhausted: requested {num_pages} pages, "
                f"only {len(self._free)} free of {self.num_pages}"
            )
        out: list[int] = []
        for _ in range(num_pages):
            p = self._free.popleft()
            self._refcount[p] = 1
            out.append(p)
        return out

    def acquire(self, page_indices: list[int]) -> None:
        """Increment refcount on already-allocated pages.

        Intended for a prefix-cache layer that wants to share a physical
        page across multiple requests (PagedAttention §4.4). Asserts each
        page is currently held by at least one owner; aliasing a free page
        would race with a future :meth:`allocate`.
        """
        for p in page_indices:
            if self._refcount[p] <= 0:
                raise RuntimeError(
                    f"acquire() on free page {p} (refcount=0); pages must "
                    f"be allocated before they can be shared"
                )
            self._refcount[p] += 1

    def free(self, page_indices: list[int]) -> None:
        """Decrement refcount on the listed pages; release at refcount=0."""
        for p in page_indices:
            rc = self._refcount[p]
            if rc <= 0:
                raise RuntimeError(
                    f"free() on already-free page {p} (refcount={rc})"
                )
            rc -= 1
            self._refcount[p] = rc
            if rc == 0:
                self._free.append(p)

    def refcount(self, page_idx: int) -> int:
        """How many requests currently hold this page (0 if free)."""
        return self._refcount[page_idx]

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        if seq_len <= 0:
            return 0
        return (seq_len + self.page_size - 1) // self.page_size

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_pages - len(self._free)

    @property
    def total_kv_tokens(self) -> int:
        """Total tokens of KV the pool can hold across all pages."""
        return self.num_pages * self.page_size

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        Each pair has shape ``(num_pages, page_size, num_kv_heads,
        head_dim)`` and is stable across the entire run: the attention
        path holds references to these and indexes into them via
        per-request page tables.
        """
        return list(zip(self._k_caches, self._v_caches))

    def k_cache(self, layer_idx: int) -> torch.Tensor:
        return self._k_caches[layer_idx]

    def v_cache(self, layer_idx: int) -> torch.Tensor:
        return self._v_caches[layer_idx]

    def slot_mapping_for_prefill(
        self,
        page_table: list[int],
        start_pos: int,
        length: int,
    ) -> list[int]:
        """Per-token flat indices for writing K/V into the pool.

        Returns ``[page_table[t // page_size] * page_size + (t % page_size)
        for t in [start_pos, start_pos + length)]`` — the flat-index form
        the paged-attention prefill kernel uses to scatter freshly computed
        K/V into the request's pages (PagedAttention §4.3, Fig. 6).
        """
        if length <= 0:
            return []
        ps = self.page_size
        return [
            page_table[t // ps] * ps + (t % ps)
            for t in range(start_pos, start_pos + length)
        ]

    @staticmethod
    def pad_block_table(
        page_tables: list[list[int]], max_pages: int
    ) -> list[list[int]]:
        """Pad per-request page tables out to a common length.

        The decode kernel takes a dense ``(B, max_pages_per_seq)`` block
        table; rows for shorter requests are padded with ``0``s. Padding
        entries are never read because the kernel's row scan is upper-
        bounded by ``cache_seqlens[b] + 1``.
        """
        return [pt + [0] * (max_pages - len(pt)) for pt in page_tables]

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> KVMemoryPool:
        """Convenience: derive `num_pages` from a memory budget."""
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        bytes_per_page = (
            2  # K + V
            * num_layers
            * page_size
            * num_kv_heads
            * head_dim
            * elem_bytes
        )
        num_pages = max(1, int(bytes_budget) // bytes_per_page)
        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
