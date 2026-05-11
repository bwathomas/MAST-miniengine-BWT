"""FlashInfer-backed paged decode kernel (Milestone 2 improvement item 7).

Replaces the per-layer ``flash_attn_with_kvcache`` call in the paged decode
forward path with FlashInfer's ``BatchDecodeWithPagedKVCacheWrapper``
(FlashInfer §3.2.2 + Appendix A — multi-tile-size FA microkernels with
GQA head-group fusion + Split-K writethrough).

Architecture:

* The wrapper plans **once per decode step** (CPU-side; sets up indptr,
  page-id lists, last-page lengths). The plan call is intentionally
  hoisted out of the per-layer loop in :func:`Engine._batched_decode_paged`
  to avoid 36× re-planning per step.
* Per layer, we explicitly write the freshly computed K/V into the pool
  via :func:`flashinfer.append_paged_kv_cache` (flash-attn's
  ``flash_attn_with_kvcache`` does this implicitly; FlashInfer separates
  the write from the read).
* Then we call ``wrapper.run(q, (k_pool, v_pool))`` to compute attention.

L4 expectations:

* On bandwidth-bound paged decode, FlashInfer's tighter tile selection
  and head-group fusion are projected to win 10-25 % over flash-attn,
  but the gap shrinks as GPU shrinks (FlashInfer §6.1 shows the largest
  delta on H100 and a smaller delta on A100). On L4 a small win is
  plausible; a regression isn't impossible.
* FlashInfer's plan call requires a CPU→GPU stream sync (it has to copy
  the indptr/indices to device buffers in the workspace). That's one
  sync per step. Our existing flash-attn path is sync-free post the
  sync-free RoPE fix in batch 1, so FlashInfer reintroduces one sync per
  step — likely the dominant downside on L4's slow PCIe.
* Mutually exclusive with ``--cuda-graph`` (the manual graph capture
  expects flash-attn's in-kernel writeback). Engine init enforces this.

Layout assumption: our KV pool stores ``k_caches[layer]`` and
``v_caches[layer]`` as separate ``(num_pages, page_size, num_kv_heads,
head_dim)`` tensors. That matches FlashInfer's ``"NHD"`` layout for the
``BatchDecodeWithPagedKVCacheWrapper`` constructor and the
``append_paged_kv_cache`` API. If you bump FlashInfer to a major version
that reshuffles the layout, this is the one place that needs to know.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


def _import_flashinfer() -> Any:
    """Lazy-import flashinfer with an actionable error.

    The wheel ships ABI-tagged builds for specific torch + cuda combos;
    install with::

        pip install flashinfer-python

    on a Colab L4 runtime that already has torch>=2.3 + CUDA 12.x. The
    error here is the user's signal to install the package; nothing in
    the default flag set imports flashinfer.
    """
    try:
        import flashinfer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "--use-flashinfer was passed but the 'flashinfer' package is not "
            "installed. Install with 'pip install flashinfer-python' (Colab L4 "
            "wheels exist for torch>=2.3 + CUDA 12). Original error: " + str(exc)
        ) from exc
    return flashinfer


class FlashInferDecoder:
    """Owns one FlashInfer workspace + decode wrapper for the engine."""

    DEFAULT_WORKSPACE_MB = 64

    def __init__(
        self,
        pool: KVMemoryPool,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        workspace_mb: int = DEFAULT_WORKSPACE_MB,
    ) -> None:
        self.pool = pool
        self.num_qo_heads = int(num_qo_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.dtype = dtype
        self.device = device

        flashinfer = _import_flashinfer()
        self._flashinfer = flashinfer

        self.workspace_bytes: int = int(workspace_mb) * 1024 * 1024
        self._workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=device
        )

        wrapper_cls = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
        self._wrapper = wrapper_cls(self._workspace, "NHD")

        # Per-step state set by ``plan``; consumed by per-layer calls.
        self._planned: bool = False
        self._batch_indices: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._kv_indptr_dev: torch.Tensor | None = None
        self._kv_indices_dev: torch.Tensor | None = None
        self._kv_last_page_dev: torch.Tensor | None = None

    @torch.inference_mode()
    def plan(
        self, page_tables: list[list[int]], cache_lens: list[int]
    ) -> None:
        """Set up the wrapper for the current decode batch.

        Builds the four CPU int32 arrays FlashInfer wants:
          * ``kv_indptr[0..B] = cumulative # pages per request``
          * ``kv_indices``    = flat concatenation of per-request page IDs
          * ``kv_last_page_len`` = per-request tokens-in-last-page (1..page_size)
          * (for ``append_paged_kv_cache``) ``batch_indices`` + ``positions``
            map each newly produced token to (row, position-in-cache).

        ``cache_lens[b]`` is the # KV positions currently in the cache;
        the new token's position is ``cache_lens[b]``.
        """
        B = len(page_tables)
        assert B > 0
        kv_indptr_list = [0]
        kv_indices_list: list[int] = []
        last_page_len_list: list[int] = []
        for pt, cl in zip(page_tables, cache_lens):
            # Number of pages this row actually uses post-append. A
            # request whose current cache len is exactly a multiple of
            # page_size will spill the new token into the next page;
            # FlashInfer needs that page in the indices list.
            new_len = cl + 1
            n_pages_used = (new_len + self.page_size - 1) // self.page_size
            assert n_pages_used <= len(pt), (
                f"FlashInfer.plan: row needs {n_pages_used} pages but "
                f"page_table only has {len(pt)}"
            )
            kv_indices_list.extend(pt[:n_pages_used])
            kv_indptr_list.append(kv_indptr_list[-1] + n_pages_used)
            last_page = new_len - (n_pages_used - 1) * self.page_size
            last_page_len_list.append(last_page)

        device = self.device
        kv_indptr_dev = torch.tensor(
            kv_indptr_list, dtype=torch.int32, device=device
        )
        kv_indices_dev = torch.tensor(
            kv_indices_list, dtype=torch.int32, device=device
        )
        last_page_dev = torch.tensor(
            last_page_len_list, dtype=torch.int32, device=device
        )
        batch_indices_dev = torch.arange(B, dtype=torch.int32, device=device)
        positions_dev = torch.tensor(
            cache_lens, dtype=torch.int32, device=device
        )

        self._wrapper.plan(
            kv_indptr_dev,
            kv_indices_dev,
            last_page_dev,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
        )

        self._batch_indices = batch_indices_dev
        self._positions = positions_dev
        self._kv_indptr_dev = kv_indptr_dev
        self._kv_indices_dev = kv_indices_dev
        self._kv_last_page_dev = last_page_dev
        self._planned = True

    @torch.inference_mode()
    def attend_and_append(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
    ) -> torch.Tensor:
        """Write k_new/v_new into pool, then run paged decode attention.

        Args:
            q     : (B, num_qo_heads, head_dim)         — freshly computed
            k_new : (B, num_kv_heads, head_dim)         — to append
            v_new : (B, num_kv_heads, head_dim)
            k_pool: (num_pages, page_size, num_kv_heads, head_dim)
            v_pool: (num_pages, page_size, num_kv_heads, head_dim)

        Returns:
            attention output (B, num_qo_heads, head_dim)
        """
        assert self._planned, "FlashInferDecoder.attend_and_append before plan()"
        self._flashinfer.append_paged_kv_cache(
            k_new,
            v_new,
            self._batch_indices,
            self._positions,
            (k_pool, v_pool),
            self._kv_indices_dev,
            self._kv_indptr_dev,
            self._kv_last_page_dev,
            kv_layout="NHD",
        )
        return self._wrapper.run(q, (k_pool, v_pool))
