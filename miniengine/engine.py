"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Two decode paths (M1):
  - decode_step(req)        : one request, used by baseline scheduler
  - batched_decode(reqs)    : many requests, one forward pass with padded
                              KV + attention mask, used by batched mode

Paged path (M2 Part A):
  - When `mode == "paged"`, prefill/batched_decode read and write KV
    through a pre-allocated `KVMemoryPool`. Each request stores a
    page_table (list of physical page indices) on `req.kv_cache`. On
    each step we gather pages into a contiguous tensor, run the M1
    attention path, and scatter only the new last-token K/V back.
    Real PagedAttention kernels replace the gather/scatter in Part B.

Prefill stays per-request — variable prompt lengths make batched prefill
complex, and decode is where the throughput gain lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 32,
        mem_fraction_static: float = 0.85,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.mem_fraction_static = mem_fraction_static
        self.pool: KVMemoryPool | None = None

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        with torch.device("meta"):
            self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()
        self.config = config

        if mode == "paged":
            self.pool = self._build_pool(config)
            logger.info(
                "KV pool ready  —  pages=%d, page_size=%d, "
                "total_kv_tokens=%d, bytes=%.2f GB",
                self.pool.num_pages,
                self.pool.page_size,
                self.pool.total_kv_tokens,
                self._pool_bytes() / (1024**3),
            )

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # ── KV pool sizing ──────────────────────────────────────────────────

    def _model_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def _pool_bytes(self) -> int:
        if self.pool is None:
            return 0
        return (
            2
            * self.pool.num_layers
            * self.pool.num_pages
            * self.pool.page_size
            * self.pool.num_kv_heads
            * self.pool.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )

    def _build_pool(self, config: ModelConfig) -> KVMemoryPool:
        if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            raise RuntimeError("paged mode currently requires a CUDA device")
        total_bytes = torch.cuda.get_device_properties(self.device).total_memory
        model_bytes = self._model_bytes()
        static_budget = int(total_bytes * self.mem_fraction_static)
        kv_budget = static_budget - model_bytes
        if kv_budget <= 0:
            raise RuntimeError(
                f"--mem-fraction-static={self.mem_fraction_static} leaves "
                f"{kv_budget} bytes for KV after a {model_bytes/1024**3:.2f} GB "
                f"model on a {total_bytes/1024**3:.2f} GB device. Increase the "
                f"fraction or use a smaller model."
            )
        logger.info(
            "KV pool budget  —  total=%.2f GB, model=%.2f GB, "
            "static_frac=%.2f → kv_budget=%.2f GB",
            total_bytes / 1024**3,
            model_bytes / 1024**3,
            self.mem_fraction_static,
            kv_budget / 1024**3,
        )
        return KVMemoryPool.from_budget(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            bytes_budget=kv_budget,
        )

    # ── Forward passes ──────────────────────────────────────────────────

    def prefill(self, request: Request) -> int:
        if self.mode == "paged":
            return self._prefill_paged(request)
        return self._prefill_classic(request)

    @torch.inference_mode()
    def _prefill_classic(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def _prefill_paged(self, request: Request) -> int:
        """Paged prefill: write the full prompt KV directly into the pool."""
        assert self.pool is not None, "paged prefill requires a KV pool"
        page_table = request.kv_cache
        assert isinstance(page_table, list), (
            "paged prefill expects req.kv_cache to be a page_table allocated "
            "by the scheduler before prefill is called"
        )

        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        self._scatter_kv_into_pages(page_table, start_pos=0, new_kv=kv_caches)

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(
            input_ids, position_ids, kv_caches=request.kv_cache
        )
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Batched decode ──────────────────────────────────────────────────

    def batched_decode(self, requests: list[Request]) -> list[int]:
        if self.mode == "paged":
            return self._batched_decode_paged(requests)
        return self._batched_decode_classic(requests)

    @torch.inference_mode()
    def _batched_decode_classic(self, requests: list[Request]) -> list[int]:
        """
        Decode one token for each request in a single forward pass.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    # ── Paged decode ────────────────────────────────────────────────────

    def _paged_kv_length(self, req: Request) -> int:
        """How many KV positions are already stored in pages for req.

        After prefill, this equals num_input_tokens (the prompt is in
        pages, the first sampled output token is *not* yet in pages —
        its K/V will be written on the next decode step).
        """
        return req.num_input_tokens + max(0, req.num_output_tokens - 1)

    @torch.inference_mode()
    def _batched_decode_paged(self, requests: list[Request]) -> list[int]:
        """Paged batched decode.

        Each request's KV lives in pool pages addressed by req.kv_cache
        (a list[int] page table). We gather pages into a padded
        contiguous KV per layer, run the M1 batched SDPA path, then
        scatter only the new last-token K/V back into the next page
        slot of every request. Part B replaces gather/scatter with a
        real paged-attention kernel.
        """
        if not requests:
            return []
        assert self.pool is not None, "paged decode requires a KV pool"

        batch_size = len(requests)
        num_layers = self.pool.num_layers
        page_size = self.pool.page_size

        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        page_tables: list[list[int]] = [req.kv_cache for req in requests]
        cache_lens = [self._paged_kv_length(req) for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens], dtype=torch.long, device=self.device
        )

        # Build (page_idx, slot) grids covering positions [0, max_cache_len)
        # for every request. Positions ≥ cache_len[b] are dummies — they
        # will be masked out by attention_mask. Doing this on CPU first
        # avoids a tensor op per (request, position).
        page_idx_rows: list[list[int]] = []
        for b in range(batch_size):
            pt = page_tables[b]
            cl = cache_lens[b]
            row: list[int] = []
            cur_page = pt[0]
            last_block = -1
            for t in range(max_cache_len):
                if t < cl:
                    block = t // page_size
                    if block != last_block:
                        cur_page = pt[block]
                        last_block = block
                    row.append(cur_page)
                else:
                    row.append(0)
            page_idx_rows.append(row)
        page_idx_grid = torch.tensor(
            page_idx_rows, device=self.device, dtype=torch.long
        )
        slot_grid = (
            torch.arange(max_cache_len, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            % page_size
        )

        # Gather: (batch, max_cache_len, kv_heads, head_dim) → permute to
        # (batch, kv_heads, max_cache_len, head_dim) matching the M1 layout.
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_pool = self.pool.k_cache(layer_idx)
            v_pool = self.pool.v_cache(layer_idx)
            k_gathered = k_pool[page_idx_grid, slot_grid].permute(0, 2, 1, 3).contiguous()
            v_gathered = v_pool[page_idx_grid, slot_grid].permute(0, 2, 1, 3).contiguous()
            padded_kv_caches.append((k_gathered, v_gathered))

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Scatter the new last-token K/V back into each request's next slot.
        # Pre-compute (page_idx, slot) per request once, then write all
        # layers in vectorized fashion.
        next_page_idx = torch.tensor(
            [page_tables[i][cache_lens[i] // page_size] for i in range(batch_size)],
            device=self.device,
            dtype=torch.long,
        )
        next_slot = torch.tensor(
            [cache_lens[i] % page_size for i in range(batch_size)],
            device=self.device,
            dtype=torch.long,
        )
        for layer_idx in range(num_layers):
            # new_kv_caches[layer][0]: (batch, kv_heads, max_cache_len+1, head_dim)
            k_new = new_kv_caches[layer_idx][0][:, :, -1, :].contiguous()
            v_new = new_kv_caches[layer_idx][1][:, :, -1, :].contiguous()
            self.pool.k_cache(layer_idx)[next_page_idx, next_slot] = k_new
            self.pool.v_cache(layer_idx)[next_page_idx, next_slot] = v_new

        # Sample
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    # ── Page scatter helper (prefill) ───────────────────────────────────

    def _scatter_kv_into_pages(
        self,
        page_table: list[int],
        start_pos: int,
        new_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Write a chunk of newly computed K/V into the pool.

        new_kv[layer] = (k, v) with shape (1, kv_heads, n_new, head_dim).
        The chunk maps onto positions [start_pos, start_pos + n_new) in
        the request's logical KV stream; each is routed to
        page_table[pos // page_size] at slot pos % page_size.
        """
        assert self.pool is not None
        n_new = new_kv[0][0].shape[2]
        if n_new == 0:
            return
        ps = self.pool.page_size
        positions = torch.arange(
            start_pos, start_pos + n_new, device=self.device, dtype=torch.long
        )
        slot = positions % ps
        page_table_t = torch.tensor(
            page_table, device=self.device, dtype=torch.long
        )
        page_idx = page_table_t[positions // ps]

        for layer_idx, (k, v) in enumerate(new_kv):
            k_flat = k.squeeze(0).transpose(0, 1).contiguous()
            v_flat = v.squeeze(0).transpose(0, 1).contiguous()
            self.pool.k_cache(layer_idx)[page_idx, slot] = k_flat
            self.pool.v_cache(layer_idx)[page_idx, slot] = v_flat
