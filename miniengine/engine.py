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

Paged path (M2 Part B):
  - When `mode == "paged"`, prefill is **packed batched** and decode is
    **paged batched**. Both paths read/write KV directly through a
    pre-allocated `KVMemoryPool` via the flash-attn paged kernels — no
    gather/scatter, no padding. Each request stores a page_table
    (list of physical page indices) on `req.kv_cache`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.cuda_graph_runner import CudaGraphRunner
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, PagedAttentionMetadata, load_weights
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
        torch_compile: bool = False,
        cuda_graph: bool = False,
        cuda_graph_batch_sizes: list[int] | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.mem_fraction_static = mem_fraction_static
        self.torch_compile = torch_compile
        self.cuda_graph = cuda_graph
        self.pool: KVMemoryPool | None = None
        self.cuda_graph_runner: CudaGraphRunner | None = None

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

        if torch_compile:
            self._compile_mlp_modules()

        if cuda_graph:
            if mode != "paged":
                raise RuntimeError("--cuda-graph requires --mode paged")
            sizes = cuda_graph_batch_sizes or [1, 2, 4, 8, 16, 32]
            self.cuda_graph_runner = CudaGraphRunner(self, sizes)
            self.cuda_graph_runner.capture_all()

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

    # ── torch.compile of MLP sub-region (M2 Part C) ─────────────────────

    def _compile_mlp_modules(self) -> None:
        """Wrap each transformer block's MLP with ``torch.compile``.

        ``mode='default'`` with ``dynamic=True`` is chosen over
        ``'reduce-overhead'`` so the compiled artifacts contain no inner
        CUDA graphs of their own — that lets us stack cleanly with the
        outer manual capture in ``CudaGraphRunner`` (CUDA does not allow
        nested graph capture). The MLP is the most stable sub-region in
        the decode path: shape ``(B, 1, hidden_size)`` and no Python
        branching, so dynamo specializes once and Inductor fuses the
        SwiGLU ``silu(gate(x)) * up(x) → down`` chain into one kernel.
        """
        layers = self.model.model.layers
        for layer in layers:
            layer.mlp = torch.compile(layer.mlp, mode="default", dynamic=True)
        logger.info(
            "torch.compile enabled on MLP of %d layers (mode=default, dynamic=True)",
            len(layers),
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
            return self.batched_prefill([request])[0]
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

    def batched_prefill(self, requests: list[Request]) -> list[int]:
        """Packed batched prefill (paged mode only).

        Flattens N prompts into one packed sequence of length ``T = ΣL_i``,
        runs a single ``flash_attn_varlen_func`` forward pass, scatters the
        per-token K/V into each request's pages via a slot mapping, and
        samples the first output token of every request from the last
        position of each prompt's logits.
        """
        if not requests:
            return []
        if self.mode != "paged":
            raise RuntimeError("batched_prefill is only available in paged mode")
        return self._batched_prefill_paged(requests)

    @torch.inference_mode()
    def _batched_prefill_paged(self, requests: list[Request]) -> list[int]:
        assert self.pool is not None, "paged prefill requires a KV pool"

        prompt_lens = [req.num_input_tokens for req in requests]
        max_seqlen = max(prompt_lens)

        packed_ids: list[int] = []
        position_chunks: list[int] = []
        slot_mapping_list: list[int] = []
        cu = [0]
        for req, L in zip(requests, prompt_lens):
            page_table = req.kv_cache
            assert isinstance(page_table, list), (
                "paged prefill expects req.kv_cache to be a page_table "
                "allocated by the scheduler before prefill is called"
            )
            packed_ids.extend(req.input_ids)
            position_chunks.extend(range(L))
            slot_mapping_list.extend(
                self.pool.slot_mapping_for_prefill(page_table, 0, L)
            )
            cu.append(cu[-1] + L)

        input_ids = torch.tensor(
            packed_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        position_ids = torch.tensor(
            position_chunks, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(
            slot_mapping_list, dtype=torch.long, device=self.device
        )
        last_token_indices = cu_seqlens[1:].to(torch.long) - 1

        meta = PagedAttentionMetadata(
            is_prefill=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            slot_mapping=slot_mapping,
        )
        kv_caches = [
            (self.pool.k_cache(i), self.pool.v_cache(i))
            for i in range(self.pool.num_layers)
        ]
        logits, _ = self.model(
            input_ids,
            position_ids,
            kv_caches=kv_caches,
            paged_metadata=meta,
            last_token_indices=last_token_indices,
        )

        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_ids.append(
                sample_token(
                    logits[:, i, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

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
        """Paged batched decode via ``flash_attn_with_kvcache``.

        Stacks one new token per request into ``(B, 1)``, builds a dense
        ``block_table`` and ``cache_seqlens`` from the per-request page
        tables, and runs a single batched forward. The kernel itself
        appends each request's new K/V at slot ``cache_seqlens[b]`` of
        the request's pages and gathers the full prefix for attention.

        When a ``CudaGraphRunner`` is available and the live batch fits a
        captured bucket, the model forward is replayed from a captured
        graph instead of issued op-by-op.
        """
        if not requests:
            return []
        assert self.pool is not None, "paged decode requires a KV pool"

        batch_size = len(requests)
        page_tables: list[list[int]] = [req.kv_cache for req in requests]
        cache_lens = [self._paged_kv_length(req) for req in requests]
        max_pages = max(len(pt) for pt in page_tables)
        max_position = max(cache_lens)

        runner = self.cuda_graph_runner
        use_graph = runner is not None and runner.covers(
            batch_size, max_pages, max_position
        )

        if use_graph:
            assert runner is not None
            bucket = runner.bucket_for(batch_size)
            scratch = runner.scratch_page
            input_ids = torch.zeros((bucket, 1), dtype=torch.long, device=self.device)
            position_ids = torch.zeros((bucket, 1), dtype=torch.long, device=self.device)
            cache_seqlens = torch.ones((bucket,), dtype=torch.int32, device=self.device)
            block_table = torch.full(
                (bucket, runner.max_pages_per_seq),
                scratch,
                dtype=torch.int32,
                device=self.device,
            )
            for i, req in enumerate(requests):
                input_ids[i, 0] = req.output_ids[-1]
                position_ids[i, 0] = cache_lens[i]
                cache_seqlens[i] = cache_lens[i]
                pt = page_tables[i]
                block_table[i, : len(pt)] = torch.as_tensor(
                    pt, dtype=torch.int32, device=self.device
                )
            logits, _ = runner.replay(
                input_ids, position_ids, block_table, cache_seqlens
            )
        else:
            input_ids = torch.tensor(
                [[req.output_ids[-1]] for req in requests],
                dtype=torch.long,
                device=self.device,
            )
            position_ids = torch.tensor(
                [[cl] for cl in cache_lens], dtype=torch.long, device=self.device
            )
            cache_seqlens = torch.tensor(
                cache_lens, dtype=torch.int32, device=self.device
            )
            block_table = torch.tensor(
                self.pool.pad_block_table(page_tables, max_pages),
                dtype=torch.int32,
                device=self.device,
            )
            meta = PagedAttentionMetadata(
                is_prefill=False,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
            )
            kv_caches = [
                (self.pool.k_cache(i), self.pool.v_cache(i))
                for i in range(self.pool.num_layers)
            ]
            logits, _ = self.model(
                input_ids,
                position_ids,
                kv_caches=kv_caches,
                paged_metadata=meta,
            )

        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids
