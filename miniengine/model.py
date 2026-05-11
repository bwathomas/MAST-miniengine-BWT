"""
Bare-bone Qwen3 transformer in pure PyTorch.

No HuggingFace model classes — just nn.Module, nn.Linear, and manual
attention with KV cache.  Weight names match the HuggingFace checkpoint
so we can load safetensors directly via load_state_dict().

Architecture (Qwen3-4B as reference):
    Embedding(151936, 2560)
    36 x TransformerBlock:
        RMSNorm → Attention(GQA + QK-Norm + RoPE) → RMSNorm → SwiGLU MLP
    RMSNorm
    LM Head (tied with embedding)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_FLASH_ATTN_INSTALL_HINT = (
    "flash-attn is required for --mode paged but is not installed. Install a "
    "prebuilt wheel matching your torch/CUDA/Python from "
    "https://github.com/Dao-AILab/flash-attention/releases (recommended), or "
    "build from source with `pip install -e \".[paged]\"` (slow). See the "
    "Colab notebook's `(Optional) Install flash-attn` cell for an auto-"
    "detected wheel URL."
)


def _import_flash_attn():
    """Lazy-import flash_attn with an actionable error if it's missing."""
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(_FLASH_ATTN_INSTALL_HINT) from e
    return flash_attn_varlen_func, flash_attn_with_kvcache


# ── Paged attention metadata ────────────────────────────────────────────


@dataclass
class PagedAttentionMetadata:
    """Per-step metadata for the paged attention kernels.

    `is_prefill=True` selects ``flash_attn_varlen_func`` and consumes
    `cu_seqlens`, `max_seqlen`, `slot_mapping`. `is_prefill=False` selects
    ``flash_attn_with_kvcache`` and consumes `block_table`, `cache_seqlens`.
    The kernels read/write the per-layer pool tensors that the engine passes
    in as `kv_caches[i] = (pool.k_cache(i), pool.v_cache(i))`.

    ``num_splits`` is the Split-KV depth forwarded to ``flash_attn_with_kvcache``
    (FlashAttention §4 IO-aware analogue; flash-attn 2.8 ``num_splits`` kwarg).
    0 = let the kernel's heuristic choose. L4 has only 58 SMs vs A100's 108,
    so the auto pick can leave occupancy on the table for our paged decode.
    """

    is_prefill: bool
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int = 0
    slot_mapping: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    num_splits: int = 0
    # When set, the decode path skips ``flash_attn_with_kvcache`` and calls
    # the supplied FlashInferDecoder. Always None on the prefill metadata.
    flashinfer: object | None = None


# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Model architecture config, loaded from HuggingFace config.json."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128  # explicit, NOT hidden_size // num_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> ModelConfig:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=getattr(hf, "rope_theta", 10000.0),
            max_position_embeddings=getattr(hf, "max_position_embeddings", 4096),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )


# ── Building blocks ─────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance in fp32 — bf16 mean-of-squares loses too much precision.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes and caches cos/sin tables, indexed by position_ids at
    forward time. The cache grows on-demand so we never allocate for the
    full 256K context upfront.

    When ``_static_cap`` is set (via :meth:`prepopulate`), the forward path
    is sync-free: it skips ``int(position_ids.max().item())`` and indexes
    straight into the cache. This mirrors the ``CudaGraphRunner`` patch but
    applies it to ALL paged paths (including ``paged`` and ``paged+compile``),
    so the host doesn't drain the launch queue on every decode step. Motivated
    by FlashInfer §D.1 "no host-sync inside the captured region" — the same
    rule shortens non-graph forwards too.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0
        self._static_cap: int = 0

    @torch.no_grad()
    def prepopulate(self, length: int) -> None:
        """Grow the cos/sin cache to ``length`` and lock further growth.

        Once called, :meth:`forward` no longer probes ``position_ids.max()``
        — callers must guarantee positions stay below ``length``. The engine
        sets this from ``ModelConfig.max_position_embeddings`` clipped to a
        sane inference cap.
        """
        if length <= self._cached_len:
            self._static_cap = max(self._static_cap, length)
            return
        self._grow_cache(length)
        self._static_cap = length

    @torch.no_grad()
    def _grow_cache(self, length: int) -> None:
        t = torch.arange(
            length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._cached_len = length

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: (batch, seq_len) integer positions.

        Returns:
            cos, sin each of shape (batch, 1, seq_len, head_dim) —
            broadcastable over the head dimension.
        """
        if self._static_cap == 0:
            max_pos = int(position_ids.max().item()) + 1
            if self._cos is None or max_pos > self._cached_len:
                length = max(max_pos, self._cached_len * 2, 256)
                self._grow_cache(length)
        elif self._cos is None:
            self._grow_cache(self._static_cap)

        cos = self._cos[position_ids].unsqueeze(2)
        sin = self._sin[position_ids].unsqueeze(2)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE to x.

    x:   (batch, num_heads, seq_len, head_dim)
    cos: (batch, seq_len, 1, head_dim)  — broadcast over heads
    sin: same shape as cos
    """
    # Cast cos/sin to x.dtype — fp32 cos/sin would silently promote q/k.
    cos = cos.transpose(1, 2).to(x.dtype)
    sin = sin.transpose(1, 2).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


# ── Attention ───────────────────────────────────────────────────────────


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA), QK-Norm,
    and Rotary Position Embeddings.

    Q projects to  num_attention_heads  × head_dim
    K projects to  num_key_value_heads  × head_dim
    V projects to  num_key_value_heads  × head_dim
    O projects back to hidden_size
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        # Qwen3: RMSNorm on Q and K after projection (per-head)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _qkv_with_rope(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V projection + per-head reshape + QK-Norm + RoPE.

        Pure compute, no opaque kernels — a clean ``torch.compile`` target.
        Returns q, k, v shaped ``(B, H_q/H_kv, T, D)`` ready for the
        attention kernel.
        """
        bsz, seq_len, _ = hidden.shape
        q = (
            self.q_proj(hidden)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k, v

    def _attn_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_metadata: PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run the attention kernel (flash-attn paged, or SDPA fallback).

        Kept **outside** any ``torch.compile`` region: the flash-attn calls
        are opaque CUDA extensions that dynamo cannot trace through, and
        the SDPA path has Python branching on ``kv_cache`` / mask presence
        that would force recompiles. Returns the flat ``(B, T, H_q*D)``
        attention output (pre o_proj).
        """
        bsz, _, seq_len, _ = q.shape
        if paged_metadata is not None:
            return self._paged_attention(
                q, k, v, bsz, seq_len, kv_cache, paged_metadata
            )

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(bsz, self.num_heads, -1, self.head_dim)

        if attention_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            is_causal = kv_cache is None and seq_len > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return out, new_kv

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_metadata: PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            hidden:         (batch, seq_len, hidden_size)
            cos, sin:       from RotaryEmbedding, broadcastable
            kv_cache:       in non-paged mode, optional (cached_k, cached_v)
                            each (batch, num_kv_heads, cache_len, head_dim);
                            in paged mode, the layer's pool (k, v) tensors of
                            shape (num_pages, page_size, num_kv_heads, head_dim).
            attention_mask: optional float mask (batch, 1, q_len, kv_len)
                            for batched decode with padded KV; 0 = attend,
                            -inf = ignore.
            paged_metadata: when set, take the PagedAttention path
                            (flash-attn varlen for prefill,
                            flash_attn_with_kvcache for decode); kv_cache is
                            then the layer's pool (k_pool, v_pool).

        Returns:
            output:       (batch, seq_len, hidden_size)
            new_kv_cache: (k, v) with updated cache (None in paged mode —
                          the kernel writes directly into the pool).
        """
        q, k, v = self._qkv_with_rope(hidden, cos, sin)
        attn_flat, new_kv = self._attn_kernel(
            q, k, v, kv_cache, attention_mask, paged_metadata
        )
        return self.o_proj(attn_flat), new_kv

    def _paged_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seq_len: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        meta: PagedAttentionMetadata,
    ) -> tuple[torch.Tensor, None]:
        assert kv_cache is not None, "paged attention requires the pool's (k, v) tensors"
        k_pool, v_pool = kv_cache
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        if meta.is_prefill:
            flash_attn_varlen_func, _ = _import_flash_attn()

            # (1, H, T, D) → (T, H, D) packed layout the varlen kernel wants.
            q_packed = q.squeeze(0).transpose(0, 1).contiguous()
            k_packed = k.squeeze(0).transpose(0, 1).contiguous()
            v_packed = v.squeeze(0).transpose(0, 1).contiguous()

            # Scatter freshly computed K/V into the pool at the requested
            # physical slots (PagedAttention §4.3 Fig. 6).
            k_pool.view(-1, num_kv_heads, head_dim).index_copy_(
                0, meta.slot_mapping, k_packed
            )
            v_pool.view(-1, num_kv_heads, head_dim).index_copy_(
                0, meta.slot_mapping, v_packed
            )

            out_packed = flash_attn_varlen_func(
                q_packed,
                k_packed,
                v_packed,
                cu_seqlens_q=meta.cu_seqlens,
                cu_seqlens_k=meta.cu_seqlens,
                max_seqlen_q=meta.max_seqlen,
                max_seqlen_k=meta.max_seqlen,
                causal=True,
            )
            out = out_packed.view(1, seq_len, self.num_heads * head_dim)
            return out, None

        q_dec = q.transpose(1, 2).contiguous()
        k_new = k.transpose(1, 2).contiguous()
        v_new = v.transpose(1, 2).contiguous()

        if meta.flashinfer is not None:
            # FlashInfer wants (B, H, D) — no seq dim — for decode-style
            # single-token-per-row attention. Our q_dec is (B, 1, H, D)
            # post-transpose, so squeeze the seq dim before/after the call.
            assert seq_len == 1, "FlashInfer paged decode is 1-token-per-row"
            out_flat = meta.flashinfer.attend_and_append(
                q_dec.squeeze(1),
                k_new.squeeze(1),
                v_new.squeeze(1),
                k_pool,
                v_pool,
            )
            out = out_flat.view(bsz, 1, self.num_heads * head_dim)
            return out, None

        _, flash_attn_with_kvcache = _import_flash_attn()

        out = flash_attn_with_kvcache(
            q_dec,
            k_pool,
            v_pool,
            k=k_new,
            v=v_new,
            cache_seqlens=meta.cache_seqlens,
            block_table=meta.block_table,
            causal=True,
            num_splits=meta.num_splits,
        )
        out = out.view(bsz, seq_len, self.num_heads * head_dim)
        return out, None


# ── MLP ─────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer block ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: LN → Attn → residual → LN → MLP → residual.

    The forward is split around the opaque flash-attn call into two
    arithmetic-heavy sub-regions, :meth:`_pre_attn` and :meth:`_post_attn`,
    so the engine can wrap each one with :func:`torch.compile` and unlock
    Inductor fusions (RMSNorm-into-GEMM, gate*up*silu, residual epilogs)
    while leaving the un-traceable flash-attn kernel un-compiled.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _pre_attn(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """input_layernorm + Q/K/V projection + QK-Norm + RoPE.

        Bigger than the MLP-only region: 3 GEMMs (Q, K, V) + 3 RMSNorms +
        RoPE — enough arithmetic to amortize ``torch.compile`` per-call
        dispatch overhead, with fusion opportunities at the LN→GEMM and
        QK-Norm→RoPE seams.
        """
        h = self.input_layernorm(hidden)
        return self.self_attn._qkv_with_rope(h, cos, sin)

    def _post_attn(
        self,
        attn_flat: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """o_proj + residual + post_attention_layernorm + MLP + residual.

        The largest single contiguous arithmetic region in the layer:
        4 GEMMs (o_proj, gate, up, down) + 1 RMSNorm + SwiGLU + 2 residual
        adds. Inductor can fuse o_proj/down_proj epilogs with the residual
        adds, fuse silu*mul, and fuse the post-norm into the gate/up
        projection inputs.
        """
        h = self.self_attn.o_proj(attn_flat)
        h = residual + h
        r = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return r + h

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_metadata: PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden
        q, k, v = self._pre_attn(hidden, cos, sin)
        attn_flat, new_kv = self.self_attn._attn_kernel(
            q, k, v, kv_cache, attention_mask, paged_metadata
        )
        out = self._post_attn(attn_flat, residual)
        return out, new_kv


# ── Full model ──────────────────────────────────────────────────────────


class TransformerModel(nn.Module):
    """The core transformer: embedding → N layers → final norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_metadata: PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """
        Args:
            input_ids:      (batch, seq_len)
            position_ids:   (batch, seq_len)
            kv_caches:      list of per-layer (key, value) caches, or None.
                            In paged mode this is the pool's per-layer
                            (k, v) tensors.
            attention_mask: optional float mask for batched-decode SDPA
            paged_metadata: optional metadata that switches every layer to
                            the PagedAttention path (flash-attn).

        Returns:
            hidden:         (batch, seq_len, hidden_size)
            new_kv_caches:  list of per-layer (key, value) with appended
                            tokens (each entry is None in paged mode).
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden, new_kv = layer(
                hidden, cos, sin, kv, attention_mask, paged_metadata
            )
            new_kv_caches.append(new_kv)

        hidden = self.norm(hidden)
        return hidden, new_kv_caches


class CausalLM(nn.Module):
    """
    Complete causal language model: transformer + LM head.

    The LM head may be tied with the embedding (Qwen3-4B) or separate.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = TransformerModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_metadata: PagedAttentionMetadata | None = None,
        last_token_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """
        Args:
            last_token_indices: optional 1-D index tensor selecting positions
                along ``seq_len`` *before* the LM head. Used by packed prefill
                to avoid materialising ``(1, T, vocab)`` logits when only the
                last position of each prompt is needed.

        Returns:
            logits:        (batch, seq_len_or_selected, vocab_size)
            new_kv_caches: per-layer KV caches (entries are None in paged mode)
        """
        hidden, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches, attention_mask, paged_metadata
        )
        if last_token_indices is not None:
            hidden = hidden[:, last_token_indices, :]
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits, new_kv_caches


# ── Weight loading ──────────────────────────────────────────────────────


def load_weights(
    model: CausalLM,
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """
    Load weights from HuggingFace safetensors into the model.

    Handles both single-file and sharded checkpoints.  Weight names in the
    checkpoint match our module hierarchy exactly (by design), so we can
    use load_state_dict() directly.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info("Downloading / locating model files for %s …", model_path)
    local_path = Path(
        snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )

    # Gather all safetensor shard files
    st_files = sorted(local_path.glob("model*.safetensors"))
    if not st_files:
        # Fallback: some repos use a single "model.safetensors"
        st_files = sorted(local_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {local_path}")

    logger.info("Loading %d safetensors shard(s) …", len(st_files))

    # Load shards straight to the target device — avoids a full CPU copy.
    state_dict: dict[str, torch.Tensor] = {}
    for f in st_files:
        for key, tensor in load_file(str(f), device=device).items():
            state_dict[key] = tensor.to(dtype=dtype)

    # Drop checkpoint keys the model doesn't expect.
    model_keys = set(model.state_dict().keys())
    extra = set(state_dict.keys()) - model_keys
    for key in extra:
        del state_dict[key]
    if extra:
        logger.info("Skipped %d unexpected checkpoint keys", len(extra))

    if "lm_head.weight" in model_keys and "lm_head.weight" not in state_dict:
        logger.info("Tying lm_head.weight to embed_tokens.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # assign=True: replace meta tensors in-place rather than copy_ into them.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    if missing:
        logger.warning("Missing keys after load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys after load: %s", unexpected)

    # RoPE inv_freq is a non-persistent buffer (not in checkpoint), so it's
    # still on the meta device after assign=True — materialize it now.
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = 1.0 / (
                module.theta
                ** (
                    torch.arange(
                        0, module.head_dim, 2, device=device, dtype=torch.float32
                    )
                    / module.head_dim
                )
            )
    logger.info(
        "Weights loaded — %d parameters on %s (%s)",
        sum(p.numel() for p in model.parameters()),
        device,
        dtype,
    )
