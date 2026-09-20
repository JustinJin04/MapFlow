# SPDX-License-Identifier: Apache-2.0
"""MInference-style dynamic block-sparse prefill attention.

This is a deliberately small baseline for long decoder prefills.  It follows
the block-sparse branch of MInference: mean-pool Q/K, select important causal
KV blocks for every query block, and run a fused sparse attention kernel.  The
existing FlexPrefill Triton kernel provides the sparse FlashAttention compute.
"""

import math

import torch

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flex_prefill_kernel import (
    triton_block_wise_attention,
)


def _paged_kv_to_dense(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    cache_block_size = cache.shape[1]
    num_pages = math.ceil(seq_len / cache_block_size)
    page_ids = block_table_row[:num_pages].to(torch.long)
    return cache.index_select(0, page_ids).flatten(0, 1)[:seq_len]


def _mean_pool_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pool [tokens, heads, dim] without materializing padded token tensors."""
    seq_len = x.shape[0]
    num_blocks = math.ceil(seq_len / block_size)
    if seq_len % block_size == 0:
        return x.view(num_blocks, block_size, *x.shape[1:]).mean(dim=1)

    full_blocks = seq_len // block_size
    pooled = []
    if full_blocks:
        pooled.append(
            x[: full_blocks * block_size]
            .view(full_blocks, block_size, *x.shape[1:])
            .mean(dim=1)
        )
    pooled.append(x[full_blocks * block_size :].mean(dim=0, keepdim=True))
    return torch.cat(pooled, dim=0)


def build_block_sparse_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
    top_k: int,
    local_blocks: int,
    sink_blocks: int,
) -> torch.Tensor:
    """Build flattened causal block indices in the format used by FlexPrefill."""
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_q_per_kv = num_q_heads // num_kv_heads
    q_blocks = _mean_pool_blocks(q, block_size)
    k_blocks = _mean_pool_blocks(k, block_size)
    num_blocks = q_blocks.shape[0]

    # [heads, query blocks, key blocks].  Grouped-query heads share pooled K.
    k_blocks = k_blocks.repeat_interleave(num_q_per_kv, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q_blocks, k_blocks)
    block_ids = torch.arange(num_blocks, device=q.device)
    causal = block_ids[None, :] <= block_ids[:, None]
    scores.masked_fill_(~causal.unsqueeze(0), torch.finfo(scores.dtype).min)

    dynamic_k = min(top_k, num_blocks)
    row_ids = block_ids[:, None]
    fixed_cols = []
    fixed_valid = []

    # MInference's A-shape component: global sink plus a local causal window.
    if sink_blocks:
        sinks = block_ids[: min(sink_blocks, num_blocks)].expand(num_blocks, -1)
        fixed_cols.append(sinks)
        fixed_valid.append(sinks <= row_ids)
    if local_blocks:
        offsets = torch.arange(local_blocks - 1, -1, -1, device=q.device)
        local = row_ids - offsets
        fixed_cols.append(local.clamp_min(0))
        fixed_valid.append(local >= 0)

    # Do not spend the dynamic budget on blocks already covered by A-shape.
    fixed = torch.cat(fixed_cols, dim=-1) if fixed_cols else row_ids
    valid = torch.cat(fixed_valid, dim=-1) if fixed_valid else torch.ones_like(
        row_ids, dtype=torch.bool
    )
    scores.scatter_(
        -1,
        fixed.unsqueeze(0).expand(num_q_heads, -1, -1),
        torch.finfo(scores.dtype).min,
    )
    dynamic = scores.topk(dynamic_k, dim=-1).indices if dynamic_k else None
    dynamic_valid = (
        torch.gather(causal.unsqueeze(0).expand(num_q_heads, -1, -1), -1, dynamic)
        if dynamic is not None
        else None
    )
    fixed = fixed.unsqueeze(0).expand(num_q_heads, -1, -1)
    valid = valid.unsqueeze(0).expand(num_q_heads, -1, -1)
    cols = torch.cat((fixed, dynamic), dim=-1) if dynamic is not None else fixed
    valid = (
        torch.cat((valid, dynamic_valid), dim=-1)
        if dynamic_valid is not None
        else valid
    )
    previous = torch.tril(
        torch.ones(cols.shape[-1], cols.shape[-1], device=q.device, dtype=torch.bool),
        diagonal=-1,
    )
    duplicate = (
        (cols.unsqueeze(-2) == cols.unsqueeze(-1))
        & valid.unsqueeze(-2)
        & previous
    ).any(dim=-1)
    valid &= ~duplicate

    # Flatten (query block, key block). Invalid slots use the sentinel consumed
    # by FlexPrefill's binning kernel and are not visited by sparse attention.
    flat = row_ids.unsqueeze(0) * num_blocks + cols
    flat.masked_fill_(~valid, num_blocks * (num_blocks + 1))
    return flat.flatten(1).unsqueeze(0)


@torch.no_grad()
def minference_varlen_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    window_size: list[int] | None,
    block_table: torch.Tensor,
    softcap: float | None = 0.0,
    scheduler_metadata: torch.Tensor | None = None,
    fa_version: int | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_splits: int = 0,
    s_aux: torch.Tensor | None = None,
    block_size: int = 64,
    top_k: int = 8,
    local_blocks: int = 4,
    sink_blocks: int = 1,
    min_seq_len: int = 4096,
) -> None:
    """Run sparse attention only for a single, full causal decoder prefill."""
    num_reqs = cu_seqlens_q.shape[0] - 1
    k_len = int(seqused_k[0].item()) if num_reqs == 1 else 0
    use_sparse = (
        num_reqs == 1
        and q.shape[0] == k_len
        and q.shape[0] >= min_seq_len
        and causal
        and alibi_slopes is None
        and not softcap
        and s_aux is None
        and (window_size is None or all(size < 0 for size in window_size))
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and v_cache.dtype == torch.bfloat16
        and q.shape[-1] <= 128
    )
    # print(f"use_sparse: {use_sparse}, num_reqs: {num_reqs}, q.shape[0]: {q.shape[0]}, k_len: {k_len}, min_seq_len: {min_seq_len}")
    if not use_sparse:
        flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            alibi_slopes=alibi_slopes,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=num_splits,
            s_aux=s_aux,
        )
        return

    k = _paged_kv_to_dense(k_cache, block_table[0], k_len)
    v = _paged_kv_to_dense(v_cache, block_table[0], k_len)
    indices = build_block_sparse_indices(
        q, k, block_size, top_k, local_blocks, sink_blocks
    )
    sparse_out = triton_block_wise_attention(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        indices,
        block_size,
        softmax_scale=softmax_scale,
    )
    out.copy_(sparse_out.squeeze(0))
