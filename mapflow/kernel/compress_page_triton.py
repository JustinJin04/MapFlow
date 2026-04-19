import torch
import triton
import triton.language as tl

@triton.jit
def _get_k_cache_block_1(
    block_table_ptr, # [max_num_seq_pages]
    block_table_stride,
    k_page_base_ptr, # [1, page_size, head_dim]
    k_cache_stride_0,
    seq_block_pos,
    pages_per_block: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    tl.device_assert(seq_block_pos >= 0, "1: seq_block_pos must be non-negative")
    tl.device_assert(seq_block_pos < 2560 // pages_per_block, "1: seq_block_pos out of range for block table")
    page_seq_pos = tl.arange(0, pages_per_block) + seq_block_pos * pages_per_block
    page_off = tl.load(block_table_ptr + page_seq_pos * block_table_stride)  # [4]
    tl.device_assert(page_off >= 0, "1: page_off must be non-negative")
    tl.device_assert(page_off < 20000, "1: page_off out of range")
    k_block = tl.load(
        k_page_base_ptr
        + page_off[:, None, None] * k_cache_stride_0
    ).reshape(block_size, head_dim)  # [block_size, head_dim]
    return k_block

@triton.jit
def _get_k_cache_block_2(
    block_table_ptr, # [max_num_seq_pages]
    block_table_stride,
    k_page_base_ptr, # [1, page_size, head_dim]
    k_cache_stride_0,
    seq_block_pos,
    pages_per_block: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    tl.device_assert(seq_block_pos >= 0, "2: seq_block_pos must be non-negative")
    tl.device_assert(seq_block_pos < 2560 // pages_per_block, "2: seq_block_pos out of range for block table")
    page_seq_pos = tl.arange(0, pages_per_block) + seq_block_pos * pages_per_block
    page_off = tl.load(block_table_ptr + page_seq_pos * block_table_stride)  # [4]
    tl.device_assert(page_off >= 0, "2: page_off must be non-negative")
    tl.device_assert(page_off < 20000, "2: page_off out of range")
    k_block = tl.load(
        k_page_base_ptr
        + page_off[:, None, None] * k_cache_stride_0
    ).reshape(block_size, head_dim)  # [block_size, head_dim]
    return k_block

@triton.jit
def _block_stats_kernel_per_head(
    q_ptr, q_stride_0, q_stride_1, q_stride_2, q_stride_3,
    k_cache_ptr, k_cache_stride_0, k_cache_stride_1, k_cache_stride_2, k_cache_stride_3,
    block_table_ptr, block_table_stride,
    block_row_lse_ptr, block_row_lse_stride_0, block_row_lse_stride_1, block_row_lse_stride_2, block_row_lse_stride_3,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    page_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    scale: tl.constexpr,
):
    q_pid = tl.program_id(0)
    k_pid = tl.program_id(1)
    head_pid = tl.program_id(2)

    num_blocks_q = tl.num_programs(0)
    num_blocks_k = tl.num_programs(1)
    q_seq_block_id = q_pid + num_blocks_k - num_blocks_q

    range_b = tl.arange(0, block_size)
    range_d = tl.arange(0, head_dim)
    range_p = tl.arange(0, page_size)

    if k_pid > q_seq_block_id:
        out_ptr = (
            block_row_lse_ptr
            + q_pid * block_row_lse_stride_0
            + k_pid * block_row_lse_stride_1
            + head_pid * block_row_lse_stride_2
            + range_b * block_row_lse_stride_3
        )
        tl.store(out_ptr, float("-inf"))
        return

    kv_head = head_pid // group_size

    q_block = tl.load(
        q_ptr
        + q_pid * q_stride_0
        + range_b[:, None] * q_stride_1
        + head_pid * q_stride_2
        + range_d[None, :] * q_stride_3
    )  # [B, D]

    k_page_base_ptr = (
        k_cache_ptr
        + kv_head * k_cache_stride_2
        + range_p[None, :, None] * k_cache_stride_1
        + range_d[None, None, :] * k_cache_stride_3
    )

    k_block = _get_k_cache_block_1(
        block_table_ptr=block_table_ptr,
        block_table_stride=block_table_stride,
        k_page_base_ptr=k_page_base_ptr,
        k_cache_stride_0=k_cache_stride_0,
        seq_block_pos=k_pid,
        pages_per_block=pages_per_block,
        block_size=block_size,
        head_dim=head_dim,
    )  # [B, D]

    qk = tl.dot(q_block, k_block.T) * scale  # [B, B]

    if q_seq_block_id == k_pid:
        causal = range_b[:, None] >= range_b[None, :]
        qk = tl.where(causal, qk, float("-inf"))

    row_max = tl.max(qk, axis=1)
    row_sumexp = tl.sum(tl.exp(qk - row_max[:, None]), axis=1)
    row_lse = row_max + tl.log(row_sumexp)

    out_ptr = (
        block_row_lse_ptr
        + q_pid * block_row_lse_stride_0
        + k_pid * block_row_lse_stride_1
        + head_pid * block_row_lse_stride_2
        + range_b * block_row_lse_stride_3
    )
    tl.store(out_ptr, row_lse)


def block_stats_triton(
    q: torch.Tensor, # [num_tokens_q, m, D]
    k_cache: torch.Tensor, # [num_pages, page_size, m_kv, D]
    block_table: torch.Tensor, # [max_num_seq_pages]
    block_size: int,
    num_blocks_q: int,
    num_blocks_k: int,
)-> torch.Tensor:
    m = q.shape[1]
    _, page_size, m_kv, D = k_cache.shape

    block_row_lse = torch.empty(num_blocks_q, num_blocks_k, m, block_size, device=q.device)
    
    q = q.view(num_blocks_q, block_size, m, D)
    grid = (num_blocks_q, num_blocks_k, m)
    _block_stats_kernel_per_head[grid](
        q_ptr=q,
        q_stride_0=q.stride(0), q_stride_1=q.stride(1), q_stride_2=q.stride(2), q_stride_3=q.stride(3),
        k_cache_ptr=k_cache,
        k_cache_stride_0=k_cache.stride(0), k_cache_stride_1=k_cache.stride(1), 
        k_cache_stride_2=k_cache.stride(2), k_cache_stride_3=k_cache.stride(3),
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        block_row_lse_ptr=block_row_lse,
        block_row_lse_stride_0=block_row_lse.stride(0), block_row_lse_stride_1=block_row_lse.stride(1), block_row_lse_stride_2=block_row_lse.stride(2), block_row_lse_stride_3=block_row_lse.stride(3),
        block_size=block_size,
        head_dim=D,
        group_size=m//m_kv,
        page_size=page_size,
        pages_per_block=block_size // page_size,
        scale=D**-0.5,
    )

    return block_row_lse

@triton.jit
def _top_p_binary_kernel(
    block_sumprob_ptr,  # [num_blocks_q, num_blocks_k, n]
    block_sumprob_stride_0,  # [num_blocks_k, n]
    block_sumprob_stride_1,  # [n]
    block_sumprob_stride_2,  # 1
    mask_ptr,  # [num_blocks_q, num_blocks_k, n]
    mask_stride_0,  # [num_blocks_k, n]
    mask_stride_1,  # [n]
    mask_stride_2,  # 1
    threshold, # top_p * block_size
    num_blocks_k,
    num_blocks_k_aligned: tl.constexpr, # next power of 2
):
    q_pid = tl.program_id(0)  # [0, num_blocks_q)
    h_pid = tl.program_id(1)  # [0, n)
    num_blocks_q = tl.num_programs(0)

    q_seq_block_id = q_pid + num_blocks_k - num_blocks_q

    range_k = tl.arange(0, num_blocks_k_aligned)
    k_mask = (range_k < num_blocks_k) & (range_k <= q_seq_block_id)
    row_ptr = (
        block_sumprob_ptr
        + q_pid * block_sumprob_stride_0
        + h_pid * block_sumprob_stride_2
        + range_k * block_sumprob_stride_1
    )
    output_mask_ptr = (
        mask_ptr
        + q_pid * mask_stride_0
        + h_pid * mask_stride_2
        + range_k * mask_stride_1
    )
    row = tl.load(row_ptr, mask=k_mask, other=0.0)  # [num_blocks_k]
    
    l = 0.0
    r = tl.max(tl.where(k_mask, row, float("-inf")))
    m = (l+r)/2.0
    # w_0 = tl.where(row < m, 0.0, row)
    w_0 = tl.where(k_mask & (row >= m), row, 0.0)
    # w_1 = tl.where(row <= l, float("inf"), row)
    w_1 = tl.where(k_mask & (row > l), row, float("inf"))
    # w_2 = tl.where(row > r, float("-inf"), row)
    w_2 = tl.where(k_mask & (row <= r), row, float("-inf"))
    if tl.sum(w_0) >= threshold:
        l = m
    else:
        r = m
    gap = tl.max(w_2) - tl.min(w_1)
    while gap > 1e-5:
        m = (l+r)/2.0
        # w_0 = tl.where(row < m, 0.0, row)
        w_0 = tl.where(k_mask & (row >= m), row, 0.0)
        # w_1 = tl.where(row <= l, float("inf"), row)
        w_1 = tl.where(k_mask & (row > l), row, float("inf"))
        # w_2 = tl.where(row > r, float("-inf"), row)
        w_2 = tl.where(k_mask & (row <= r), row, float("-inf"))
        if tl.sum(w_0) >= threshold:
            l = m
        else:
            r = m
        gap = tl.max(w_2) - tl.min(w_1)

    final_mask = (row >= l).to(tl.int8)
    tl.store(output_mask_ptr, final_mask, mask=k_mask)

def top_p_binary_triton(block_sumprob: torch.Tensor, top_p: float, block_size: int):
    """
    block_sumprob: [num_blocks_q, num_blocks_k, n]
    mask: [num_blocks_q, num_blocks_k, n]
    """
    num_blocks_q, num_blocks_k, n = block_sumprob.shape
    mask = torch.zeros_like(block_sumprob, dtype=torch.bool)
    grid = (num_blocks_q, n)
    threshold = top_p * block_size
    _top_p_binary_kernel[grid](
        block_sumprob_ptr=block_sumprob,
        block_sumprob_stride_0=block_sumprob.stride(0),
        block_sumprob_stride_1=block_sumprob.stride(1),
        block_sumprob_stride_2=block_sumprob.stride(2),
        mask_ptr=mask,
        mask_stride_0=mask.stride(0),
        mask_stride_1=mask.stride(1),
        mask_stride_2=mask.stride(2),
        threshold=threshold,
        num_blocks_k=num_blocks_k,
        num_blocks_k_aligned=triton.next_power_of_2(num_blocks_k),
    )
    return mask

def top_p_baseline(block_sumprob, top_p, block_size):
    """
    block_sumprob: [num_blocks_q, num_blocks_k, n]
    mask: [num_blocks_q, num_blocks_k, n]
    """
    sorted_scores, sorted_indices = torch.sort(block_sumprob, dim=1, descending=True)  # [num_blocks_q, num_blocks_k, n]
    cumsum = torch.cumsum(sorted_scores, dim=1)  # [num_blocks_q, num_blocks_k, n]
    first_col_true = torch.ones_like(cumsum[:, :1, :], device=block_sumprob.device, dtype=torch.bool)  # [num_blocks_q, 1, n]
    shifted_sorted_mask = torch.cat([first_col_true, (cumsum <= top_p * block_size)[:, :-1, :]], dim=1)  # [num_blocks_q, num_blocks_k, n]
    mask = torch.zeros_like(shifted_sorted_mask, device=block_sumprob.device, dtype=torch.bool)  # [num_blocks_q, num_blocks_k, n]
    mask.scatter_(dim=1, index=sorted_indices, src=shifted_sorted_mask)  # [num_blocks_q, num_blocks_k, n]
    return mask

def get_bsr_metadata(mask):
    """
    mask: [num_blocks_q, num_blocks_kv, n]
    return: 
     - crow_indices: [num_blocks_q+1],
     - col_indices: [nnz],
    """
    num_blocks_q, num_blocks_kv, n = mask.shape
    mask = mask.to(torch.bool)
    num_blocks_q, num_blocks_kv, n = mask.shape
    crow_indices = torch.zeros((num_blocks_q + 1), device=mask.device, dtype=torch.int32)
    crow_indices[1:] = torch.cumsum(mask.sum(dim=(-1, -2)), dim=0)
    col_indices = torch.nonzero(mask.view(num_blocks_q, -1), as_tuple=True)[1]
    # nnz_offsets = torch.full((num_blocks_q, num_blocks_kv, n), -1, device=mask.device, dtype=torch.int32)
    # nnz_offsets[mask] = torch.arange(col_indices.shape[0], device=mask.device)
    return crow_indices, col_indices

negative_tensor = torch.tensor(-1, device="cuda", dtype=torch.int32)

def get_bsr_metadata_nnz_offsets(mask):
    """
    mask: [num_blocks_q, num_blocks_kv, n], torch.bool
    return:
     - crow_indices: [num_blocks_q+1],
     - col_indices: [nnz],
     - nnz_offsets: [num_blocks_q, num_blocks_kv, n], -1 means not selected, otherwise gives the offset in values array
    """
    num_blocks_q, num_blocks_kv, n = mask.shape
    mask_int32 = mask.to(torch.int32)
    mask_q_sum = mask_int32.sum(dim=(1, 2))  # [num_blocks_q]
    crow_indices = torch.zeros((num_blocks_q + 1), device=mask.device, dtype=torch.int32)
    torch.cumsum(mask_q_sum, dim=0, out=crow_indices[1:])
    col_indices = torch.nonzero(mask.view(num_blocks_q, -1), as_tuple=True)[1]
    mask_flat = mask_int32.view(-1)
    exc_cumsum = torch.cumsum(mask_flat, dim=0) - mask_flat
    nnz_offsets_flat = torch.where(
        mask.view(-1), 
        exc_cumsum,
        negative_tensor
    )
    nnz_offsets = nnz_offsets_flat.view(num_blocks_q, num_blocks_kv, n)
    return crow_indices, col_indices, nnz_offsets

@triton.jit
def _sparse_extract_kernel(
    q_ptr, # [num_blocks_q, block_size, m, D]
    q_stride_0, q_stride_1, q_stride_2, q_stride_3,
    k_cache_ptr, # [num_pages, page_size, m_kv, D]
    k_cache_stride_0, k_cache_stride_1, k_cache_stride_2, k_cache_stride_3,
    block_table_ptr, # [max_num_seq_pages]
    block_table_stride,
    row_lse_ptr, # [num_blocks_q, num_heads, block_size]
    row_lse_stride_0, row_lse_stride_1, row_lse_stride_2,
    nnz_offsets_ptr, # [num_blocks_q, num_blocks_kv, m]
    nnz_offsets_stride_0, nnz_offsets_stride_1, nnz_offsets_stride_2,
    values_ptr, # [nnz, block_size, block_size]
    values_stride_0, values_stride_1, values_stride_2,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,  # m // m_kv
    page_size: tl.constexpr,
    pages_per_block: tl.constexpr,  # block_size // page_size
    scale: tl.constexpr,
):
    q_pid = tl.program_id(0)
    k_pid = tl.program_id(1)
    head_pid = tl.program_id(2)
    num_blocks_q = tl.num_programs(0)
    num_blocks_k = tl.num_programs(1)
    q_seq_block_id = q_pid + num_blocks_k - num_blocks_q
    
    range_b = tl.arange(0, block_size)
    range_d = tl.arange(0, head_dim)
    range_p = tl.arange(0, page_size)

    q_block = tl.load(
        q_ptr
        + q_pid * q_stride_0
        + head_pid * q_stride_2
        + range_b[:, None] * q_stride_1
        + range_d[None, :] * q_stride_3
    ) # [block_size, head_dim]
    k_page_base_ptr = (
        k_cache_ptr
        + head_pid//group_size * k_cache_stride_2
        + range_p[None, :, None] * k_cache_stride_1
        + range_d[None, None, :] * k_cache_stride_3
    ) # [1, page_size, head_dim]
    k_block = _get_k_cache_block_2(
        block_table_ptr=block_table_ptr,
        block_table_stride=block_table_stride,
        k_page_base_ptr=k_page_base_ptr,
        k_cache_stride_0=k_cache_stride_0,
        seq_block_pos=k_pid,
        pages_per_block=pages_per_block,
        block_size=block_size,
        head_dim=head_dim,
    ) # [block_size, head_dim]
    nnz_offset = tl.load(
        nnz_offsets_ptr
        + q_pid * nnz_offsets_stride_0
        + k_pid * nnz_offsets_stride_1
        + head_pid * nnz_offsets_stride_2
    )
    row_lse = tl.load(
        row_lse_ptr
        + q_pid * row_lse_stride_0
        + head_pid * row_lse_stride_1
        + range_b * row_lse_stride_2
    ) # [block_size]
    value_block_ptr = (
        values_ptr
        + range_b[:, None] * values_stride_1
        + range_b[None, :] * values_stride_2
    ) # [block_size, block_size], offset by nnz_offset for each block

    qk_block = tl.dot(q_block, k_block.T) * scale
    if q_seq_block_id == k_pid:
        mask = range_b[:, None] >= range_b[None, :]
        qk_block = tl.where(mask, qk_block, float("-inf"))
    attn_block = tl.exp(qk_block - row_lse[:, None])  # [block_size, block_size]
    tl.store(value_block_ptr + nnz_offset * values_stride_0, attn_block, mask=nnz_offset >= 0)

def sparse_extract_triton(
    q: torch.Tensor, # [num_tokens, num_heads, head_dim]
    k_cache: torch.Tensor, # [num_pages, page_size, num_kv_heads, head_dim]
    block_table: torch.Tensor, # [max_num_seq_pages]
    num_tokens_kv: int,
    nnz_offsets: torch.Tensor, # [num_blocks_q, num_blocks_kv, num_heads]
    row_lse: torch.Tensor, # [num_blocks_q, num_heads, block_size]
    values: torch.Tensor, # [nnz, block_size, block_size]
):
    assert nnz_offsets.dtype == torch.int32, f"Expected nnz_offsets to be int32, but got {nnz_offsets.dtype}"
    num_blocks_q, m, block_size = row_lse.shape
    q = q.view(num_blocks_q, block_size, m, -1)
    D = q.shape[-1]
    _, page_size, m_kv, _ = k_cache.shape
    num_blocks_kv = num_tokens_kv // block_size

    grid = (num_blocks_q, num_blocks_kv, m)
    _sparse_extract_kernel[grid](
        q_ptr=q,
        q_stride_0=q.stride(0), q_stride_1=q.stride(1), q_stride_2=q.stride(2), q_stride_3=q.stride(3),
        k_cache_ptr=k_cache,
        k_cache_stride_0=k_cache.stride(0), k_cache_stride_1=k_cache.stride(1), 
        k_cache_stride_2=k_cache.stride(2), k_cache_stride_3=k_cache.stride(3),
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        row_lse_ptr=row_lse,
        row_lse_stride_0=row_lse.stride(0), row_lse_stride_1=row_lse.stride(1), row_lse_stride_2=row_lse.stride(2),
        nnz_offsets_ptr=nnz_offsets,
        nnz_offsets_stride_0=nnz_offsets.stride(0), nnz_offsets_stride_1=nnz_offsets.stride(1), nnz_offsets_stride_2=nnz_offsets.stride(2),
        values_ptr=values,
        values_stride_0=values.stride(0), values_stride_1=values.stride(1), values_stride_2=values.stride(2),
        block_size=block_size,
        head_dim=D,
        group_size=m//m_kv,
        page_size=page_size,
        pages_per_block=block_size // page_size,
        scale=D**-0.5,
    )

def get_top_p_metas(
    q: torch.Tensor, # [num_tokens, num_heads, head_dim]
    k_cache: torch.Tensor, # [num_pages, page_size, num_kv_heads, head_dim]
    block_table: torch.Tensor, # [max_num_seq_pages]
    num_tokens_kv: int,
    block_size: int,
    top_p: float,
):
    num_tokens_q, m, D = q.shape
    _, page_size, m_kv, _ = k_cache.shape
    assert num_tokens_q % block_size == 0
    assert num_tokens_kv % block_size == 0
    num_blocks_q = num_tokens_q // block_size
    num_blocks_k = num_tokens_kv // block_size

    # Compute block stats
    block_row_lse = block_stats_triton(
        q=q, 
        k_cache=k_cache, 
        block_table=block_table,
        block_size=block_size,
        num_blocks_q=num_blocks_q,
        num_blocks_k=num_blocks_k,
    )

    row_lse = torch.logsumexp(block_row_lse, dim=1)  # [Qb, m, B]
    block_sumprob = torch.exp(block_row_lse - row_lse.unsqueeze(1)).sum(dim=-1)  # [Qb, Kb, m]

    # Top-p filter
    mask = top_p_binary_triton(block_sumprob, top_p, block_size)

    return mask, row_lse

@triton.jit
def _dense_to_sparse_indices_kernel(
    mask_ptr,
    cumsum_ptr,
    col_indices_ptr,
    col_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = offsets < n_elements
    
    m = tl.load(mask_ptr + offsets, mask=valid_mask, other=0)
    
    # cumsum gives inclusive prefix sum, so the destination index is cumsum - 1
    write_idx = tl.load(cumsum_ptr + offsets, mask=valid_mask, other=0) - 1
    
    # Calculate column index equivalent to mask.view(num_blocks_q, -1)
    col_idx = (offsets % col_dim).to(tl.int32)
    
    # Only write to the pre-allocated buffer where mask is True
    tl.store(col_indices_ptr + write_idx, col_idx, mask=valid_mask & (m == 1))

def get_bsr_metadata_async(mask: torch.Tensor):
    """
    Calculates BSR metadata completely asynchronously.
    Returns: crow_indices, a pre-allocated col_buffer, nnz_offsets, and a 1D GPU tensor of the exact nnz.
    """
    num_blocks_q, num_blocks_kv, n = mask.shape
    col_dim = num_blocks_kv * n
    n_elements = mask.numel()
    
    mask_int32 = mask.to(torch.int32)
    mask_flat = mask_int32.view(-1)
    
    # 1. crow_indices (Async)
    mask_q_sum = mask_int32.sum(dim=(1, 2))
    crow_indices = torch.zeros((num_blocks_q + 1), device=mask.device, dtype=torch.int32)
    torch.cumsum(mask_q_sum, dim=0, out=crow_indices[1:])
    
    # 2. nnz_offsets (Async)
    cumsum = torch.cumsum(mask_flat, dim=0)
    exc_cumsum = cumsum - mask_flat
    nnz_offsets = torch.where(mask.view(-1), exc_cumsum, negative_tensor).view(num_blocks_q, num_blocks_kv, n).to(torch.int32)
    
    # 3. col_indices into pre-allocated buffer (Async)
    # Allocating max possible size is practically free due to PyTorch's caching allocator
    col_prealloc = torch.empty(n_elements, dtype=torch.int32, device=mask.device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dense_to_sparse_indices_kernel[grid](
        mask_flat, cumsum, col_prealloc, col_dim, n_elements, BLOCK_SIZE=1024
    )
    
    # 4. Total nnz (Keep it on GPU to prevent an implicit sync!)
    total_nnz_tensor = cumsum[-1:] 
    
    return crow_indices, col_prealloc, nnz_offsets, total_nnz_tensor