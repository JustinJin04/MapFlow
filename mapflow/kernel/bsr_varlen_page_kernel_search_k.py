import math
from tqdm import tqdm
import numpy as np
import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    for group_row in [2, 4, 8]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4, 5]:
                configs.append(
                    triton.Config({'group_row': group_row}, num_warps=num_warps, num_stages=num_stages)
                )
    return configs

SUPPORTED_MAX_SPLIT_K = 64
SUPPORTED_SPLIT_KS = tuple(range(2, SUPPORTED_MAX_SPLIT_K+1))
def get_split_k_configs(split_ks=SUPPORTED_SPLIT_KS):
    configs = []
    for split_k in split_ks:
        for group_row in [2, 4, 8]:
            for num_warps in [4, 8]:
                for num_stages in [2, 3, 4, 5]:
                    configs.append(
                        triton.Config({'split_k': split_k, 'group_row': group_row}, num_warps=num_warps, num_stages=num_stages)
                    )
    return configs

@triton.jit()
def swizzle_tile(pid,
                 m, n,
                 block_m: tl.constexpr, block_n: tl.constexpr, group_row: tl.constexpr):

    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_row * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_row, group_row)

    pid_m = group_id * group_row + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.jit
def _get_recv_value_block(
    cur_block_table_ptr, # [max_num_pages_per_req]
    block_table_stride_1,
    cur_col_value_cache_ptr, # [1, 16, D]
    value_cache_page_stride, # 16 * num_kv_heads * head_dim
    bsr_seq_pos,
    pages_per_block: tl.constexpr, # BLOCK_SIZE // page_size, 4
    BLOCK_SIZE: tl.constexpr,
    D_block_size: tl.constexpr,
):
    tl.device_assert(bsr_seq_pos >= 0, "bsr_seq_pos must be non-negative")
    tl.device_assert(bsr_seq_pos < 2560 // pages_per_block, "bsr_seq_pos out of range for block table")
    page_seq_pos = tl.arange(0, pages_per_block) + bsr_seq_pos * pages_per_block  # [4]
    page_offs = tl.load(cur_block_table_ptr + block_table_stride_1 * page_seq_pos)  # [4]

    value_block = tl.load(
        cur_col_value_cache_ptr
        + page_offs[:, None, None] * value_cache_page_stride
    )  # [4, 16, D]
    
    return tl.reshape(value_block, (BLOCK_SIZE, D_block_size))  # [64, D]

def prune_num_warps_for_n32(configs, named_args, **kwargs):
    n = named_args.get("N", kwargs.get("N", None))
    if n == 32:
        pruned = [cfg for cfg in configs if cfg.num_warps != 4]
        return pruned if pruned else configs
    elif n == 16:
        pruned = [cfg for cfg in configs if cfg.num_warps != 8]
        return pruned if pruned else configs
    return configs

def early_config_prune(configs, named_args, **kwargs):
    configs = prune_num_warps_for_n32(configs, named_args, **kwargs)

    def _prev_pow2(x: int) -> int:
        if x <= 1:
            return 1
        return 1 << ((x - 1).bit_length() - 1)

    def _next_pow2(x: int) -> int:
        if x <= 1:
            return 1
        p = _prev_pow2(x)
        return p if p == x else (p << 1)

    est = kwargs["estimated_split_k"]
    max_split_k = kwargs["max_split_k"]
    supported_split_ks = {
        cfg.kwargs["split_k"]
        for cfg in configs
        if "split_k" in cfg.kwargs
    }

    prev_pow2 = _prev_pow2(est)
    next_pow2 = _next_pow2(est)
    raw_keep = {
        est // 2,
        est - 1,
        est,
        est + 1,
        est * 2,
        prev_pow2,
        next_pow2,
    }
    keep = {
        k for k in raw_keep
        if 2 <= k <= max_split_k and k in supported_split_ks
    }

    pruned = [cfg for cfg in configs if cfg.kwargs["split_k"] in keep]
    return pruned if pruned else configs

@triton.autotune(
    configs=get_configs(),
    key=['N', 'num_packed_block_rows_arg', 'avg_row_nnz_bucket', 'density'],
    prune_configs_by={
        'early_config_prune': prune_num_warps_for_n32,
    },
    cache_results=True,
)
@triton.jit
def _decompose_varlen_page_fused(
    data_blocks_ptr,
    data_blocks_stride_0,
    data_blocks_stride_1,
    data_blocks_stride_2,
    packed_block_indices_ptr,
    block_indices_stride,
    packed_crow_indices_ptr,
    crow_indices_stride,
    packed_col_indices_ptr,
    col_indices_stride,
    cu_crow_indices_ptr,  # [num_reqs + 1],
    cu_crow_indices_stride,
    cu_col_indices_ptr,  # [num_reqs + 1],
    cu_col_indices_stride,
    block_table_ptr, # [num_reqs, max_num_pages_per_req]
    block_table_stride_0,
    block_table_stride_1,
    value_cache_ptr, # [..., 16, num_kv_heads*head_dim]
    value_cache_page_stride, # 16 * num_kv_heads * head_dim
    value_cache_row_stride, # num_kv_heads * head_dim
    value_cache_col_stride, # 1
    packed_output_ptr,  # [num_tokens, num_heads*head_dim]
    output_row_stride,
    output_col_stride,
    cu_seqlens_ptr,  # [num_reqs + 1], each req truncated to aligned lengths
    cu_seqlens_stride,
    weights_ptr,  # [M, N]
    packed_row_block_pid_to_seq_id_ptr,  # [num_packed_row_blocks], like [0, 0, 1, 1, 1, 1, 1, 2, 2, 2] for [2, 5, 3]
    packed_row_block_pid_to_seq_id_stride,
    packed_row_block_pid_to_row_block_pid_ptr,  # [num_packed_row_blocks], like [0, 1, 0, 1, 2, 3, 4, 0, 1, 2] for [2, 5, 3]
    packed_row_block_pid_to_row_block_pid_stride,
    num_packed_block_rows_arg,  # autotune key: total packed row blocks
    avg_row_nnz_bucket,          # autotune key: bucketed avg nnz per block-row
    density,                    # autotune key: estimated density (0.03, 0.04, ..., 0.15)
    gqa_group_size: tl.constexpr,      # N // N_kv
    GROUP_SIZE: tl.constexpr,      # gqa_group_size * BLOCK_SIZE
    M: tl.constexpr,
    M_ALIGNED: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HeadDim: tl.constexpr,
    data_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    page_size: tl.constexpr, # 16
    pages_per_block: tl.constexpr, # BLOCK_SIZE // page_size, 4
    D_block_size: tl.constexpr,  # equals to BLOCK_SIZE or HeadDim
    group_row: tl.constexpr,  # provided by autotune
):
    pid = tl.program_id(0)  # Combined packed_row&col pid
    n_block_cols = (N // gqa_group_size) * (HeadDim // D_block_size)
    n_packed_block_rows = tl.num_programs(0) // n_block_cols  # num_tokens // BLOCK_SIZE
    packed_row_block_pid, kv_col_block_pid = swizzle_tile(
        pid,
        n_packed_block_rows,
        n_block_cols,
        1, 1,
        group_row
    )
    seq_id = tl.load(packed_row_block_pid_to_seq_id_ptr + packed_row_block_pid * packed_row_block_pid_to_seq_id_stride)
    row_block_pid = tl.load(packed_row_block_pid_to_row_block_pid_ptr + packed_row_block_pid * packed_row_block_pid_to_row_block_pid_stride)

    crow_indices_ptr = packed_crow_indices_ptr + tl.load(cu_crow_indices_ptr + cu_crow_indices_stride * seq_id) * crow_indices_stride
    col_indices_ptr = packed_col_indices_ptr + tl.load(cu_col_indices_ptr + cu_col_indices_stride * seq_id) * col_indices_stride
    # values_ptr = packed_values_ptr + tl.load(cu_col_indices_ptr + cu_col_indices_stride * seq_id) * values_nnz_stride
    block_index_ptr = packed_block_indices_ptr + tl.load(cu_col_indices_ptr + cu_col_indices_stride * seq_id) * block_indices_stride
    output_ptr = packed_output_ptr + tl.load(cu_seqlens_ptr + cu_seqlens_stride * seq_id) * output_row_stride
    cur_block_table_ptr = block_table_ptr + seq_id * block_table_stride_0

    output_tiled_row_stride = output_row_stride * BLOCK_SIZE
    output_tiled_col_stride = output_col_stride * D_block_size


    n_block_within_head = HeadDim // D_block_size
    kv_head_idx = kv_col_block_pid // n_block_within_head
    block_offset_in_head = kv_col_block_pid % n_block_within_head
    start_q_head_idx = kv_head_idx * gqa_group_size

    range_g = tl.arange(0, GROUP_SIZE)
    range_b = tl.arange(0, BLOCK_SIZE)
    range_d = tl.arange(0, D_block_size)
    # range_p = tl.arange(0, page_size)
    group_ids = range_g // BLOCK_SIZE # [0, 0, ..., 0, 1, ..., 1]
    row_in_block = range_g % BLOCK_SIZE # [0, 1, ..., B-1, 0, 1, ..., B-1]

    crow_indices_offset_ptr = crow_indices_ptr + crow_indices_stride * row_block_pid
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
    row_nnz = nnz_offset_next - nnz_offset
    if row_nnz == 0:
        return
    col_index_nnz_ptr = col_indices_ptr + col_indices_stride * nnz_offset
    block_index_nnz_ptr = block_index_ptr + block_indices_stride * nnz_offset

    values_block_base_ptrs = (
        data_blocks_ptr
        + data_blocks_stride_1 * row_in_block[:, None] 
        + data_blocks_stride_2 * range_b[None, :]
    )

    cur_col_value_cache_ptr = (
        value_cache_ptr
        + kv_col_block_pid * value_cache_col_stride * D_block_size
        + value_cache_row_stride * tl.arange(0, page_size)[None, :, None]
        + value_cache_col_stride * range_d[None, None, :]
    ) # [1, 16, D]

    # range_m = tl.arange(0, M)[:, None]  # [M, 1]
    range_m = tl.arange(0, M_ALIGNED)[:, None]
    mask_m = range_m < M
    # weights_cache = tl.load(weights_ptr + range_m * N + target_head_idxs[None, :])  # [M, GROUP_SIZE]
    weights_cache = tl.load(
        weights_ptr + range_m * N + start_q_head_idx + tl.arange(0, gqa_group_size)[None, :],
        mask=mask_m,
        other=0.0
    )
    attn_acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE), dtype=data_dtype)
    output_acc = tl.zeros((GROUP_SIZE, D_block_size), dtype=acc_dtype)

    # Main Loop
    # last_seq_pos = start_pos // BLOCK_SIZE
    last_seq_pos = tl.load(col_index_nnz_ptr) // M
    for _ in range(row_nnz):
        col_index = tl.load(col_index_nnz_ptr)
        head_idx_in_bsr = col_index % M
        current_seq_pos = col_index // M

        # Flush logic
        if current_seq_pos != last_seq_pos:
            # since it's not the last seq block, no need to mask load
            # current_recv_ptrs = (
            #     recv_values_base_ptr
            #     + recv_values_tiled_row_stride * last_seq_pos
            # )
            # recv_values_blocks = tl.load(current_recv_ptrs)
            recv_values_blocks = _get_recv_value_block(
                cur_block_table_ptr=cur_block_table_ptr,
                block_table_stride_1=block_table_stride_1,
                cur_col_value_cache_ptr=cur_col_value_cache_ptr,
                value_cache_page_stride=value_cache_page_stride,
                bsr_seq_pos=last_seq_pos,
                pages_per_block=pages_per_block,
                BLOCK_SIZE=BLOCK_SIZE,
                D_block_size=D_block_size,
            )

            # [gqa_group_size*BLOCK_SIZE, BLOCK_SIZE] x [BLOCK_SIZE, D_block_size]
            output_acc += tl.dot(
                attn_acc,
                recv_values_blocks,
                out_dtype=acc_dtype,
            )
            attn_acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE), dtype=data_dtype)
            last_seq_pos = current_seq_pos

        # Accumulate attention from sender map
        # since 2048000 * 64 * 64 exceeds the max of int32, we need to convert it to int64 first
        block_idx = tl.load(block_index_nnz_ptr).to(tl.int64)
        value_block = tl.load(
            values_block_base_ptrs
            + block_idx * data_blocks_stride_0
        )
        sel = (head_idx_in_bsr == range_m) & mask_m
        weights_sum = tl.sum(tl.where(sel, weights_cache, 0.0), axis=0)[:, None]  # [gqa_group_size, 1]
        weights = tl.broadcast_to(weights_sum, (gqa_group_size, BLOCK_SIZE))
        weights = tl.reshape(weights, (gqa_group_size * BLOCK_SIZE, 1))  # [gqa_group_size*BLOCK_SIZE, 1]

        # [G*B, B] * [G*B, 1]
        attn_acc += value_block * weights

        col_index_nnz_ptr += col_indices_stride
        block_index_nnz_ptr += block_indices_stride
    
    # Final Flush
    recv_values_blocks = _get_recv_value_block(
        cur_block_table_ptr=cur_block_table_ptr,
        block_table_stride_1=block_table_stride_1,
        cur_col_value_cache_ptr=cur_col_value_cache_ptr,
        value_cache_page_stride=value_cache_page_stride,
        bsr_seq_pos=last_seq_pos,
        pages_per_block=pages_per_block,
        BLOCK_SIZE=BLOCK_SIZE,
        D_block_size=D_block_size,
    )
    output_acc += tl.dot(
        attn_acc.to(data_dtype),
        recv_values_blocks,
        out_dtype=acc_dtype,
    )

    # Store Output
    head_stride = output_tiled_col_stride * n_block_within_head
    block_in_head_stride = output_tiled_col_stride
    output_ptrs = (
        output_ptr
        + output_tiled_row_stride * row_block_pid
        + head_stride * (start_q_head_idx + group_ids)[:, None]  # Use group_ids to map to heads
        + block_in_head_stride * block_offset_in_head
        + output_row_stride * row_in_block[:, None]        # Use row_in_block for row offsets
        + output_col_stride * range_d[None, :]
    )
    
    tl.store(output_ptrs, output_acc.to(data_dtype))

def bsr_varlen_page_kernel(
    data_blocks: torch.Tensor, # [..., block_size, block_size]
    block_indices: torch.Tensor, # [max_num_blocks],
    crow: torch.Tensor,   # [max_num_seq_blocks],
    col: torch.Tensor,    # [max_num_blocks],
    cu_crow_indices: torch.Tensor, # [num_reqs + 1],
    cu_col_indices: torch.Tensor,  # [num_reqs + 1],
    # cu_block_rows: torch.Tensor,      # [num_reqs + 1], [0, 2, 7, 10] for [2, 5, 3]
    bsr_batch_offsets: torch.Tensor,  # [num_reqs + 1],  [0, 2*64+2, 7*64+7, 10*64+10] for [2, 5, 3] with block_size=64
    num_packed_block_rows: int,
    packed_row_block_pid_to_seq_id: torch.Tensor, # [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    packed_row_block_pid_to_row_block_pid: torch.Tensor, # [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    weights: torch.Tensor,  # [M, N]
    block_table: torch.Tensor, # [num_reqs, max_num_pages_per_req]
    value_cache: torch.Tensor, # [..., 16, num_kv_heads, head_dim]
    output: torch.Tensor,     # [total_seq_len, num_heads, head_dim]
    density: int,
):
    m, n = weights.shape
    # _, L, n_kv, D = recv_value_cache.shape
    _, page_size, n_kv, D = value_cache.shape
    block_size = data_blocks.shape[1]
    value_cache_view = value_cache.view(-1, page_size, n_kv * D)
    output_view = output.view(-1, n * D)

    if data_blocks.dtype == torch.float16:
        data_dtype = tl.float16
    elif data_blocks.dtype == torch.bfloat16:
        data_dtype = tl.bfloat16
    elif data_blocks.dtype == torch.float32:
        data_dtype = tl.float32
    else:
        assert 0

    # Compute autotune keys
    total_nnz = col.shape[0]
    avg_row_nnz = total_nnz // max(1, num_packed_block_rows)
    avg_row_nnz_bucket = 1 << max(0, math.ceil(math.log2(max(1, avg_row_nnz))))
    avg_row_nnz_bucket = min(avg_row_nnz_bucket, 4096)

    def grid_fn(META):
        d_block = META['D_block_size']
        return (num_packed_block_rows * (n_kv * D) // d_block, )
    _decompose_varlen_page_fused[grid_fn](
        data_blocks_ptr=data_blocks,
        data_blocks_stride_0=data_blocks.stride(0),
        data_blocks_stride_1=data_blocks.stride(1),
        data_blocks_stride_2=data_blocks.stride(2),
        packed_block_indices_ptr=block_indices,
        block_indices_stride=block_indices.stride(0),
        packed_crow_indices_ptr=crow,
        crow_indices_stride=crow.stride(0),
        packed_col_indices_ptr=col,
        col_indices_stride=col.stride(0),
        cu_crow_indices_ptr=cu_crow_indices,
        cu_crow_indices_stride=cu_crow_indices.stride(0),
        cu_col_indices_ptr=cu_col_indices,
        cu_col_indices_stride=cu_col_indices.stride(0),
        block_table_ptr=block_table,
        block_table_stride_0=block_table.stride(0),
        block_table_stride_1=block_table.stride(1),
        value_cache_ptr=value_cache_view,
        value_cache_page_stride=value_cache_view.stride(0),
        value_cache_row_stride=value_cache_view.stride(1),
        value_cache_col_stride=value_cache_view.stride(2),
        packed_output_ptr=output_view,
        output_row_stride=output_view.stride(0),
        output_col_stride=output_view.stride(1),
        cu_seqlens_ptr=bsr_batch_offsets,
        cu_seqlens_stride=1,
        weights_ptr=weights,
        packed_row_block_pid_to_seq_id_ptr=packed_row_block_pid_to_seq_id,
        packed_row_block_pid_to_seq_id_stride=1,
        packed_row_block_pid_to_row_block_pid_ptr=packed_row_block_pid_to_row_block_pid,
        packed_row_block_pid_to_row_block_pid_stride=1,
        num_packed_block_rows_arg=num_packed_block_rows,
        avg_row_nnz_bucket=avg_row_nnz_bucket,
        density=density,
        gqa_group_size=n // n_kv,
        GROUP_SIZE=n // n_kv * block_size,
        M=m,
        M_ALIGNED=triton.next_power_of_2(m),
        N=n,
        BLOCK_SIZE=block_size,
        HeadDim=D,
        D_block_size=D,
        page_size=page_size,
        pages_per_block=block_size // page_size,
        data_dtype=data_dtype,
        acc_dtype=tl.float32,
    )

@triton.autotune(
    configs=get_split_k_configs(),
    key=['N', 'num_packed_block_rows_arg', 'avg_row_nnz_bucket', 'estimated_split_k', 'density'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'top_k': 8,
    },
    cache_results=True,
)
@triton.jit
def _decompose_varlen_page_fused_splitk_stage1(
    data_blocks_ptr,
    data_blocks_stride_0,
    data_blocks_stride_1,
    data_blocks_stride_2,
    packed_block_indices_ptr,
    block_indices_stride,
    packed_crow_indices_ptr,
    crow_indices_stride,
    packed_col_indices_ptr,
    col_indices_stride,
    cu_crow_indices_ptr,
    cu_crow_indices_stride,
    cu_col_indices_ptr,
    cu_col_indices_stride,
    block_table_ptr,
    block_table_stride_0,
    block_table_stride_1,
    value_cache_ptr,
    value_cache_page_stride,
    value_cache_row_stride,
    value_cache_col_stride,

    # partial workspace: [split_k, total_seq_len, N * HeadDim], fp32
    partial_ptr,
    partial_split_stride,
    partial_row_stride,
    partial_col_stride,

    cu_seqlens_ptr,
    cu_seqlens_stride,
    weights_ptr,
    packed_row_block_pid_to_seq_id_ptr,
    packed_row_block_pid_to_seq_id_stride,
    packed_row_block_pid_to_row_block_pid_ptr,
    packed_row_block_pid_to_row_block_pid_stride,

    # autotune key
    num_packed_block_rows_arg,
    avg_row_nnz_bucket,
    density,
    estimated_split_k,
    max_split_k,

    gqa_group_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    M: tl.constexpr,
    M_ALIGNED: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HeadDim: tl.constexpr,
    data_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    page_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    split_k: tl.constexpr,
    D_block_size: tl.constexpr,  # equals to BLOCK_SIZE or HeadDim
    group_row: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_split = tl.program_id(1)

    n_block_cols = (N // gqa_group_size) * (HeadDim // D_block_size)
    n_packed_block_rows = tl.num_programs(0) // n_block_cols

    packed_row_block_pid, kv_col_block_pid = swizzle_tile(
        pid_tile,
        n_packed_block_rows,
        n_block_cols,
        1, 1,
        group_row,
    )

    seq_id = tl.load(
        packed_row_block_pid_to_seq_id_ptr
        + packed_row_block_pid * packed_row_block_pid_to_seq_id_stride
    )
    row_block_pid = tl.load(
        packed_row_block_pid_to_row_block_pid_ptr
        + packed_row_block_pid * packed_row_block_pid_to_row_block_pid_stride
    )

    crow_indices_ptr = (
        packed_crow_indices_ptr
        + tl.load(cu_crow_indices_ptr + cu_crow_indices_stride * seq_id) * crow_indices_stride
    )
    col_indices_ptr = (
        packed_col_indices_ptr
        + tl.load(cu_col_indices_ptr + cu_col_indices_stride * seq_id) * col_indices_stride
    )
    block_index_ptr = (
        packed_block_indices_ptr
        + tl.load(cu_col_indices_ptr + cu_col_indices_stride * seq_id) * block_indices_stride
    )
    cur_block_table_ptr = block_table_ptr + seq_id * block_table_stride_0

    seq_start = tl.load(cu_seqlens_ptr + cu_seqlens_stride * seq_id)
    partial_base_ptr = (
        partial_ptr
        + pid_split * partial_split_stride
        + seq_start * partial_row_stride
    )

    n_block_within_head = HeadDim // D_block_size
    kv_head_idx = kv_col_block_pid // n_block_within_head
    block_offset_in_head = kv_col_block_pid % n_block_within_head
    start_q_head_idx = kv_head_idx * gqa_group_size

    range_g = tl.arange(0, GROUP_SIZE)
    range_b = tl.arange(0, BLOCK_SIZE)
    range_d = tl.arange(0, D_block_size)

    group_ids = range_g // BLOCK_SIZE
    row_in_block = range_g % BLOCK_SIZE

    crow_indices_offset_ptr = crow_indices_ptr + crow_indices_stride * row_block_pid
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
    row_nnz = nnz_offset_next - nnz_offset

    # -------- split this row_nnz --------
    split_begin = (row_nnz * pid_split) // split_k
    split_end   = (row_nnz * (pid_split + 1)) // split_k
    split_nnz   = split_end - split_begin

    values_block_base_ptrs = (
        data_blocks_ptr
        + data_blocks_stride_1 * row_in_block[:, None]
        + data_blocks_stride_2 * range_b[None, :]
    )

    cur_col_value_cache_ptr = (
        value_cache_ptr
        + kv_col_block_pid * value_cache_col_stride * D_block_size
        + value_cache_row_stride * tl.arange(0, page_size)[None, :, None]
        + value_cache_col_stride * range_d[None, None, :]
    )

    # range_m = tl.arange(0, M)[:, None]
    range_m = tl.arange(0, M_ALIGNED)[:, None]
    mask_m = range_m < M
    weights_cache = tl.load(
        weights_ptr + range_m * N + start_q_head_idx + tl.arange(0, gqa_group_size)[None, :],
        mask=mask_m,
        other=0.0
    )

    attn_acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE), dtype=data_dtype)
    output_acc = tl.zeros((GROUP_SIZE, D_block_size), dtype=acc_dtype)

    if split_nnz > 0:
        col_index_nnz_ptr = col_indices_ptr + col_indices_stride * (nnz_offset + split_begin)
        block_index_nnz_ptr = block_index_ptr + block_indices_stride * (nnz_offset + split_begin)

        last_seq_pos = tl.load(col_index_nnz_ptr) // M

        for _ in range(split_nnz):
            col_index = tl.load(col_index_nnz_ptr)
            head_idx_in_bsr = col_index % M
            current_seq_pos = col_index // M

            if current_seq_pos != last_seq_pos:
                recv_values_blocks = _get_recv_value_block(
                    cur_block_table_ptr=cur_block_table_ptr,
                    block_table_stride_1=block_table_stride_1,
                    cur_col_value_cache_ptr=cur_col_value_cache_ptr,
                    value_cache_page_stride=value_cache_page_stride,
                    bsr_seq_pos=last_seq_pos,
                    pages_per_block=pages_per_block,
                    BLOCK_SIZE=BLOCK_SIZE,
                    D_block_size=D_block_size,
                )

                output_acc += tl.dot(
                    attn_acc,
                    recv_values_blocks,
                    out_dtype=acc_dtype,
                )
                attn_acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE), dtype=data_dtype)
                last_seq_pos = current_seq_pos

            block_idx = tl.load(block_index_nnz_ptr).to(tl.int64)
            value_block = tl.load(values_block_base_ptrs + block_idx * data_blocks_stride_0)

            sel = (head_idx_in_bsr == range_m) & mask_m
            weights_sum = tl.sum(tl.where(sel, weights_cache, 0.0), axis=0)[:, None]
            weights = tl.broadcast_to(weights_sum, (gqa_group_size, BLOCK_SIZE))
            weights = tl.reshape(weights, (gqa_group_size * BLOCK_SIZE, 1))

            attn_acc += value_block * weights

            col_index_nnz_ptr += col_indices_stride
            block_index_nnz_ptr += block_indices_stride

        recv_values_blocks = _get_recv_value_block(
            cur_block_table_ptr=cur_block_table_ptr,
            block_table_stride_1=block_table_stride_1,
            cur_col_value_cache_ptr=cur_col_value_cache_ptr,
            value_cache_page_stride=value_cache_page_stride,
            bsr_seq_pos=last_seq_pos,
            pages_per_block=pages_per_block,
            BLOCK_SIZE=BLOCK_SIZE,
            D_block_size=D_block_size,
        )
        output_acc += tl.dot(attn_acc.to(data_dtype), recv_values_blocks, out_dtype=acc_dtype)

    # store partial fp32 tile
    partial_tiled_row_stride = partial_row_stride * BLOCK_SIZE
    partial_tiled_col_stride = partial_col_stride * D_block_size
    head_stride = partial_tiled_col_stride * n_block_within_head
    block_in_head_stride = partial_tiled_col_stride

    partial_ptrs = (
        partial_base_ptr
        + partial_tiled_row_stride * row_block_pid
        + head_stride * (start_q_head_idx + group_ids)[:, None]
        + block_in_head_stride * block_offset_in_head
        + partial_row_stride * row_in_block[:, None]
        + partial_col_stride * range_d[None, :]
    )
    tl.store(partial_ptrs, output_acc)

@triton.autotune(
    configs=get_configs(),
    key=['N', 'split_k', 'num_packed_block_rows_arg', 'avg_row_nnz_bucket'],
    cache_results=True,
)
@triton.jit
def _splitk_reduce_tiles(
    partial_ptr,
    partial_split_stride,
    partial_row_stride,
    partial_col_stride,

    packed_output_ptr,
    output_row_stride,
    output_col_stride,

    cu_seqlens_ptr,
    cu_seqlens_stride,
    packed_row_block_pid_to_seq_id_ptr,
    packed_row_block_pid_to_seq_id_stride,
    packed_row_block_pid_to_row_block_pid_ptr,
    packed_row_block_pid_to_row_block_pid_stride,

    # autotune key placeholders
    num_packed_block_rows_arg,
    avg_row_nnz_bucket,

    gqa_group_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HeadDim: tl.constexpr,
    out_dtype: tl.constexpr,
    split_k: tl.constexpr,
    D_block_size: tl.constexpr,  # equals to BLOCK_SIZE or HeadDim
    group_row: tl.constexpr,
):
    pid = tl.program_id(0)

    n_block_cols = (N // gqa_group_size) * (HeadDim // D_block_size)
    n_packed_block_rows = tl.num_programs(0) // n_block_cols

    packed_row_block_pid, kv_col_block_pid = swizzle_tile(
        pid,
        n_packed_block_rows,
        n_block_cols,
        1, 1,
        group_row,
    )

    seq_id = tl.load(
        packed_row_block_pid_to_seq_id_ptr
        + packed_row_block_pid * packed_row_block_pid_to_seq_id_stride
    )
    row_block_pid = tl.load(
        packed_row_block_pid_to_row_block_pid_ptr
        + packed_row_block_pid * packed_row_block_pid_to_row_block_pid_stride
    )
    seq_start = tl.load(cu_seqlens_ptr + cu_seqlens_stride * seq_id)

    n_block_within_head = HeadDim // D_block_size
    kv_head_idx = kv_col_block_pid // n_block_within_head
    block_offset_in_head = kv_col_block_pid % n_block_within_head
    start_q_head_idx = kv_head_idx * gqa_group_size

    range_g = tl.arange(0, GROUP_SIZE)
    range_d = tl.arange(0, D_block_size)
    group_ids = range_g // BLOCK_SIZE
    row_in_block = range_g % BLOCK_SIZE

    partial_tiled_row_stride = partial_row_stride * BLOCK_SIZE
    partial_tiled_col_stride = partial_col_stride * D_block_size
    partial_head_stride = partial_tiled_col_stride * n_block_within_head
    partial_block_in_head_stride = partial_tiled_col_stride

    acc = tl.zeros((GROUP_SIZE, D_block_size), dtype=tl.float32)

    for sp in range(split_k):
        partial_base_ptr = (
            partial_ptr
            + sp * partial_split_stride
            + seq_start * partial_row_stride
        )
        partial_ptrs = (
            partial_base_ptr
            + partial_tiled_row_stride * row_block_pid
            + partial_head_stride * (start_q_head_idx + group_ids)[:, None]
            + partial_block_in_head_stride * block_offset_in_head
            + partial_row_stride * row_in_block[:, None]
            + partial_col_stride * range_d[None, :]
        )
        acc += tl.load(partial_ptrs)

    output_tiled_row_stride = output_row_stride * BLOCK_SIZE
    output_tiled_col_stride = output_col_stride * D_block_size
    output_head_stride = output_tiled_col_stride * n_block_within_head
    output_block_in_head_stride = output_tiled_col_stride

    out_ptrs = (
        packed_output_ptr
        + seq_start * output_row_stride
        + output_tiled_row_stride * row_block_pid
        + output_head_stride * (start_q_head_idx + group_ids)[:, None]
        + output_block_in_head_stride * block_offset_in_head
        + output_row_stride * row_in_block[:, None]
        + output_col_stride * range_d[None, :]
    )
    tl.store(out_ptrs, acc.to(out_dtype))

def bsr_varlen_page_splitk_kernel(
    data_blocks,
    block_indices,
    crow,
    col,
    cu_crow_indices,
    cu_col_indices,
    bsr_batch_offsets,
    num_packed_block_rows,
    packed_row_block_pid_to_seq_id,
    packed_row_block_pid_to_row_block_pid,
    weights,
    block_table,
    value_cache,
    output,
    density,
    estimated_split_k,
    max_split_k,
):
    m, n = weights.shape
    _, page_size, n_kv, D = value_cache.shape
    block_size = data_blocks.shape[1]

    value_cache_view = value_cache.view(-1, page_size, n_kv * D)
    output_view = output.view(-1, n * D)

    if data_blocks.dtype == torch.float16:
        data_dtype = tl.float16
    elif data_blocks.dtype == torch.bfloat16:
        data_dtype = tl.bfloat16
    elif data_blocks.dtype == torch.float32:
        data_dtype = tl.float32
    else:
        raise TypeError(data_blocks.dtype)

    # Compute autotune keys
    total_nnz = col.shape[0]
    avg_row_nnz = total_nnz // max(1, num_packed_block_rows)
    avg_row_nnz_bucket = 1 << max(0, math.ceil(math.log2(max(1, avg_row_nnz))))
    avg_row_nnz_bucket = min(avg_row_nnz_bucket, 4096)

    # partial workspace
    partial = torch.empty(
        (max_split_k, output_view.shape[0], output_view.shape[1]),
        device=output.device,
        dtype=torch.float32,
    )

    def grid_stage1(META):
        d_block = META["D_block_size"]
        split_k = META["split_k"]
        n_block_cols = (n_kv * D) // d_block
        return (num_packed_block_rows * n_block_cols, split_k)

    _decompose_varlen_page_fused_splitk_stage1[grid_stage1](
        data_blocks_ptr=data_blocks,
        data_blocks_stride_0=data_blocks.stride(0),
        data_blocks_stride_1=data_blocks.stride(1),
        data_blocks_stride_2=data_blocks.stride(2),

        packed_block_indices_ptr=block_indices,
        block_indices_stride=block_indices.stride(0),
        packed_crow_indices_ptr=crow,
        crow_indices_stride=crow.stride(0),
        packed_col_indices_ptr=col,
        col_indices_stride=col.stride(0),

        cu_crow_indices_ptr=cu_crow_indices,
        cu_crow_indices_stride=cu_crow_indices.stride(0),
        cu_col_indices_ptr=cu_col_indices,
        cu_col_indices_stride=cu_col_indices.stride(0),

        block_table_ptr=block_table,
        block_table_stride_0=block_table.stride(0),
        block_table_stride_1=block_table.stride(1),

        value_cache_ptr=value_cache_view,
        value_cache_page_stride=value_cache_view.stride(0),
        value_cache_row_stride=value_cache_view.stride(1),
        value_cache_col_stride=value_cache_view.stride(2),

        partial_ptr=partial,
        partial_split_stride=partial.stride(0),
        partial_row_stride=partial.stride(1),
        partial_col_stride=partial.stride(2),

        cu_seqlens_ptr=bsr_batch_offsets,
        cu_seqlens_stride=1,

        weights_ptr=weights,
        packed_row_block_pid_to_seq_id_ptr=packed_row_block_pid_to_seq_id,
        packed_row_block_pid_to_seq_id_stride=1,
        packed_row_block_pid_to_row_block_pid_ptr=packed_row_block_pid_to_row_block_pid,
        packed_row_block_pid_to_row_block_pid_stride=1,

        num_packed_block_rows_arg=num_packed_block_rows,
        avg_row_nnz_bucket=avg_row_nnz_bucket,
        density=density,
        estimated_split_k=estimated_split_k,
        max_split_k=max_split_k,

        gqa_group_size=n // n_kv,
        GROUP_SIZE=(n // n_kv) * block_size,
        M=m,
        M_ALIGNED=triton.next_power_of_2(m),
        N=n,
        BLOCK_SIZE=block_size,
        HeadDim=D,
        D_block_size=D,
        page_size=page_size,
        pages_per_block=block_size // page_size,
        data_dtype=data_dtype,
        acc_dtype=tl.float32,
    )

    split_k = _decompose_varlen_page_fused_splitk_stage1.best_config.kwargs["split_k"]

    def grid_stage2(META):
        d_block = META["D_block_size"]
        n_block_cols = (n_kv * D) // d_block
        return (num_packed_block_rows * n_block_cols,)

    _splitk_reduce_tiles[grid_stage2](
        partial_ptr=partial,
        partial_split_stride=partial.stride(0),
        partial_row_stride=partial.stride(1),
        partial_col_stride=partial.stride(2),

        packed_output_ptr=output_view,
        output_row_stride=output_view.stride(0),
        output_col_stride=output_view.stride(1),

        cu_seqlens_ptr=bsr_batch_offsets,
        cu_seqlens_stride=1,

        packed_row_block_pid_to_seq_id_ptr=packed_row_block_pid_to_seq_id,
        packed_row_block_pid_to_seq_id_stride=1,
        packed_row_block_pid_to_row_block_pid_ptr=packed_row_block_pid_to_row_block_pid,
        packed_row_block_pid_to_row_block_pid_stride=1,
    
        num_packed_block_rows_arg=num_packed_block_rows,
        avg_row_nnz_bucket=avg_row_nnz_bucket,

        gqa_group_size=n // n_kv,
        GROUP_SIZE=(n // n_kv) * block_size,
        N=n,
        BLOCK_SIZE=block_size,
        HeadDim=D,
        D_block_size=D,
        out_dtype=data_dtype,
        split_k=split_k,
    )


MIN_DENSITY=3  # 0.03
MAX_DENSITY=10  # 0.10
# entry point
def bsr_varlen_page_triton(
    data_blocks,
    block_indices,
    crow,
    col,
    cu_crow_indices,
    cu_col_indices,
    bsr_batch_offsets,
    num_packed_block_rows,
    packed_row_block_pid_to_seq_id,
    packed_row_block_pid_to_row_block_pid,
    weights,
    block_table,
    value_cache,
    output,
    density,
    max_split_k=32,
    target_waves=3,
):
    assert max_split_k <= SUPPORTED_MAX_SPLIT_K, f"max_split_k {max_split_k} should be less than or equal to {SUPPORTED_MAX_SPLIT_K}"
    _, page_size, n_kv, D = value_cache.shape

    sm_count = torch.cuda.get_device_properties(output.device).multi_processor_count

    base_tiles = num_packed_block_rows * n_kv
    estimated_split_k = max(1, math.ceil(target_waves * sm_count / max(1, base_tiles)))
    estimated_split_k = min(max_split_k, estimated_split_k)

    density = min(MAX_DENSITY, max(MIN_DENSITY, density))

    if estimated_split_k == 1:
        return bsr_varlen_page_kernel(
            data_blocks,
            block_indices,
            crow,
            col,
            cu_crow_indices,
            cu_col_indices,
            bsr_batch_offsets,
            num_packed_block_rows,
            packed_row_block_pid_to_seq_id,
            packed_row_block_pid_to_row_block_pid,
            weights,
            block_table,
            value_cache,
            output,
            density,
        )
    else:
        return bsr_varlen_page_splitk_kernel(
            data_blocks,
            block_indices,
            crow,
            col,
            cu_crow_indices,
            cu_col_indices,
            bsr_batch_offsets,
            num_packed_block_rows,
            packed_row_block_pid_to_seq_id,
            packed_row_block_pid_to_row_block_pid,
            weights,
            block_table,
            value_cache,
            output,
            density,
            estimated_split_k=estimated_split_k,
            max_split_k=max_split_k,
        )


def get_bsr_attn_map(
    num_row_blocks: int, # [1, 2, ..., 64]
    num_avg_nnz_per_row_block: int, # [1, 2, ..., 1024]
    col_upper_bound: int,
    block_size: int, 
    dtype,
    device,
    density_ratio: float,
):
    crow_indices = [0]
    col_indices = []
    nnz_range = np.arange(0, min(num_avg_nnz_per_row_block // density_ratio, col_upper_bound), dtype=np.int32)
    for row_idx in range(num_row_blocks):
        selected_nnz = np.sort(np.random.choice(nnz_range, size=num_avg_nnz_per_row_block, replace=False))
        col_indices.extend(selected_nnz.tolist())
        crow_indices.append(crow_indices[-1] + num_avg_nnz_per_row_block)
    nnz = crow_indices[-1]
    values = torch.randn((nnz, block_size, block_size), device=device, dtype=dtype)
    crow_indices = torch.tensor(crow_indices, device=device, dtype=torch.int32)
    col_indices = torch.tensor(col_indices, device=device, dtype=torch.int32)

    return values, crow_indices, col_indices

def warmup(
    m: int,
    n: int,
    n_kv: int,
    block_size: int,
    D: int,
    num_row_blocks: int,
    num_avg_nnz_per_row_block: int,
    density: int,
    dtype=torch.bfloat16,
    device="cuda",
):
    L = num_row_blocks * block_size
    col_upper_bound = 2560 // 4 * m
    density_ratio = density / 100
    value_blocks, crow, col = get_bsr_attn_map(num_row_blocks, num_avg_nnz_per_row_block, col_upper_bound, block_size, dtype, device, density_ratio)
    block_indices = torch.arange(value_blocks.shape[0], device=device, dtype=torch.int32)
    cu_crow_indices = torch.tensor([0, crow.shape[0]], dtype=torch.int32, device=device)
    cu_col_indices = torch.tensor([0, col.shape[0]], dtype=torch.int32, device=device)
    bsr_batch_offsets = torch.tensor([0], dtype=torch.int32, device=device)
    num_packed_block_rows = L // block_size
    packed_row_block_pid_to_seq_id = torch.zeros((num_packed_block_rows,), dtype=torch.int32, device=device)
    packed_row_block_pid_to_row_block_pid = torch.arange(num_packed_block_rows, dtype=torch.int32, device=device)
    weights = torch.randn((m, n), device=device, dtype=dtype)
    block_table = torch.arange(2560, device=device, dtype=torch.int32).unsqueeze(0) # [1, 4096]
    value_cache = torch.randn((5120, 16, n_kv, D), device=device, dtype=dtype)
    output = torch.empty((L, n, D), device=device, dtype=dtype)

    bsr_varlen_page_triton(
        data_blocks=value_blocks,
        block_indices=block_indices,
        crow=crow,
        col=col,
        cu_crow_indices=cu_crow_indices,
        cu_col_indices=cu_col_indices,
        bsr_batch_offsets=bsr_batch_offsets,
        num_packed_block_rows=num_packed_block_rows,
        packed_row_block_pid_to_seq_id=packed_row_block_pid_to_seq_id,
        packed_row_block_pid_to_row_block_pid=packed_row_block_pid_to_row_block_pid,
        weights=weights,
        block_table=block_table,
        value_cache=value_cache,
        output=output,
        density=density,
    )

def main(m, n):
    # num_row_blocks_list = np.arange(1, 65).tolist()
    num_row_blocks_list = np.arange(64, 0, -1).tolist()
    # num_row_blocks_list = [64]
    num_avg_nnz_per_row_block_list = (1 << np.arange(6, 13)).tolist()
    # num_avg_nnz_per_row_block_list = (1 << np.arange(6, 8)).tolist()
    density_list = np.arange(MIN_DENSITY, MAX_DENSITY + 1).tolist()
    print(f"start warmup. m: {m}, n: {n}\n"
          f"density_list: {density_list}\n"
          f"num_row_blocks_list: {num_row_blocks_list}\n"
          f"num_avg_nnz_per_row_block_list: {num_avg_nnz_per_row_block_list}", flush=True)
    # print(num_row_blocks_list)
    # print(num_avg_nnz_per_row_block_list)
    for density in tqdm(density_list):
        for num_row_blocks in num_row_blocks_list:
            for num_avg_nnz_per_row_block in num_avg_nnz_per_row_block_list:
                warmup(
                    m=m,
                    n=n,
                    n_kv=8,
                    block_size=64,
                    D=128,
                    num_row_blocks=num_row_blocks,
                    num_avg_nnz_per_row_block=num_avg_nnz_per_row_block,
                    density=density,
                )

    print(f"warmup finished.", flush=True)

def warmup_20k(m, n):
    num_row_blocks_list = np.arange(320, 269, -1).tolist()
    num_avg_nnz_per_row_block_list = (1 << np.arange(6, 13)).tolist()
    # density_list = np.arange(3, 10 + 1).tolist()    
    # density_list = np.arange(3, 8 + 1).tolist()
    density_list = np.arange(3, 7 + 1).tolist()
    # density_list = np.arange(3, 6 + 1).tolist()
    # density_list = np.arange(3, 5 + 1).tolist()
    # density_list = np.arange(3, 4 + 1).tolist()
    # density_list = np.arange(3, 3 + 1).tolist()
    print(f"start warmup. m: {m}, n: {n}\n"
          f"density_list: {density_list}\n"
          f"num_row_blocks_list: {num_row_blocks_list}\n"
          f"num_avg_nnz_per_row_block_list: {num_avg_nnz_per_row_block_list}", flush=True)
    # print(num_row_blocks_list)
    # print(num_avg_nnz_per_row_block_list)
    for density in tqdm(density_list):
        for num_row_blocks in num_row_blocks_list:
            for num_avg_nnz_per_row_block in num_avg_nnz_per_row_block_list:
                warmup(
                    m=m,
                    n=n,
                    n_kv=8,
                    block_size=64,
                    D=128,
                    num_row_blocks=num_row_blocks,
                    num_avg_nnz_per_row_block=num_avg_nnz_per_row_block,
                    density=density,
                )

    print(f"warmup finished.", flush=True)

HAS_WARMUP = False
def run_warmup_once(m, n):
    global HAS_WARMUP
    if not HAS_WARMUP:
        # main(m, n)
        # warmup_20k(m, n)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.reset_accumulated_memory_stats()
        HAS_WARMUP = True
