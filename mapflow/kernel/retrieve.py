import torch
import triton
import triton.language as tl

@triton.jit
def _retrieve(
    hit_nnz_ptr, # [H, L]
    src_off_ptr,  # [H, L]
    dst_off_ptr,  # [H, L]
    src_ptr,  # [H, L, ...] in flattened format
    dst_ptr,  # [L, H, ...] in flattened format
    L: tl.constexpr,
    max_nnz: tl.constexpr,
):
    h = tl.program_id(0)
    l = tl.program_id(1)

    nnz = tl.load(hit_nnz_ptr + h * L + l)
    src_off = tl.load(src_off_ptr + h * L + l)
    dst_off = tl.load(dst_off_ptr + h * L + l)

    range_max_nnz = tl.arange(0, max_nnz)
    mask = range_max_nnz < nnz

    value = tl.load(
        src_ptr + src_off + range_max_nnz,
        mask=mask
    )
    tl.store(
        dst_ptr + dst_off + range_max_nnz,
        value,
        mask=mask
    )


zero_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")


def retrieve_triton(
    hit_nnz: torch.Tensor, # [H, L]
    src_off: torch.Tensor,  # [H, L]
    dst_off: torch.Tensor,  # [H, L]
    src_tensor: torch.Tensor,  # [H, L, ...] in flattened format
    max_nnz: int = 65536
):
    # generate src_off and dst_off
    H, L = hit_nnz.shape
    # src_off = torch.cat([zero_tensor, torch.cumsum(hit_nnz.view(-1), dim=0)])[:-1].view(H, L)  # [H, L]
    # dst_off = torch.cat([zero_tensor, torch.cumsum(hit_nnz.T.contiguous().view(-1), dim=0)])[:-1].view(L, H).T.contiguous()  # [H, L]
    dst_tensor = torch.empty_like(src_tensor)  # [L, H, ...] in flattened format

    grid = (H, L)
    _retrieve[grid](
        hit_nnz_ptr=hit_nnz,
        src_off_ptr=src_off,
        dst_off_ptr=dst_off,
        src_ptr=src_tensor,
        dst_ptr=dst_tensor,
        L=L,
        max_nnz=max_nnz
    )
    return dst_tensor


@triton.jit
def _retrieve_optimize(
    hit_nnz_ptr, # [H, L]
    src_off_ptr,  # [H, L]
    dst_off_ptr,  # [H, L]
    src_ptr,  # [H, L, ...] in flattened format
    dst_ptr,  # [L, H, ...] in flattened format
    L: tl.constexpr,
    max_nnz: tl.constexpr,
):
    h = tl.program_id(0)

    range_max_nnz = tl.arange(0, max_nnz)

    for l in range(L):
        nnz = tl.load(hit_nnz_ptr + h * L + l)
        src_off = tl.load(src_off_ptr + h * L + l)
        dst_off = tl.load(dst_off_ptr + h * L + l)
        mask = range_max_nnz < nnz
        value = tl.load(
            src_ptr + src_off + range_max_nnz,
            mask=mask
        )
        tl.store(
            dst_ptr + dst_off + range_max_nnz,
            value,
            mask=mask
        )


def retrieve_triton_optimize(
    hit_nnz: torch.Tensor, # [H, L]
    src_off: torch.Tensor,  # [H, L]
    dst_off: torch.Tensor,  # [H, L]
    src_tensor: torch.Tensor,  # Flattened
    max_nnz: int = 65536
):
    H, L = src_off.shape
    
    # generate src_off and dst_off
    # src_off = torch.cat([zero_tensor, torch.cumsum(hit_nnz.view(-1), dim=0)])[:-1].view(H, L)  
    # dst_off = torch.cat([zero_tensor, torch.cumsum(hit_nnz.T.contiguous().view(-1), dim=0)])[:-1].view(L, H).T.contiguous()  
    dst_tensor = torch.empty_like(src_tensor)  

    grid = (H,)     
    _retrieve_optimize[grid](
        hit_nnz_ptr=hit_nnz,
        src_off_ptr=src_off,
        dst_off_ptr=dst_off,
        src_ptr=src_tensor,
        dst_ptr=dst_tensor,
        L=L,
        max_nnz=max_nnz
    )
    return dst_tensor


@triton.jit()
def _retrieve_layer(
    src_prefix_sum_ptr, # [H]
    src_prefix_sum_ptr_stride,
    dst_prefix_sum_ptr, # [H+1]
    dst_prefix_sum_ptr_stride,
    src_ptr,  # [H, L, ...] in flattened format
    dst_ptr,  # [L, H, ...] in flattened format
    max_nnz: tl.constexpr,
):
    h = tl.program_id(0)
    src_start = tl.load(src_prefix_sum_ptr + h * src_prefix_sum_ptr_stride)
    dst_start = tl.load(dst_prefix_sum_ptr + h * dst_prefix_sum_ptr_stride)
    dst_end = tl.load(dst_prefix_sum_ptr + (h + 1) * dst_prefix_sum_ptr_stride)

    range_max_nnz = tl.arange(0, max_nnz)
    dst_size = dst_end - dst_start
    mask = range_max_nnz < dst_size
    value = tl.load(
        src_ptr + src_start + range_max_nnz,
        mask=mask
    )
    tl.store(
        dst_ptr + dst_start + range_max_nnz,
        value,
        mask=mask
    )

def retrieve_layer_triton(
    src_prefix_sum: torch.Tensor, # [H]
    dst_prefix_sum: torch.Tensor, # [H+1]
    src_tensor: torch.Tensor,  # [H, L, ...] in flattened format
    dst_total_size: int,
    max_nnz: int = 65536
)-> torch.Tensor:
    dst_tensor = torch.empty(dst_total_size, dtype=src_tensor.dtype, device=src_tensor.device)
    H = src_prefix_sum.shape[0]
    grid = (H,)
    _retrieve_layer[grid](
        src_prefix_sum_ptr=src_prefix_sum,
        src_prefix_sum_ptr_stride=src_prefix_sum.stride(0),
        dst_prefix_sum_ptr=dst_prefix_sum,
        dst_prefix_sum_ptr_stride=dst_prefix_sum.stride(0),
        src_ptr=src_tensor,
        dst_ptr=dst_tensor,
        max_nnz=max_nnz
    )
    return dst_tensor



def retrieve_cpu(
    hit_nnz: torch.Tensor,     # [H, L]
    src_tensor: torch.Tensor,  # [H, L, ...] in flattened format
):
    H, L = hit_nnz.shape
    
    # 1. 计算按行主序 (Row-major) 的源偏移量
    hit_nnz_flat = hit_nnz.flatten()
    src_offsets = torch.zeros_like(hit_nnz_flat)
    src_offsets[1:] = hit_nnz_flat[:-1].cumsum(dim=0)
    
    # 2. 计算按列主序 (Column-major) 的目标偏移量
    # 转置后再展平，计算累加和
    hit_nnz_col = hit_nnz.t().flatten()
    dst_offsets_col = torch.zeros_like(hit_nnz_col)
    dst_offsets_col[1:] = hit_nnz_col[:-1].cumsum(dim=0)
    
    # 关键：将列主序的偏移量，重新对齐回行主序的视角 (H, L)
    dst_offsets = dst_offsets_col.view(L, H).t().flatten()
    
    # 3. 计算每个数据块的索引偏移量 (Shift)
    shifts = dst_offsets - src_offsets
    
    # 4. 利用 repeat_interleave 将 Block 级别的 Shift 展开为 Element 级别的 Shift
    # 这是最核心的一步：比如某个块有 3 个元素，偏移量是 +5，这步就会生成 [5, 5, 5]
    element_shifts = torch.repeat_interleave(shifts, hit_nnz_flat)
    
    # 5. 计算全局 Scatter 索引
    N = hit_nnz_flat.sum().item()
    base_indices = torch.arange(N, device=hit_nnz.device)
    dst_indices = base_indices + element_shifts
    
    # 6. 单次向量化赋值：消除所有的切片对象创建！
    dst_tensor = torch.empty_like(src_tensor)
    dst_tensor[dst_indices] = src_tensor
    
    return dst_tensor


