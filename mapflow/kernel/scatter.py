import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_scatter_blocks_kernel(
    dst_ptr, # [n_dst, B, B]
    dst_s0, dst_s1, dst_s2,
    src_ptr, # [n_src, B, B]
    src_s0, src_s1, src_s2,
    idx_ptr, # [n_src]
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    dst_block = tl.load(idx_ptr + pid).to(tl.int64)
    row_offs = tl.arange(0, TILE_M)
    col_offs = tl.arange(0, TILE_N)

    for rm in tl.range(0, BLOCK_SIZE, TILE_M):
        rows = rm + row_offs
        row_mask = rows < BLOCK_SIZE

        for cn in tl.range(0, BLOCK_SIZE, TILE_N):
            cols = cn + col_offs
            col_mask = cols < BLOCK_SIZE
            mask = row_mask[:, None] & col_mask[None, :]

            src_ptrs = (
                src_ptr
                + pid * src_s0
                + rows[:, None] * src_s1
                + cols[None, :] * src_s2
            )
            dst_ptrs = (
                dst_ptr
                + dst_block * dst_s0
                + rows[:, None] * dst_s1
                + cols[None, :] * dst_s2
            )

            vals = tl.load(src_ptrs, mask=mask, other=0)
            tl.store(dst_ptrs, vals, mask=mask)


@torch.no_grad()
def chunked_scatter(
    dst: torch.Tensor,          # [n_dst, B, B]
    indices: torch.Tensor,      # [n_src]
    src: torch.Tensor,          # [n_src, B, B]
    chunk_n: int = 4096,
    stream: torch.cuda.Stream |None = None,
):
    assert dst.device == src.device == indices.device
    assert dst.dtype == src.dtype
    n_src, B, _ = src.shape
    assert indices.shape == (n_src,)
    if stream is None:
        stream = torch.cuda.current_stream(dst.device)
    with torch.cuda.stream(stream):
        for i in range(0, n_src, chunk_n):
            chunk_size = min(chunk_n, n_src - i)
            grid = (chunk_size,)
            _chunked_scatter_blocks_kernel[grid](
                dst_ptr=dst,
                dst_s0=dst.stride(0), dst_s1=dst.stride(1), dst_s2=dst.stride(2),
                src_ptr=src[i:i+chunk_size],
                src_s0=src.stride(0), src_s1=src.stride(1), src_s2=src.stride(2),
                idx_ptr=indices[i:i+chunk_size],
                BLOCK_SIZE=B,
                TILE_M=16,
                TILE_N=16,
            )


def main():
    num_blocks = 4096000
    block_size = 64
    num_indices = 600000
    src_data = torch.randn(num_indices, block_size, block_size, device="cuda", dtype=torch.bfloat16)
    data_bloks = torch.empty(num_blocks, block_size, block_size, device="cuda", dtype=torch.bfloat16)
    # draw num_indices random different integers in the range [0, num_blocks)
    pool_indices = torch.randperm(num_blocks, device="cuda")[:num_indices]

    for _ in range(10):
        data_bloks[pool_indices] = src_data

    chunk_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    for chunk_n in chunk_sizes:
        torch.cuda.synchronize()
        for _ in range(10):
            chunked_scatter(data_bloks, pool_indices, src_data, chunk_n=chunk_n)


if __name__ == "__main__":
    main()
