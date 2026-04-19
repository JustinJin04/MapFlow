import torch
from mapflow.core import prof_marker

@torch.no_grad()
def weight_col_filter_top_p(q, k, weights, block_size, top_p=1.0):
    """
    q: [num_tokens_q, num_heads, head_dim]
    k: [num_tokens_kv, num_kv_heads, head_dim]
    weights: [num_heads, num_recv_heads]
    
    returns:
     - crow_indices
     - col_indices
     - values: [nnz, block_size, block_size]
     - placement: [L, H] in 1-dim (flattened)
      L0H0, L0H1, ..., L1H0, L1H1, ... 
    """
    # print(f"q.shape: {q.shape}, k.shape: {k.shape}")
    num_tokens_q, num_heads, head_dim = q.shape
    num_tokens_kv, num_kv_heads, _ = k.shape
    _, num_recv_heads = weights.shape
    assert num_tokens_q % block_size == 0
    assert num_tokens_kv % block_size == 0

    group_size = num_heads // num_kv_heads
    num_blocks_q = num_tokens_q // block_size
    num_blocks_kv = num_tokens_kv // block_size
    scale = head_dim ** 0.5

    q = q.view(num_blocks_q, block_size, num_heads, head_dim)
    k = k.view(num_blocks_kv, block_size, num_kv_heads, head_dim)

    crow_indices_list = [0]
    # crow_indices_tensor = torch.zeros((1,), device=q.device, dtype=torch.long) # [0]
    col_indices_tensor = torch.tensor([], device=q.device, dtype=torch.long)
    values_list = []
    tril_mask = ~torch.tril(torch.ones((block_size, block_size), device=q.device)).bool()
    for q_block_idx in range(num_blocks_q):
        q_block = q[q_block_idx, :, :, :]  # [block_size, num_heads, head_dim]
        num_causal_blocks_kv = num_blocks_kv - num_blocks_q + q_block_idx + 1
        k_blocks_causal = k[:num_causal_blocks_kv, :, :, :] # [num_causal_blocks_kv, block_size, num_kv_heads, head_dim]
        k_blocks_causal = k_blocks_causal.view(num_causal_blocks_kv, block_size, num_kv_heads, 1, head_dim).expand(-1, -1, -1, group_size, -1).reshape(num_causal_blocks_kv, block_size, num_heads, head_dim)  # [num_causal_blocks_kv, block_size, num_heads, head_dim]

        q_block = q_block.permute(1, 0, 2)  # [num_heads, block_size, head_dim]
        k_blocks_causal = k_blocks_causal.reshape(-1, num_heads, head_dim).permute(1, 2, 0) # [num_heads, head_dim, num_causal_blocks_kv * block_size]
        # print(f"dtype of q_block: {q_block.dtype}, dtype of k_blocks_causal: {k_blocks_causal.dtype}")
        attn_scores = torch.matmul(q_block, k_blocks_causal) / scale  # [num_heads, block_size, num_causal_blocks_kv * block_size]
        attn_scores = attn_scores.view(num_heads, block_size, num_causal_blocks_kv, block_size)
        attn_scores[:, :, -1, :].masked_fill_(tril_mask, float("-inf"))
        attn_scores = attn_scores.view(num_heads, block_size, num_causal_blocks_kv * block_size)
        attn_scores = torch.softmax(attn_scores, dim=-1)  # [num_heads, block_size, num_causal_blocks_kv * block_size]
        
        # transform with weights
        attn_scores = attn_scores.view(num_heads, -1).T @ weights # [-1, num_recv_heads]
        attn_scores = attn_scores.T.reshape(num_recv_heads, block_size, num_causal_blocks_kv, block_size).permute(0, 2, 1, 3).contiguous()  # [num_recv_heads, num_causal_blocks_kv, block_size, block_size]
        
        block_sum = attn_scores.sum(dim=(-1, -2))  # [num_recv_heads, num_causal_blocks_kv], sum(dim=-1) = block_size for each head
        sorted_scores, sorted_indices = torch.sort(block_sum, dim=-1, descending=True)  # [num_recv_heads, num_causal_blocks_kv]
        cumsum = torch.cumsum(sorted_scores, dim=-1)  # [num_recv_heads, num_causal_blocks_kv]
        first_col_true = torch.ones_like(cumsum[..., :1], device=q.device, dtype=torch.bool)  # [num_recv_heads, 1]
        shifted_sorted_mask = torch.cat([first_col_true, (cumsum <= top_p * block_size)[..., :-1]], dim=-1)  # [num_recv_heads, num_causal_blocks_kv]
        top_p_mask = torch.zeros_like(shifted_sorted_mask, device=q.device, dtype=torch.bool)  # [num_recv_heads, num_causal_blocks_kv]
        top_p_mask.scatter_(dim=-1, index=sorted_indices, src=shifted_sorted_mask)  # [num_recv_heads, num_causal_blocks_kv]

        attn_scores_permute = attn_scores.permute(1, 0, 2, 3).contiguous()  # [num_causal_blocks_kv, num_recv_heads, block_size, block_size]
        top_p_mask_permute = top_p_mask.permute(1, 0).contiguous()  # [num_causal_blocks_kv, num_recv_heads]
        nnz_rows = top_p_mask_permute.sum()
        (nz_col_indices,) = torch.nonzero(top_p_mask_permute.view(-1), as_tuple=True)
        # crow_indices_tensor = torch.cat([crow_indices_tensor, crow_indices_tensor[-1:] + nnz_rows], dim=0)
        crow_indices_list.append(crow_indices_list[-1] + nz_col_indices.shape[0])
        col_indices_tensor = torch.cat([col_indices_tensor, nz_col_indices], dim=0)
        values_list.append(
            attn_scores_permute[top_p_mask_permute]  # [nnz, block_size, block_size]
        )

    values_tensor = torch.cat(values_list, dim=0)  # [nnz, block_size, block_size]
    return crow_indices_list, col_indices_tensor, values_tensor

