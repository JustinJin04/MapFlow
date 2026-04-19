from __future__ import annotations

from typing import Optional
import torch


class BlockPool:
    """
    Block allocator allowing non-contiguous allocation over a fixed torch tensor:
        data_blocks: [max_num_blocks, block_size, block_size]

    malloc(n): returns a tensor of n block indices (not necessarily contiguous)
    free(indices): frees the blocks at the specified indices

    Internals:
      - free_indices: A 1D tensor acting as a stack of available block indices.
      - num_free: Integer pointer indicating the current top of the stack (number of free blocks).
    """

    def __init__(
        self,
        max_num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        if max_num_blocks <= 0:
            raise ValueError(f"max_num_blocks must be > 0, got {max_num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        self.max_num_blocks = int(max_num_blocks)
        self.block_size = int(block_size)
        self.dtype = dtype
        self.device = device


        # Stack of free indices. Initially, all indices [0, max_num_blocks-1] are free.
        # We perform stack operations by manipulating the 'num_free' pointer.
        # Free indices are stored in self.free_indices[:self.num_free]
        self.free_indices = torch.arange(
            self.max_num_blocks, dtype=torch.int32, device=self.device
        )
        self.num_free = self.max_num_blocks

    def malloc(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate num_blocks (non-contiguous allowed).
        
        Returns:
            torch.Tensor: A 1D LongTensor of shape [num_blocks] containing the allocated indices.
        
        Raises:
            MemoryError: If there are not enough free blocks.
        """
        n = int(num_blocks)
        if n <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")

        if self.num_free < n:
            raise MemoryError(
                f"DataBlocks OOM: requested {n} blocks, but only {self.num_free} are free."
            )

        # Pop n indices from the end of the free list
        start_idx = self.num_free - n
        allocated_indices = self.free_indices[start_idx : self.num_free].clone()
        
        # Update the pointer
        self.num_free -= n

        return allocated_indices

    def free(self, indices: torch.Tensor | list) -> None:
        """
        Free the blocks at the specified indices.
        
        Args:
            indices: A 1D tensor of block indices to free.
        """
        if not isinstance(indices, torch.Tensor):
            # Handle list inputs just in case, though tensor is preferred
            indices = torch.tensor(indices, dtype=torch.int32, device=self.device)
        
        n = indices.numel()
        if n == 0:
            return

        if self.num_free + n > self.max_num_blocks:
            raise ValueError(
                f"Double free detected or invalid indices: attempting to free {n} blocks, "
                f"but capacity is {self.max_num_blocks} and current free is {self.num_free}."
            )

        # Push indices back onto the free stack
        self.free_indices[self.num_free : self.num_free + n] = indices
        self.num_free += n

    def used_capacity(self) -> int:
        """Total number of used blocks."""
        return self.max_num_blocks - self.num_free

    def free_capacity(self) -> int:
        """Total number of free blocks."""
        return self.num_free

    def total_capacity(self) -> int:
        """Total capacity of the pool."""
        return self.max_num_blocks