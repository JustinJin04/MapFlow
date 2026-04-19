from dataclasses import dataclass
from collections import OrderedDict
import os
import numpy as np
import torch
import torch.multiprocessing as mp

from mapflow.server.memory import BlockPool
from mapflow.core import prof_marker
from mapflow.core.mp_ctx_queue import BiDirQueue, TensorBufferMeta
from mapflow.kernel.retrieve import retrieve_triton

mp_ctx = mp.get_context('spawn')

@dataclass
class BlockMetadataAllLayers:
    col_indices: torch.Tensor
    block_indices: torch.Tensor
    layer_list: list[int] # [num_layers]
    nnz_blocks_cpu: np.ndarray   # shape [L], dtype=int32
    pin_count: int = 0

@dataclass
class HitMetas:
    hit_col_tensor: torch.Tensor | None
    hit_block_tensor: torch.Tensor | None
    req_crow_tensor: torch.Tensor | None # [L, H+1]
    layer_cu_offsets: list[int]  # [num_layers + 1]

    def to_buffer_meta(self, queue: BiDirQueue) -> dict[str, TensorBufferMeta|list[int]]:
        if self.hit_col_tensor is not None:
            return {
                "hit_col_tensor": queue.wrap_resp_tensor(self.hit_col_tensor),
                "hit_block_tensor": queue.wrap_resp_tensor(self.hit_block_tensor),
                "req_crow_tensor": queue.wrap_resp_tensor(self.req_crow_tensor),
                "layer_cu_offsets": self.layer_cu_offsets
            }
        else:
            return {
                "hit_col_tensor": None,
                "hit_block_tensor": None,
                "req_crow_tensor": None,
                "layer_cu_offsets": self.layer_cu_offsets
            }
    
    @staticmethod
    def from_buffer_meta(buffer_meta_dict: dict[str, TensorBufferMeta|None|list[int]], queue: BiDirQueue) -> "HitMetas":
        if buffer_meta_dict["hit_col_tensor"] is not None:
            hit_col_tensor = queue.read_resp_tensor(buffer_meta_dict["hit_col_tensor"])
            hit_block_tensor = queue.read_resp_tensor(buffer_meta_dict["hit_block_tensor"])
            req_crow_tensor = queue.read_resp_tensor(buffer_meta_dict["req_crow_tensor"])
            layer_cu_offsets = buffer_meta_dict["layer_cu_offsets"]
            return HitMetas(
                hit_col_tensor=hit_col_tensor,
                hit_block_tensor=hit_block_tensor,
                req_crow_tensor=req_crow_tensor,
                layer_cu_offsets=layer_cu_offsets
            )
        else:
            return HitMetas(
                hit_col_tensor=None,
                hit_block_tensor=None,
                req_crow_tensor=None,
                layer_cu_offsets=buffer_meta_dict["layer_cu_offsets"]
            )


class RetrieveAwareLRUAdmission:
    def __init__(self, miss_blacklist_capacity: int | None = 10000):
        self.cold_lru: OrderedDict[str, None] = OrderedDict()
        self.retrieved_lru: OrderedDict[str, None] = OrderedDict()

        self.miss_blacklist: OrderedDict[str, None] = OrderedDict()
        self.miss_blacklist_capacity = miss_blacklist_capacity

    def cache_size(self) -> int:
        return len(self.cold_lru) + len(self.retrieved_lru)

    def contains(self, hash_str: str) -> bool:
        return hash_str in self.cold_lru or hash_str in self.retrieved_lru

    def should_store(self, hash_str: str) -> bool:
        return hash_str not in self.miss_blacklist

    def on_store(self, hash_str: str):
        if self.contains(hash_str):
            return
        self.cold_lru[hash_str] = None

    def on_retrieve_hit(self, hash_str: str):
        if hash_str in self.cold_lru:
            self.cold_lru.pop(hash_str, None)
            self.retrieved_lru[hash_str] = None
            return

        if hash_str in self.retrieved_lru:
            self.retrieved_lru.move_to_end(hash_str)
            return

        raise KeyError(f"{hash_str} not found in any LRU queue")

    def on_retrieve_miss(self, hash_str: str):
        self.miss_blacklist[hash_str] = None
        self.miss_blacklist.move_to_end(hash_str)
        self._trim_blacklist_if_needed()

    def _trim_blacklist_if_needed(self):
        if self.miss_blacklist_capacity is None:
            return
        while len(self.miss_blacklist) > self.miss_blacklist_capacity:
            self.miss_blacklist.popitem(last=False)

    def remove(self, hash_str: str):
        self.cold_lru.pop(hash_str, None)
        self.retrieved_lru.pop(hash_str, None)

    def pop_evict_candidate(self) -> tuple[str, str] | None:
        if self.retrieved_lru:
            hash_str, _ = self.retrieved_lru.popitem(last=False)
            return hash_str, "retrieved"

        if self.cold_lru:
            hash_str, _ = self.cold_lru.popitem(last=False)
            return hash_str, "cold"

        return None

    def defer_evict_candidate(self, hash_str: str, tier: str):
        if tier == "retrieved":
            self.retrieved_lru[hash_str] = None
        elif tier == "cold":
            self.cold_lru[hash_str] = None
        else:
            raise ValueError(f"Unknown tier: {tier}")


class FIFOCachePolicy:
    """
    最基本的 cache policy：
    - 不区分 cold / retrieved
    - 不做 retrieve-aware
    - 不做 miss blacklist / admission control
    - 按进入 cache 的先后顺序做 FIFO eviction

    为了兼容现有 BlockManagerWorker：
    - 保留 cold_lru / retrieved_lru 两个属性
    - 实际只使用 cold_lru 作为 FIFO 队列
    - retrieved_lru 始终为空
    """

    def __init__(self, miss_blacklist_capacity: int | None = 1000):
        # 兼容现有 worker 的直接访问
        self.cold_lru: OrderedDict[str, None] = OrderedDict()
        self.retrieved_lru: OrderedDict[str, None] = OrderedDict()

        # 保留这个参数只是为了兼容现有构造签名，实际不用
        self.miss_blacklist_capacity = miss_blacklist_capacity

    def cache_size(self) -> int:
        return len(self.cold_lru) + len(self.retrieved_lru)

    def contains(self, hash_str: str) -> bool:
        return hash_str in self.cold_lru or hash_str in self.retrieved_lru

    def should_store(self, hash_str: str) -> bool:
        # 最基本版本：不过滤，永远允许写入
        return True

    def on_store(self, hash_str: str):
        if self.contains(hash_str):
            return
        # FIFO：新元素进入队尾
        self.cold_lru[hash_str] = None

    def on_retrieve_hit(self, hash_str: str):
        # FIFO 不因为 hit 改变顺序
        if self.contains(hash_str):
            return
        raise KeyError(f"{hash_str} not found in FIFO queue")

    def on_retrieve_miss(self, hash_str: str):
        # 非 retrieve-aware：miss 不影响 admission / eviction
        return

    def remove(self, hash_str: str):
        self.cold_lru.pop(hash_str, None)
        self.retrieved_lru.pop(hash_str, None)

    def pop_evict_candidate(self) -> tuple[str, str] | None:
        # 兼容旧接口，tier 仍返回 "cold"
        if self.cold_lru:
            hash_str, _ = self.cold_lru.popitem(last=False)
            return hash_str, "cold"
        return None

    def defer_evict_candidate(self, hash_str: str, tier: str):
        # 兼容旧接口：无论传入什么 tier，都放回 FIFO 队尾
        if tier not in ("cold", "retrieved", "fifo"):
            raise ValueError(f"Unknown tier: {tier}")
        self.cold_lru[hash_str] = None


class BlockManagerWorker(mp_ctx.Process):
    def __init__(
        self,
        store_queue: BiDirQueue,
        retrieve_queue_list: list[BiDirQueue],
        layer_list: list[int],
        block_size: int,
        dtype: torch.dtype,
        max_num_blocks: int,
        device: str,
    ):
        """
        store_queue: {
            "items": dict[str, dict[int, torch.Tensor]]
        }, {
            "metas": dict[str, LayerBlockMetadata | None] (None means already stored and no need to store again)
        }, {
            "meta_list": list[LayerBlockMetadata] (proxy copy has done, unpin all the metas)
        }
        retrieve_queue: {
            "hash_strs": list[str],
        }, {
            "hit_list": list[bool],
            "meta_list": list[LayerBlockMetadata]
        }, {
            "meta_list": list[LayerBlockMetadata] (proxy copy has done, unpin all the metas)
        }
        """
        super().__init__(daemon=True)
        self.store_queue = store_queue
        self.retrieve_queue_list = retrieve_queue_list
        self.layer_list = layer_list

        # Initialize block pool to manage free/used blocks (malloc or free)
        self.block_pool = BlockPool(
            max_num_blocks=max_num_blocks,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        # Metadata management
        self.hash_to_layer_blockmetadata: dict[str, BlockMetadataAllLayers] = {}
        self.pending_layer_blockmetadata: dict[str, BlockMetadataAllLayers] = {}
        # self.lru_order: OrderedDict[str, None] = OrderedDict()

        if os.environ.get("BLOCK_CACHE_POLICY", "") == "FIFO":
            print(f"Using FIFO cache policy", flush=True)
            self.cache_policy = FIFOCachePolicy()
        else:
            print(f"Using RetrieveAwareLRUAdmission cache policy", flush=True)
            self.cache_policy = RetrieveAwareLRUAdmission()
        
        # Pre-allocate a zero tensor for efficient concatenation in _retrieve_pin
        self.zero_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        self.zero_tensor_layers = torch.zeros(len(layer_list), dtype=torch.int32, device=device).unsqueeze(1)

    def run(self):
        while True:
            # Check store queue
            if not self.store_queue.req_empty():
                store_req = self.store_queue.get_req()
                if "items" in store_req:
                    items = store_req["items"]
                    block_indices = self._store_pending(items)
                    self.store_queue.put_resp({"block_indices": block_indices})
                    # sync with cpu to make sure all the store operations are done before proceed to retrieve
                    # torch.cuda.current_stream().synchronize()
                elif "ack_hash_strs" in store_req:
                    hash_strs = store_req["ack_hash_strs"]
                    self._ack_pending(hash_strs)
                else:
                    assert 0, f"Unknown store_req: {store_req}"

            # Check retrieve queues
            for retrieve_queue in self.retrieve_queue_list:
                # while not retrieve_queue.req_empty():
                if not retrieve_queue.req_empty():
                    retrieve_req = retrieve_queue.get_req()
                    if "hash_strs" in retrieve_req:
                        num_retrieve_layers = retrieve_req["num_retrieve_layers"]
                        hash_strs = retrieve_req["hash_strs"]
                        self._retrieve_pin(num_retrieve_layers, hash_strs, retrieve_queue)
                    elif "unpin_hash_strs" in retrieve_req:
                        hash_strs = retrieve_req["unpin_hash_strs"]
                        self._unpin(hash_strs)
                    else:
                        assert 0, f"Unknown retrieve_req: {retrieve_req}"

    @prof_marker(f"_store_pending")
    def _store_pending(
        self,
        items: dict[str, dict[str, TensorBufferMeta | list[int]]]
    )-> dict[str, TensorBufferMeta]:
        """
        items: hash_str -> {
            "col_indices": TensorBufferMeta
            "nnz_blocks_list": [num_layers]
            "nnz_blocks_tensor": TensorBufferMeta (num_layers)
        }
        """

        block_indices = {}
        skipped_hash_strs: list[str] = []

        for hash_str, metas_dict in items.items():
            if hash_str in self.hash_to_layer_blockmetadata or hash_str in self.pending_layer_blockmetadata:
                continue

            if not self.cache_policy.should_store(hash_str):
                skipped_hash_strs.append(hash_str)
                continue

            col_indices = self.store_queue.read_req_tensor(metas_dict["col_indices"]).clone()

            # Only for debug
            # if os.environ.get("BLOCK_MANAGER_DEBUG") == "1":
            #     assert col_indices.dtype == torch.int32, f"col_indices dtype should be int32, but got {col_indices.dtype}"
            #     assert col_indices.max().item() < 2560 // 4 * 40, f"col_indices max value {col_indices.max().item()}, col_indices: {col_indices.tolist()}"

            nnz_blocks_cpu = np.asarray(metas_dict["nnz_blocks_list"], dtype=np.int32)

            nnz_blocks = col_indices.shape[0]
            if self.block_pool.free_capacity() < nnz_blocks:
                self._evict_until_available(nnz_blocks)

            indices = self.block_pool.malloc(num_blocks=nnz_blocks)

            layer_blockmetadata = BlockMetadataAllLayers(
                col_indices=col_indices,
                block_indices=indices,
                layer_list=self.layer_list,
                nnz_blocks_cpu=nnz_blocks_cpu,
                pin_count=0
            )
            # print(f"hash_str: {hash_str}, layer_blockmetadata: {layer_blockmetadata}")
            # self.hash_to_layer_blockmetadata[hash_str] = layer_blockmetadata
            # self.cache_policy.on_store(hash_str)
            self.pending_layer_blockmetadata[hash_str] = layer_blockmetadata

            block_indices[hash_str] = self.store_queue.wrap_resp_tensor(indices)

        torch.cuda.current_stream().synchronize()
        # print(f"skipped {len(skipped_hash_strs)} items. ")
        print(
            f"Stored {len(block_indices)} out of {len(items)}, "
            f"skipped {len(skipped_hash_strs)} items."
        )
        return block_indices

    @prof_marker(f"_retrieve_triton")
    def _retrieve_pin(
        self,
        num_retrieve_layers: int,
        hash_strs: list[str],
        retrieve_queue: BiDirQueue
    ) -> tuple[list[bool], dict[str, TensorBufferMeta | list[int]]]:
        if len(hash_strs) == 0:
            return [], []

        assert num_retrieve_layers <= len(self.layer_list), f"num_retrieve_layers={num_retrieve_layers}"

        num_hits = 0
        col_indices_list: list[torch.Tensor] = []
        block_indices_list: list[torch.Tensor] = []
        nnz_blocks_cpu_list: list[np.ndarray] = []

        for i, hash_str in enumerate(hash_strs):
            layer_blockmetadata = self.hash_to_layer_blockmetadata.get(hash_str)
            if layer_blockmetadata is None:
                for j in range(i, len(hash_strs)):
                    self.cache_policy.on_retrieve_miss(hash_strs[j])
                break

            num_hits += 1
            self.cache_policy.on_retrieve_hit(hash_str)
            layer_blockmetadata.pin_count += 1

            nnz_blocks_cpu = layer_blockmetadata.nnz_blocks_cpu[-num_retrieve_layers:]
            nnz = nnz_blocks_cpu.sum()
            col_indices_list.append(layer_blockmetadata.col_indices[-nnz:])
            block_indices_list.append(layer_blockmetadata.block_indices[-nnz:])
            nnz_blocks_cpu_list.append(nnz_blocks_cpu)

        prefix_hit_list = [True] * num_hits + [False] * (len(hash_strs) - num_hits)
        print(f"BlockManager: Retrieved {sum(prefix_hit_list)}/{len(hash_strs)} prefix items.")
        # print(f"retrieve hashes: {hash_strs}")
        retrieve_queue.put_resp({
            "hit_list": prefix_hit_list,
        })

        if num_hits == 0:
            hit_metas = HitMetas(
                hit_col_tensor=None,
                hit_block_tensor=None,
                req_crow_tensor=None,
                layer_cu_offsets=[0] * (num_retrieve_layers + 1)
            ).to_buffer_meta(retrieve_queue)

            retrieve_queue.put_resp_1({
                "hit_metas": hit_metas,
            })
            return

        # L = len(self.layer_list)
        L = num_retrieve_layers
        H = num_hits
        src_col_indices_tensor = torch.cat(col_indices_list)  # [total_nnz]
        src_block_indices_tensor = torch.cat(block_indices_list)  # [total_nnz]
        cpu_nnz_blocks = np.stack(nnz_blocks_cpu_list)  # [H, L], int32
        hit_nnz_tensor = torch.from_numpy(cpu_nnz_blocks).pin_memory().to("cuda", non_blocking=True)
        hit_nnz_transposed = hit_nnz_tensor.T.contiguous()  # [L, H]
        src_off = torch.cat([self.zero_tensor, torch.cumsum(hit_nnz_tensor.view(-1), dim=0)])[:-1].view(H, L)  # [H, L]
        dst_off = torch.cat([self.zero_tensor, torch.cumsum(hit_nnz_transposed.view(-1), dim=0)])[:-1].view(L, H).T.contiguous()  # [H, L]
        hit_col_indices = retrieve_triton(
            hit_nnz=hit_nnz_tensor,
            src_off=src_off,
            dst_off=dst_off,
            src_tensor=src_col_indices_tensor,
        )
        hit_block_indices = retrieve_triton(
            hit_nnz=hit_nnz_tensor,
            src_off=src_off,
            dst_off=dst_off,
            src_tensor=src_block_indices_tensor,
        )
        layer_cu_offsets_list = np.concatenate(
            [np.zeros((1,), dtype=np.int32), np.cumsum(cpu_nnz_blocks.transpose().sum(axis=1))]
        ).tolist()

        hit_metas = HitMetas(
            hit_col_tensor=hit_col_indices,
            hit_block_tensor=hit_block_indices,
            req_crow_tensor=torch.cat(
                [self.zero_tensor_layers[:L], torch.cumsum(hit_nnz_transposed, dim=1)], dim=1
            ).view(L, H + 1).to(torch.int32),
            layer_cu_offsets=layer_cu_offsets_list
        ).to_buffer_meta(retrieve_queue)

        torch.cuda.current_stream().synchronize()
        retrieve_queue.put_resp_1({
            "hit_metas": hit_metas,
        })

    def _ack_pending(
        self,
        hash_strs: list[str]
    ):
        for hash_str in hash_strs:
            assert hash_str in self.pending_layer_blockmetadata, f"{hash_str} not in pending metadata"
            layer_blockmetadata = self.pending_layer_blockmetadata.pop(hash_str)
            self.hash_to_layer_blockmetadata[hash_str] = layer_blockmetadata
            self.cache_policy.on_store(hash_str)
            assert layer_blockmetadata.pin_count == 0
        print(
            f"BlockManager: Stored {len(hash_strs)} items, "
            f"Current cache size: {self.cache_policy.cache_size()} items. "
            f"Block pool usage: {self.block_pool.used_capacity()}/{self.block_pool.total_capacity()} blocks."
        )

    def _unpin(
        self,
        hash_strs: list[str]
    ):
        for hash_str in hash_strs:
            assert hash_str in self.hash_to_layer_blockmetadata, f"{hash_str} not in {self.hash_to_layer_blockmetadata.keys()}"
            layer_blockmetadata = self.hash_to_layer_blockmetadata[hash_str]
            assert layer_blockmetadata.pin_count > 0, f"Unpinning error: pin count for {hash_str} is already 0."
            layer_blockmetadata.pin_count -= 1

    def _try_evict_one_from_queue(self, queue: OrderedDict[str, None]) -> bool:
        checked = 0
        initial_size = len(queue)

        while queue and checked < initial_size:
            hash_str, _ = queue.popitem(last=False)
            meta = self.hash_to_layer_blockmetadata.get(hash_str)

            if meta is None:
                # 清理潜在不一致状态
                self.cache_policy.remove(hash_str)
                checked += 1
                continue

            if meta.pin_count > 0:
                # 还在使用，放回队尾
                queue[hash_str] = None
                checked += 1
                continue

            # 真正执行 eviction
            self.block_pool.free(meta.block_indices)
            del self.hash_to_layer_blockmetadata[hash_str]
            self.cache_policy.remove(hash_str)  # 幂等清理，防止双队列残留
            return True

        return False

    @prof_marker(f"_evict_until_available")
    def _evict_until_available(self, required_blocks: int):
        while self.block_pool.free_capacity() < required_blocks:
            # 先尝试从 retrieved_lru 驱逐
            evicted = self._try_evict_one_from_queue(self.cache_policy.retrieved_lru)

            # 如果 retrieved_lru 这一轮全都 pinned / 不可驱逐，再尝试 cold_lru
            if not evicted:
                evicted = self._try_evict_one_from_queue(self.cache_policy.cold_lru)

            # 两边都驱逐不了，才是真的没空间了
            if not evicted:
                break

        if self.block_pool.free_capacity() < required_blocks:
            raise MemoryError(
                f"BlockManager OOM: Requested {required_blocks} blocks, "
                f"but only {self.block_pool.free_capacity()} available. "
                f"Cache contains {self.cache_policy.cache_size()} items "
                f"(others may be pinned)."
            )

class BlockManager:
    def __init__(
        self,
        layer_list: list[int],
        block_size: int,
        dtype: torch.dtype,
        tp_size: int,
        max_num_blocks: int,
        max_num_recvs: int = 2,
        device: str = "cuda",
    ):
        self.store_queue = BiDirQueue(mp_ctx=mp_ctx, device=device)
        self.retrieve_queue_list = [BiDirQueue(mp_ctx=mp_ctx, device=device) for _ in range(max_num_recvs)]

        # Shared between send worker and recv worker
        self.data_blocks = torch.zeros(
            (max_num_blocks, block_size, block_size),
            dtype=dtype,
            device=f"cuda:{tp_size}",
            # pin_memory=True
        )

        self.worker_process = BlockManagerWorker(
            store_queue=self.store_queue,
            retrieve_queue_list=self.retrieve_queue_list,
            layer_list=layer_list,
            block_size=block_size,
            dtype=dtype,
            max_num_blocks=max_num_blocks,
            device=device,
        )
        self.worker_process.start()


class BlockManagerProxy:
    def __init__(
        self,
        queue: BiDirQueue, # store or retrieve queue
    ):
        self.queue = queue

    @prof_marker("prefix_retrieve_hit_list")
    def prefix_retrieve_hit_list(
        self,
        num_retrieve_layers: int,
        hash_strs: list[str]
    ) -> list[bool]:
        self.queue.put_req({
            "num_retrieve_layers": num_retrieve_layers,
            "hash_strs": hash_strs
        })
        with prof_marker("prefix_retrieve_wait_for_hit_list"):
            resp = self.queue.get_resp()
        hit_list = resp["hit_list"]
        return hit_list

    @prof_marker("prefix_retrieve_hit_metas")
    def prefix_retrieve_hit_metas(
        self,
    ) -> HitMetas:
        with prof_marker("prefix_retrieve_wait_for_hit_metas"):
            resp = self.queue.get_resp_1()
        hit_metas = HitMetas.from_buffer_meta(resp["hit_metas"], self.queue)
        # hit_metas = HitMetas.from_list(resp["hit_metas"])
        return hit_metas

    # def retrieve_blocks(
    #     self,
    #     block_indices: torch.Tensor,
    #     dst_tensor: torch.Tensor
    # ):
    #     assert dst_tensor.shape[0] == block_indices.shape[0], f"dst_tensor.shape: {dst_tensor.shape}, block_indices.shape: {block_indices.shape}"
    #     print(f"num_blocks: {block_indices.shape[0]}", flush=True)
    #     with prof_marker(f"{self.data_blocks.device} retrieve"):
    #         blocks = self.data_blocks[block_indices]  # [num_blocks, block_size, block_size]
    #     with prof_marker(f"{self.data_blocks.device} to {dst_tensor.device}"):
    #         dst_tensor.copy_(blocks, non_blocking=True)

    def retrieve_unpin(
        self,
        hash_strs: list[str]
    ):
        self.queue.put_req({
            "unpin_hash_strs": hash_strs
        })
