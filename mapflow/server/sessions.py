import os
from dataclasses import dataclass, field
import traceback
import numpy as np
import torch
import torch.multiprocessing as mp
from vllm.v1.core.sched.output import SchedulerOutput
from mapflow.batch_states import BatchState
from mapflow.core.zmq import ZMQCommunicator
from mapflow.core.mp_ctx_queue import BiDirQueue, TensorBufferMeta
from mapflow.core.prof_marker import prof_marker
from mapflow.server.block_manager import BlockManagerProxy
from mapflow.kernel.scatter import chunked_scatter

mp_ctx = mp.get_context('spawn')

def cat_col_among_tp(col_dict, num_heads, tp_size, return_invperm=False):
    parts = []
    H = num_heads
    row_stride_delta = (tp_size - 1) * H

    for tp_rank in range(tp_size):
        col_tensor = col_dict.get(tp_rank)
        if col_tensor is None:
            continue
        if not (0 <= tp_rank < tp_size):
            raise RuntimeError(f"tp_rank out of range: tp_rank={tp_rank}, tp_size={tp_size}")
        if col_tensor.numel() == 0:
            continue

        # old = row * H + col
        # new = old + row * ((tp_size - 1) * H) + tp_rank * H
        row_idx = torch.div(col_tensor, H, rounding_mode="floor")
        remapped = col_tensor + row_idx * row_stride_delta + tp_rank * H
        parts.append(remapped)

    assert len(parts) > 0, "No column indices found across TP ranks"

    # 当前拼接顺序天然是 [tp, block, head]
    tp_major = torch.cat(parts)

    # 发送给 block manager 的顺序需要是 [block, tp, head]
    sort_idx = torch.argsort(tp_major)
    sorted_cols = tp_major[sort_idx].to(torch.int32)

    if not return_invperm:
        return sorted_cols

    # block manager 返回后，用 invperm 直接恢复为 [tp, block, head]
    invperm = torch.empty_like(sort_idx)
    invperm[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
    return sorted_cols, invperm

def permute_block_indices(
    block_indices: torch.Tensor,
    col_indices: torch.Tensor,
    nnz_blocks_list: list[int],
    block_pos: int,
    tp_size: int,
    num_heads: int,
):
    num_blocks = block_pos + 1
    old_row_stride = tp_size * num_heads
    new_row_stride = num_blocks * num_heads

    ret = []
    cur_off = 0
    for nnz in nnz_blocks_list:
        vals = block_indices[cur_off:cur_off + nnz]
        cols = col_indices[cur_off:cur_off + nnz]

        b = torch.div(cols, old_row_stride, rounding_mode="floor")
        rem = cols - b * old_row_stride
        t = torch.div(rem, num_heads, rounding_mode="floor")
        h = rem - t * num_heads

        mapped = t * new_row_stride + b * num_heads + h
        order = torch.argsort(mapped)
        ret.append(vals[order])

        cur_off += nnz

    return torch.cat(ret).to(torch.int32) if ret else block_indices.new_empty((0,))

@dataclass
class RoundStoreState:
    round_id: int
    buf_idx: int | None = None
    hashes_list: list[list[str]] | None = None
    raw_data: dict[int, dict[int, list | None]] = field(default_factory=dict)
    events: list[torch.cuda.Event] = field(default_factory=list)

class StoreItems:
    def __init__(
        self,
        layer_list: list[int],
        tp_size: int,
        num_heads: int, # per tp worker
        data_blocks: torch.Tensor,
        crow_indices_remote_buffers: dict[int, list[torch.Tensor]],
        col_indices_remote_buffers: dict[int, list[torch.Tensor]],
        blocks_remote_buffers: dict[int, list[torch.Tensor]],
    ):
        self.layer_list = layer_list
        self.tp_size = tp_size
        self.num_heads = num_heads
        self.data_blocks = data_blocks
        self.crow_indices_remote_buffers = crow_indices_remote_buffers
        self.col_indices_remote_buffers = col_indices_remote_buffers
        self.blocks_remote_buffers = blocks_remote_buffers

        self.crow_cpu_pinned = torch.zeros((640,), dtype=torch.int32, device="cpu", pin_memory=True)

        self.permuted_indices_cat_buffer_pinned = torch.empty((data_blocks.shape[0],), dtype=torch.int32, device="cpu", pin_memory=True)
        block_size = data_blocks.shape[1]
        # self.block_tensor_gpu_buffer = torch.empty((600000, block_size, block_size), dtype=data_blocks.dtype, device=data_blocks.device)
        # self.block_tensor_gpu_buffer = torch.empty((100000, block_size, block_size), dtype=data_blocks.dtype, device=data_blocks.device)
        self.block_tensor_gpu_buffer = torch.empty((10000, block_size, block_size), dtype=data_blocks.dtype, device=data_blocks.device)


    @prof_marker("insert_block")
    def insert_block(self, round_state: RoundStoreState, queue: BiDirQueue):
        store_cols: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
        store_blocks: dict[str, dict[int, dict[int, torch.Tensor]]] = {}

        current_hashes_list = round_state.hashes_list
        buf_idx = round_state.buf_idx

        with prof_marker("prepare_store_items"):
            for layer_idx in self.layer_list:
                for tp_rank in range(self.tp_size):
                    data = round_state.raw_data[layer_idx][tp_rank]
                    assert len(current_hashes_list) == len(data), (
                        f"len(current_hashes_list): {len(current_hashes_list)}, len(data): {len(data)}"
                    )

                    for hashes, crow_col_block in zip(current_hashes_list, data):
                        num_blocks_k = crow_col_block["num_blocks_k"]
                        crow_start, crow_end = crow_col_block["crow"]
                        col_start, col_end = crow_col_block["col"]
                        block_start, block_end = crow_col_block["block"]

                        assert len(hashes) == crow_end - crow_start - 1, (
                            f"len(hashes): {len(hashes)}, crow_end: {crow_end}, crow_start: {crow_start}"
                        )

                        crow_tensor_gpu = self.crow_indices_remote_buffers[tp_rank][buf_idx][crow_start:crow_end]
                        col_tensor_cpu = self.col_indices_remote_buffers[tp_rank][buf_idx][col_start:col_end].to(
                            "cpu", non_blocking=True
                        )
                        block_tensor_gpu = self.blocks_remote_buffers[tp_rank][buf_idx][block_start:block_end]

                        assert crow_tensor_gpu.shape[0] <= self.crow_cpu_pinned.shape[0], (
                            f"crow_tensor_gpu.shape: {crow_tensor_gpu.shape[0]}"
                        )

                        self.crow_cpu_pinned[:crow_tensor_gpu.shape[0]].copy_(crow_tensor_gpu, non_blocking=True)
                        torch.cuda.current_stream(f"cuda:{tp_rank}").synchronize()

                        for i, hash_str in enumerate(hashes):
                            start_idx = self.crow_cpu_pinned[i].item()
                            end_idx = self.crow_cpu_pinned[i + 1].item()

                            if hash_str not in store_cols:
                                store_cols[hash_str] = {}
                            if layer_idx not in store_cols[hash_str]:
                                store_cols[hash_str][layer_idx] = {}

                            store_cols[hash_str][layer_idx][tp_rank] = col_tensor_cpu[start_idx:end_idx]

                            if hash_str not in store_blocks:
                                store_blocks[hash_str] = {}
                            if layer_idx not in store_blocks[hash_str]:
                                store_blocks[hash_str][layer_idx] = {}

                            store_blocks[hash_str][layer_idx][tp_rank] = block_tensor_gpu[start_idx:end_idx]

        send_cols = {}
        hash_to_invperm: dict[str, torch.Tensor] = {}

        with prof_marker("cat_col_among_tp"):
            for hash_str, layer_to_cols in store_cols.items():
                col_indices_list = []
                invperm_list = []
                nnz_blocks_list = []
                base = 0

                for layer_idx in self.layer_list:
                    col_indices_layer, invperm_layer = cat_col_among_tp(
                        col_dict=layer_to_cols[layer_idx],
                        num_heads=self.num_heads,
                        tp_size=self.tp_size,
                        return_invperm=True,
                    )

                    col_indices_list.append(col_indices_layer)
                    invperm_list.append(invperm_layer + base)

                    nnz = col_indices_layer.numel()
                    nnz_blocks_list.append(nnz)
                    base += nnz

                col_indices = torch.cat(col_indices_list)
                invperm = torch.cat(invperm_list)

                send_cols[hash_str] = {
                    "col_indices": queue.wrap_req_tensor(col_indices),
                    "nnz_blocks_list": nnz_blocks_list,
                }
                hash_to_invperm[hash_str] = invperm

        if len(send_cols) == 0:
            return

        # sync across each gpu stream
        for tp_rank in range(self.tp_size):
            torch.cuda.current_stream(f"cuda:{tp_rank}").synchronize()

        queue.put_req({
            "items": send_cols
        })

        resp = queue.get_resp()
        block_indices_dict: dict[str, TensorBufferMeta] = resp["block_indices"]

        if len(block_indices_dict) == 0:
            return

        ordered_hashes = list(block_indices_dict.keys())

        with prof_marker(f"permute_block_indices_and_insert_{len(ordered_hashes)}"):
            with prof_marker("apply_cached_invperm"):
                write_off = 0

                for hash_str in ordered_hashes:
                    indices_tensor = queue.read_resp_tensor(block_indices_dict[hash_str])
                    invperm = hash_to_invperm[hash_str]

                    if invperm.device != indices_tensor.device:
                        invperm = invperm.to(indices_tensor.device, non_blocking=True)

                    # 直接用缓存的逆置换恢复成 [tp, block, head] 顺序
                    permuted = indices_tensor.index_select(0, invperm)
                    n = permuted.numel()

                    assert write_off + n <= self.permuted_indices_cat_buffer_pinned.shape[0], (
                        f"permuted_indices buffer overflow: write_off={write_off}, n={n}, "
                        f"buffer_size={self.permuted_indices_cat_buffer_pinned.shape[0]}"
                    )

                    self.permuted_indices_cat_buffer_pinned[write_off:write_off + n].copy_(
                        permuted, non_blocking=True
                    )
                    write_off += n

                permuted_indices = self.permuted_indices_cat_buffer_pinned[:write_off].to(
                    self.data_blocks.device, non_blocking=True
                )

            global_offset = 0
            buffer_offset = 0

            for hash_str in ordered_hashes:
                for layer_idx in self.layer_list:
                    for r in range(self.tp_size):
                        assert r in store_blocks[hash_str][layer_idx]
                        nnz = store_blocks[hash_str][layer_idx][r].shape[0]

                        if buffer_offset + nnz > self.block_tensor_gpu_buffer.shape[0]:
                            pool_indices = permuted_indices[global_offset:global_offset + buffer_offset]
                            chunked_scatter(
                                dst=self.data_blocks,
                                indices=pool_indices,
                                src=self.block_tensor_gpu_buffer[:buffer_offset],
                            )
                            global_offset += buffer_offset
                            buffer_offset = 0

                        assert nnz <= self.block_tensor_gpu_buffer.shape[0], (
                            f"Buffer overflow: buffer_offset {buffer_offset}, nnz {nnz}, "
                            f"buffer size {self.block_tensor_gpu_buffer.shape[0]}"
                        )

                        self.block_tensor_gpu_buffer[buffer_offset:buffer_offset + nnz].copy_(
                            store_blocks[hash_str][layer_idx][r],
                            non_blocking=True,
                        )
                        buffer_offset += nnz

            # final flush
            if buffer_offset > 0:
                pool_indices = permuted_indices[global_offset:global_offset + buffer_offset]
                chunked_scatter(
                    dst=self.data_blocks,
                    indices=pool_indices,
                    src=self.block_tensor_gpu_buffer[:buffer_offset],
                )
                global_offset += buffer_offset

        # sync across each gpu stream(including receiver's gpu)
        for tp_rank in range(self.tp_size):
            torch.cuda.current_stream(f"cuda:{tp_rank}").synchronize()
        torch.cuda.current_stream(self.data_blocks.device).synchronize()

        queue.put_req({
            "ack_hash_strs": ordered_hashes
        })


class SendSessionWorker(mp_ctx.Process):
    def __init__(
        self,
        send_layer_list: list[int],
        queue: BiDirQueue,
        data_blocks: torch.Tensor,
        tp_size: int,
        num_heads: int,
        sender_info_list: list[dict],
        port_queue: mp.Queue,
    ):
        """
        manage a large cpu tensor pool for blocks
        operate with all the send TP workers
        1. receive hashes of current iteration from one of sender
        2. for each layer, each TP worker send the address of col/blocks to session,
        and session decide where to store and tell block manager the hashes->(col, block_indices)
        """
        super().__init__(daemon=True)
        self.send_layer_list = send_layer_list
        self.queue = queue
        self.data_blocks = data_blocks
        self.tp_size = tp_size
        self.num_heads = num_heads
        self.sender_info_list = sender_info_list
        self.port_queue = port_queue
    
    def _new_round_state(self, round_id: int) -> RoundStoreState:
        return RoundStoreState(
            round_id=round_id,
            raw_data={
                layer_idx: {r: None for r in range(self.tp_size)}
                for layer_idx in self.send_layer_list
            }
        )

    def _init_before_run(self):
        self.pending_rounds: dict[int, RoundStoreState] = {}
        self.comm = ZMQCommunicator()
        crow_buffers = {
            info["rank"]: [wrapper.to_tensor() for wrapper in info["crow_ipc"]]
            for info in self.sender_info_list
        }
        col_buffers = {
            info["rank"]: [wrapper.to_tensor() for wrapper in info["col_ipc"]]
            for info in self.sender_info_list
        }
        block_buffers = {
            info["rank"]: [wrapper.to_tensor() for wrapper in info["block_ipc"]]
            for info in self.sender_info_list
        }
        self.store_items = StoreItems(
            layer_list=self.send_layer_list,
            tp_size=self.tp_size,
            num_heads=self.num_heads,
            data_blocks=self.data_blocks,
            crow_indices_remote_buffers=crow_buffers,
            col_indices_remote_buffers=col_buffers,
            blocks_remote_buffers=block_buffers,
        )
        self.port_queue.put(self.comm.my_port)
        self.port_to_rank = {info['port']: info['rank'] for info in self.sender_info_list}

        self.events_list: list[torch.cuda.Event] = []

    def _is_round_ready(self, state: RoundStoreState) -> bool:
        if state.hashes_list is None:
            return False
        if state.buf_idx is None:
            return False
        if len(state.events) != self.tp_size:
            return False
        for layer_idx in self.send_layer_list:
            for tp_rank in range(self.tp_size):
                if state.raw_data[layer_idx][tp_rank] is None:
                    return False
        return True

    def _store(self, msg):
        tp_rank = self.port_to_rank[msg["src_port"]]
        layer_idx = msg["layer_idx"]
        round_id = msg["round_id"]
        buf_idx = msg["buf_idx"]
        data = msg["data"]

        state = self.pending_rounds.get(round_id)
        if state is None:
            state = self._new_round_state(round_id)
            self.pending_rounds[round_id] = state

        if state.buf_idx is None:
            state.buf_idx = buf_idx
        else:
            assert state.buf_idx == buf_idx, (
                f"buf_idx mismatch in round {round_id}: {state.buf_idx} vs {buf_idx}"
            )

        if layer_idx == self.send_layer_list[0] and tp_rank == 0:
            state.hashes_list = msg["hashes_list"]

        assert state.raw_data[layer_idx][tp_rank] is None, (
            f"duplicate STORE: round={round_id}, layer={layer_idx}, tp={tp_rank}"
        )
        state.raw_data[layer_idx][tp_rank] = data

        if layer_idx == self.send_layer_list[-1]:
            state.events.append(msg["event"].reconstruct_event())

        if self._is_round_ready(state):
            for event in state.events:
                event.synchronize()

            self.store_items.insert_block(state, self.queue)

            # 通知所有 sender：这个 buffer 可以复用了
            for info in self.sender_info_list:
                self.comm.send(
                    {
                        "cmd": "RELEASE_BUFFER",
                        "round_id": round_id,
                        "buf_idx": buf_idx,
                    },
                    dst_port=info["port"],
                )

            del self.pending_rounds[round_id]

    def run(self):
        self._init_before_run()

        while True:
            msg = self.comm.recv()
            cmd = msg["cmd"]
            if cmd == "STORE":
                self._store(msg)
            else:
                raise NotImplementedError


class RetrieveItems:
    def __init__(
        self,
        ipc_wrapper_dict: dict,
    ):
        self.miss_indices = ipc_wrapper_dict["miss_indices"].to_tensor()
        self.hit_indices = ipc_wrapper_dict["hit_indices"].to_tensor()
        self.flash_attn_cu_seqlens_q = ipc_wrapper_dict["flash_attn_cu_seqlens_q"].to_tensor()
        self.bsr_batch_offsets = ipc_wrapper_dict["bsr_batch_offsets"].to_tensor()
        self.packed_row_block_pid_to_seq_id = ipc_wrapper_dict["packed_row_block_pid_to_seq_id"].to_tensor()
        self.packed_row_block_pid_to_row_block_pid = ipc_wrapper_dict["packed_row_block_pid_to_row_block_pid"].to_tensor()
        self.flash_attn_req_indices = ipc_wrapper_dict["flash_attn_req_indices"].to_tensor()
        self.hit_req_indices = ipc_wrapper_dict["hit_req_indices"].to_tensor()
        self.flash_attn_seqused_k = ipc_wrapper_dict["flash_attn_seqused_k"].to_tensor()
        self.layers = {}
        for layer_idx_str in ipc_wrapper_dict["layers"].keys():
            layer_idx = int(layer_idx_str)
            self.layers[layer_idx] = {
                "block_indices": ipc_wrapper_dict["layers"][layer_idx_str]["block_indices"].to_tensor(),  # receiver tensor handle
                "crow": ipc_wrapper_dict["layers"][layer_idx_str]["crow"].to_tensor(),
                "col": ipc_wrapper_dict["layers"][layer_idx_str]["col"].to_tensor(),
                "cu_crow_indices": ipc_wrapper_dict["layers"][layer_idx_str]["cu_crow_indices"].to_tensor(),
                "cu_col_indices": ipc_wrapper_dict["layers"][layer_idx_str]["cu_col_indices"].to_tensor(),
                "cu_col_indices_cpu": ipc_wrapper_dict["layers"][layer_idx_str]["cu_col_indices_cpu"].to_tensor(),
            }
    
    def transfer_meta_tensor(self, batch_state: BatchState):
        # This function can be called after retrieve_hashes to transfer all meta tensors to gpu for later use
        miss_indices = batch_state.miss_indices
        hit_indices = batch_state.hit_indices
        flash_attn_cu_seqlens_q = batch_state.flash_attn_cu_seqlens_q
        bsr_batch_offsets = batch_state.bsr_batch_offsets
        packed_row_block_pid_to_seq_id = batch_state.packed_row_block_pid_to_seq_id
        packed_row_block_pid_to_row_block_pid = batch_state.packed_row_block_pid_to_row_block_pid
        flash_attn_req_indices = batch_state.flash_attn_req_indices
        hit_req_indices = batch_state.hit_req_indices
        flash_attn_seqused_k = batch_state.flash_attn_seqused_k

        # bounds check
        assert self.miss_indices.shape[0] >= miss_indices.shape[0], f"miss_indices shape: {miss_indices.shape}, buffer shape: {self.miss_indices.shape}"
        assert self.hit_indices.shape[0] >= hit_indices.shape[0], f"hit_indices shape: {hit_indices.shape}, buffer shape: {self.hit_indices.shape}"
        assert self.flash_attn_cu_seqlens_q.shape[0] >= flash_attn_cu_seqlens_q.shape[0], f"flash_attn_cu_seqlens_q shape: {flash_attn_cu_seqlens_q.shape}, buffer shape: {self.flash_attn_cu_seqlens_q.shape}"
        assert self.bsr_batch_offsets.shape[0] >= bsr_batch_offsets.shape[0], f"bsr_batch_offsets shape: {bsr_batch_offsets.shape}, buffer shape: {self.bsr_batch_offsets.shape}"
        assert self.packed_row_block_pid_to_seq_id.shape[0] >= packed_row_block_pid_to_seq_id.shape[0], f"packed_row_block_pid_to_seq_id shape: {packed_row_block_pid_to_seq_id.shape}, buffer shape: {self.packed_row_block_pid_to_seq_id.shape}"
        assert self.packed_row_block_pid_to_row_block_pid.shape[0] >= packed_row_block_pid_to_row_block_pid.shape[0], f"packed_row_block_pid_to_row_block_pid shape: {packed_row_block_pid_to_row_block_pid.shape}, buffer shape: {self.packed_row_block_pid_to_row_block_pid.shape}"
        assert self.flash_attn_req_indices.shape[0] >= flash_attn_req_indices.shape[0], f"flash_attn_req_indices shape: {flash_attn_req_indices.shape}, buffer shape: {self.flash_attn_req_indices.shape}"
        assert self.hit_req_indices.shape[0] >= hit_req_indices.shape[0], f"hit_req_indices shape: {hit_req_indices.shape}, buffer shape: {self.hit_req_indices.shape}"
        assert self.flash_attn_seqused_k.shape[0] >= flash_attn_seqused_k.shape[0], f"flash_attn_seqused_k shape: {flash_attn_seqused_k.shape}, buffer shape: {self.flash_attn_seqused_k.shape}"

        self.miss_indices[:len(miss_indices)].copy_(miss_indices, non_blocking=True)
        self.hit_indices[:len(hit_indices)].copy_(hit_indices, non_blocking=True)
        self.flash_attn_cu_seqlens_q[:len(flash_attn_cu_seqlens_q)].copy_(flash_attn_cu_seqlens_q, non_blocking=True)
        self.bsr_batch_offsets[:len(bsr_batch_offsets)].copy_(bsr_batch_offsets, non_blocking=True)
        self.packed_row_block_pid_to_seq_id[:len(packed_row_block_pid_to_seq_id)].copy_(packed_row_block_pid_to_seq_id, non_blocking=True)
        self.packed_row_block_pid_to_row_block_pid[:len(packed_row_block_pid_to_row_block_pid)].copy_(packed_row_block_pid_to_row_block_pid, non_blocking=True)
        self.flash_attn_req_indices[:len(flash_attn_req_indices)].copy_(flash_attn_req_indices, non_blocking=True)
        self.hit_req_indices[:len(hit_req_indices)].copy_(hit_req_indices, non_blocking=True)
        self.flash_attn_seqused_k[:len(flash_attn_seqused_k)].copy_(flash_attn_seqused_k, non_blocking=True)


class RecvSessionWorker(mp_ctx.Process):
    def __init__(
        self,
        send_layer_list: list[int],
        ipc_wrapper: dict,
        max_num_reqs: int,
        queue: BiDirQueue,
        port_queue: mp.Queue,
    ):
        super().__init__(daemon=True)
        self.send_layer_list = send_layer_list
        self.ipc_wrapper = ipc_wrapper
        self.max_num_reqs = max_num_reqs
        self.queue = queue
        self.port_queue = port_queue

    def _init_before_run(self):
        # Initialize client's BatchState
        self.batch_state = BatchState(max_num_reqs=self.max_num_reqs, device="cpu")

        ## Initialize ipc_wrapper handles
        self.retrieve_items = RetrieveItems(self.ipc_wrapper)

        # Initialize retrieve stream
        self.compute_stream = torch.cuda.Stream()

        # Preallocated Tensor buffer to avoid cpu&gpu synchronization
        self.zero_tensor_cpu = torch.zeros((1,), dtype=torch.int32, device="cpu")
        self.zero_np = np.array([0], dtype=np.int32)

        self.cu_crow_indices_cpu_pinned = {layer_idx: torch.zeros(
            self.max_num_reqs+1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ) for layer_idx in self.send_layer_list}
        self.cu_col_indices_cpu_pinned = {layer_idx: torch.zeros(
            self.max_num_reqs+1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ) for layer_idx in self.send_layer_list}

        # Initialize BlockManagerProxy for communication with BlockManagerWorker
        self.block_manager_proxy = BlockManagerProxy(
            queue=self.queue,
        )

        # Create UDPSocket for communication with receiver_thread
        self.comm = ZMQCommunicator()
        self.recv_port: int | None = None  # set after recv from receiver thread

        self.port_queue.put(self.comm.my_port)

    def run(self):
        self._init_before_run()
        # wait for receiver thread to respond with recv_port
        while True:
            try:
                msg = self.comm.recv()
                self.recv_port = msg["src_port"]
                scheduler_output = msg["scheduler_output"]
                self.retrieve_hashes(scheduler_output)
            except Exception as e:
                print(f"[RecvSessionWorker] Exception in receiver process: {e}")
                traceback.print_exc()

    @prof_marker("retrieve_hashes")
    def retrieve_hashes(
        self, 
        scheduler_output: SchedulerOutput
    ):
        with prof_marker("update_from_scheduler_output"):
            self.batch_state.update_from_scheduler_output(scheduler_output)

        # No respond if empty batch
        if len(self.batch_state.current_batch_layout) == 0:
            return
        
        # No need to respond for decoding only step
        if max([num_tokens for _, num_tokens in self.batch_state.current_batch_layout]) <= 1:
            return

        # First get hit_list (cpu operations)
        hit_list: list[list[bool]] = [] # note that may have empty list for non-hit request
        hit_hashes_list: list[str] = []
        for hash_strs in self.batch_state.aligned_hashes:
            hit_list_per_req = self.block_manager_proxy.prefix_retrieve_hit_list(
                num_retrieve_layers=len(self.send_layer_list),
                hash_strs = hash_strs
            )
            hit_list.append(hit_list_per_req)
            for h, is_hit in zip(hash_strs, hit_list_per_req):
                if is_hit:
                    hit_hashes_list.append(h)
        with prof_marker("update_hits"):
            self.batch_state.update_hits(hit_list)

        # self.batch_state.print_hit_miss_status()

        # Send back current batch states to receiver thread
        data = {
            "req_ids": self.batch_state.req_ids,
            "current_batch_layout": self.batch_state.current_batch_layout,
            "slot_mapping": self.batch_state.slot_mapping,
            "req_states": {req_id: (self.batch_state.get_req_state(req_id).num_computed_tokens, self.batch_state.get_req_state(req_id).num_new_tokens) for req_id in self.batch_state.req_ids},
            "num_miss_tokens": self.batch_state.num_miss_tokens,
            "num_hit_tokens": self.batch_state.num_hit_tokens,
            "flash_attn_max_seqlen_q": self.batch_state.flash_attn_max_seqlen_q,
            "num_packed_block_rows": self.batch_state.num_packed_block_rows,
            "num_hit_reqs": self.batch_state.num_hit_reqs,
            "num_flash_attn_reqs": self.batch_state.num_flash_attn_reqs,
            "hit_trapezoid_sizes": self.batch_state.hit_trapezoid_sizes,
        }
        self.comm.send(data, self.recv_port)

        # Next get hit_metas
        hit_metas = {
            "hit_col_tensor": [],  # tensor or None (None means no hit for this request)
            "hit_block_tensor": [], # tensor or None (None means no hit for this request)
            "req_crow_tensor": [],
            "layer_cu_offsets": []
        }
        with prof_marker("retrieve_hit_metas"):
            with torch.cuda.stream(self.compute_stream):
                for i, _ in enumerate(self.batch_state.aligned_hashes):
                    hit_metas_per_req = self.block_manager_proxy.prefix_retrieve_hit_metas()
                    hit_metas["hit_col_tensor"].append(hit_metas_per_req.hit_col_tensor if hit_metas_per_req.hit_col_tensor is not None else None)
                    hit_metas["hit_block_tensor"].append(hit_metas_per_req.hit_block_tensor if hit_metas_per_req.hit_block_tensor is not None else None)
                    hit_metas["req_crow_tensor"].append(hit_metas_per_req.req_crow_tensor if hit_metas_per_req.req_crow_tensor is not None else None)
                    hit_metas["layer_cu_offsets"].append(hit_metas_per_req.layer_cu_offsets)

        # Skip retrieval if there are no hit tokens
        if self.batch_state.num_hit_tokens == 0:
            return

        # Continue perform non-blocking transfer of retrieved blocks
        with prof_marker("transfer metadata tensor"):
            with torch.cuda.stream(self.compute_stream):
                self.retrieve_items.transfer_meta_tensor(self.batch_state)

        for layer_off, (layer_idx, wrapper) in enumerate(self.retrieve_items.layers.items()):
            with prof_marker(f"transfer data, layer_{layer_idx}"):
                recv_block_indices = wrapper["block_indices"] # receiver tensor handle
                recv_crow = wrapper["crow"]
                recv_col = wrapper["col"]
                recv_cu_crow_indices = wrapper["cu_crow_indices"]
                recv_cu_col_indices = wrapper["cu_col_indices"]
                recv_cu_col_indices_cpu = wrapper["cu_col_indices_cpu"]

                self.cu_crow_indices_cpu_pinned[layer_idx].zero_()
                self.cu_col_indices_cpu_pinned[layer_idx].zero_()

                num_hit_reqs = 0 
                for i, cur_req_hits in enumerate(hit_list):
                    num_hits = sum(cur_req_hits)
                    if num_hits == 0:
                        continue

                    with torch.cuda.stream(self.compute_stream):
                        hit_col_tensor: torch.Tensor = hit_metas["hit_col_tensor"][i]
                        hit_block_tensor: torch.Tensor = hit_metas["hit_block_tensor"][i]
                        req_crow_tensor: torch.Tensor = hit_metas["req_crow_tensor"][i]
                        layer_cu_offsets: list[int] = hit_metas["layer_cu_offsets"][i]
                        layer_start = layer_cu_offsets[layer_off]
                        layer_end = layer_cu_offsets[layer_off+1]
                        # print(f"layer_idx: {layer_idx}, num_hits: {num_hits}, layer_start: {layer_start}, layer_end: {layer_end}")

                        cur_req_col = hit_col_tensor[layer_start:layer_end]
                        cur_req_block_indices = hit_block_tensor[layer_start:layer_end]
                        cur_req_crow = req_crow_tensor[layer_off] # [H+1]
                        num_crow = cur_req_crow.shape[0]
                        num_col = cur_req_col.shape[0]
                        start_crow = self.cu_crow_indices_cpu_pinned[layer_idx][num_hit_reqs].item()
                        end_crow = start_crow + num_crow
                        # always bounds check before copy_
                        assert end_crow <= recv_crow.shape[0], f"recv_crow shape: {recv_crow.shape}, end_crow: {end_crow}"
                        recv_crow[start_crow:end_crow].copy_(cur_req_crow, non_blocking=True)
                        self.cu_crow_indices_cpu_pinned[layer_idx][num_hit_reqs+1] = end_crow
                        start_col = self.cu_col_indices_cpu_pinned[layer_idx][num_hit_reqs].item()
                        end_col = start_col + num_col
                        assert end_col <= recv_col.shape[0], f"recv_col shape: {recv_col.shape}, end_col: {end_col}"
                        recv_col[start_col:end_col].copy_(cur_req_col, non_blocking=True)
                        recv_block_indices[start_col:end_col].copy_(cur_req_block_indices, non_blocking=True)
                        self.cu_col_indices_cpu_pinned[layer_idx][num_hit_reqs+1] = end_col

                    num_hit_reqs += 1

                with torch.cuda.stream(self.compute_stream):
                    assert num_hit_reqs + 1 <= recv_cu_crow_indices.shape[0], f"Number of hit requests {num_hit_reqs + 1} exceeds buffer size {recv_cu_crow_indices.shape[0]}"
                    recv_cu_crow_indices[:num_hit_reqs+1].copy_(self.cu_crow_indices_cpu_pinned[layer_idx][:num_hit_reqs+1], non_blocking=True)
                    recv_cu_col_indices[:num_hit_reqs+1].copy_(self.cu_col_indices_cpu_pinned[layer_idx][:num_hit_reqs+1], non_blocking=True)
                    recv_cu_col_indices_cpu[:num_hit_reqs+1].copy_(self.cu_col_indices_cpu_pinned[layer_idx][:num_hit_reqs+1], non_blocking=True)

                self.compute_stream.synchronize()
                self.comm.send({"layer_idx": layer_idx}, self.recv_port)

        # unpin all retrieved hit blocks
        self.block_manager_proxy.retrieve_unpin(hit_hashes_list)
