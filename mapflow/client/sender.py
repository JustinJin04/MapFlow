import os
import threading
import queue
import torch
from dataclasses import dataclass

from vllm.v1.core.sched.output import SchedulerOutput
from mapflow.kernel.compress_page_triton import get_top_p_metas, get_bsr_metadata_async, sparse_extract_triton
from mapflow.batch_states import BatchState
from mapflow.core.ipc_utils import CudaIPCWrapper, CudaIpcEventWrapper
from mapflow.core.zmq import ZMQCommunicator
from mapflow.core.prof_marker import prof_marker


class DoubleIPCBuffer:
    def __init__(
        self,
        shape: tuple[int],
        dtype: torch.dtype,
        device: str,
        num_buffers: int = 2,
    ):
        self.shape = shape
        self.max_elems = shape[0]
        self.num_buffers = num_buffers
        self.buffers = [
            torch.empty(shape, dtype=dtype, device=device)
            for _ in range(num_buffers)
        ]
        self.heads = [0 for _ in range(num_buffers)]
        self.device = device

    def get_ipc_wrappers(self):
        return [CudaIPCWrapper(buf) for buf in self.buffers]

    def reset_round(self, buf_idx: int):
        assert 0 <= buf_idx < self.num_buffers
        self.heads[buf_idx] = 0

    def allocate(self, buf_idx: int, num_elems: int) -> tuple[int, int]:
        assert 0 <= buf_idx < self.num_buffers
        assert num_elems <= self.max_elems, (
            f"num_elems={num_elems}, capacity={self.max_elems}"
        )
        head = self.heads[buf_idx]
        end = head + num_elems
        assert end <= self.max_elems, (
            f"double buffer overflow: buf_idx={buf_idx}, "
            f"head={head}, num_elems={num_elems}, capacity={self.max_elems}"
        )
        self.heads[buf_idx] = end
        return head, end

    def insert(self, buf_idx: int, data: torch.Tensor) -> tuple[int, int]:
        assert len(data.shape) == len(self.shape)
        if len(data.shape) >= 2:
            assert data.shape[1:] == self.shape[1:]
        start, end = self.allocate(buf_idx, data.shape[0])
        self.buffers[buf_idx][start:end].copy_(data, non_blocking=True)
        return start, end

@dataclass
class CompressOutput:
    layer_idx: int
    num_blocks_k: int
    crow_start: int
    crow_end: int
    col_start: int
    col_end: int
    block_start: int
    block_end: int
    hash_strs: list[str]


@dataclass
class ReqCompressSnapshot:
    slot: int
    num_new_tokens: int
    num_tokens_need_to_drop: int
    num_last_pending_tokens: int
    num_pending_tokens: int
    num_total_tokens: int
    block_hashes: list[str]


class CompressManager:
    def __init__(
        self,
        layer_list: list[int],
        max_num_reqs: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        top_p: float,
        crow_ipc_buffer: DoubleIPCBuffer,
        col_ipc_buffer: DoubleIPCBuffer,
        block_ipc_buffer: DoubleIPCBuffer,
        rank: int,
        device: str,
    ):
        self.layer_list = layer_list
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_p = top_p
        self.crow_ipc_buffer = crow_ipc_buffer
        self.col_ipc_buffer = col_ipc_buffer
        self.block_ipc_buffer = block_ipc_buffer
        self.rank = rank

        self.q_pending_cache = {layer_idx: 
            torch.empty(
                (max_num_reqs, block_size, num_heads, head_dim), 
                dtype=torch.bfloat16,
                device=device
            ) for layer_idx in layer_list
        }

        self.pinned_nnz = torch.zeros((len(layer_list) * max_num_reqs,), dtype=torch.int32, pin_memory=True)

    def compress(
        self,
        batch_snapshot: list[ReqCompressSnapshot],
        querys: dict[int, torch.Tensor],
        key_caches: dict[int, torch.Tensor],
        block_table: torch.Tensor,
        buf_idx: int,
        stream: torch.cuda.Stream,
    ) -> dict[int, list[CompressOutput]]:
        
        ret = {layer_idx: [] for layer_idx in self.layer_list}
        
        # --- PHASE 1: Asynchronous Execution & Metadata Collection ---
        async_tasks = {layer_idx: [] for layer_idx in self.layer_list}
        nnz_tensors = []
        
        for layer_idx in self.layer_list:
            q = querys[layer_idx]
            key_cache = key_caches[layer_idx]

            cur_off = 0
            for i, req_snap in enumerate(batch_snapshot):
                with prof_marker(f"prepare for compress_{layer_idx}"):
                    if req_snap.num_new_tokens == 1:
                        # skip decode tokens
                        cur_off += 1
                        continue

                    q_slice = q[cur_off : cur_off + req_snap.num_new_tokens]
                    cur_off += req_snap.num_new_tokens

                    # Case 0.0: all tokens needs to be dropped
                    if req_snap.num_tokens_need_to_drop == req_snap.num_new_tokens:
                        continue
                    if req_snap.num_tokens_need_to_drop == 0:
                        # Case 0.1: already elimiate impact from prefix cache
                        num_last_q_pending_cache_tokens = req_snap.num_last_pending_tokens
                        num_need_cached_tokens = req_snap.num_new_tokens
                    else:
                        # Case 0.2: drop first part of q_slice and only keep the last part
                        q_slice = q_slice[req_snap.num_tokens_need_to_drop:]
                        num_last_q_pending_cache_tokens = 0
                        num_need_cached_tokens = req_snap.num_new_tokens - req_snap.num_tokens_need_to_drop

                    # Case 1: num_new_tokens (plus past content) cannot fullfill one block
                    num_total_new_tokens = num_last_q_pending_cache_tokens + num_need_cached_tokens
                    if num_total_new_tokens < self.block_size:
                        # Just append to pending cache
                        p_start = num_last_q_pending_cache_tokens
                        p_end = p_start + num_need_cached_tokens
                        self.q_pending_cache[layer_idx][req_snap.slot, p_start:p_end] = q_slice
                        continue
                    
                    # Case 2: form at least one new block
                    # Prepare aligned query and key for compression
                    num_process_tokens = num_total_new_tokens // self.block_size * self.block_size
                    num_pending_tokens = num_total_new_tokens % self.block_size
                    assert num_pending_tokens == req_snap.num_pending_tokens
                    
                    query = torch.empty((num_process_tokens, self.num_heads, self.head_dim), dtype=q.dtype, device=q.device)
                    query[:num_last_q_pending_cache_tokens].copy_(
                        self.q_pending_cache[layer_idx][req_snap.slot, :num_last_q_pending_cache_tokens]
                    )
                    query[num_last_q_pending_cache_tokens:num_process_tokens].copy_(
                        q_slice[: num_process_tokens - num_last_q_pending_cache_tokens]
                    )
                    k_end = req_snap.num_total_tokens - num_pending_tokens
                    num_query_blocks = num_process_tokens // self.block_size
                    hash_strs = req_snap.block_hashes[-num_query_blocks:]
                    
                    # Fill query from pending cache and new slice
                    self.q_pending_cache[layer_idx][req_snap.slot, :num_pending_tokens].copy_(
                        q_slice[num_process_tokens - num_last_q_pending_cache_tokens:]
                    )

                # Actual aligned compression
                with prof_marker(f"get_mask_row_lse_{layer_idx}"):
                    mask, row_lse = get_top_p_metas(
                        q=query,
                        k_cache=key_cache,
                        block_table=block_table[i],
                        num_tokens_kv=k_end,
                        block_size=self.block_size,
                        top_p=self.top_p
                    )

                # Generate metadata entirely asynchronously
                with prof_marker(f"get_bsr_metadata_async_{layer_idx}"):
                    crow_indices, col_prealloc, nnz_offsets, nnz_tensor = get_bsr_metadata_async(mask)
                
                async_tasks[layer_idx].append({
                    "k_end": k_end,
                    "hash_strs": hash_strs,
                    "query": query,
                    "block_table": block_table[i],
                    "row_lse": row_lse,
                    "crow_indices": crow_indices,
                    "col_prealloc": col_prealloc,
                    "nnz_offsets": nnz_offsets,
                })
                nnz_tensors.append(nnz_tensor)

        if not nnz_tensors:
            return ret

        with prof_marker("phase2_sync"):
            # --- PHASE 2: Single Batch Sync via Pinned Memory ---
            all_nnz_gpu = torch.cat(nnz_tensors)
            total_tasks = len(nnz_tensors)
            
            if self.pinned_nnz.shape[0] < total_tasks:
                self.pinned_nnz = torch.zeros((total_tasks * 2,), dtype=torch.int32, pin_memory=True)

            self.pinned_nnz[:total_tasks].copy_(all_nnz_gpu, non_blocking=True)
            
            # 🔥 这里的同步现在只会阻塞运行此函数的后台线程及其绑定的 CUDA Stream 🔥
            # torch.cuda.current_stream().synchronize()
            stream.synchronize()
        
        # --- PHASE 3: IPC Allocation & Sparse Extraction ---
        task_idx = 0
        for layer_idx in self.layer_list:
            key_cache = key_caches[layer_idx]
            
            for task in async_tasks[layer_idx]:
                exact_nnz = self.pinned_nnz[task_idx].item()
                task_idx += 1
                
                exact_col = task["col_prealloc"][:exact_nnz]
                
                crow_start, crow_end = self.crow_ipc_buffer.insert(buf_idx, task["crow_indices"])
                col_start, col_end = self.col_ipc_buffer.insert(buf_idx, exact_col)
                block_start, block_end = self.block_ipc_buffer.allocate(buf_idx, exact_nnz)


                q_blocks = task["query"].shape[0] // self.block_size
                k_blocks = task["k_end"] // self.block_size
                total_blocks = (2*k_blocks - q_blocks + 1) * q_blocks // 2 * self.num_heads
                density = exact_nnz / total_blocks if total_blocks > 0 else 0.0
                # print(f"Layer {layer_idx}: exact_nnz={exact_nnz}, total_blocks={total_blocks}, density={density:.4f}", flush=True)


                with prof_marker(f"sparse_extract_triton_{layer_idx}"):
                    sparse_extract_triton(
                        q=task["query"],
                        k_cache=key_cache,
                        block_table=task["block_table"],
                        num_tokens_kv=task["k_end"],
                        nnz_offsets=task["nnz_offsets"],
                        row_lse=task["row_lse"],
                        values=self.block_ipc_buffer.buffers[buf_idx][block_start:block_end],
                    )

                ret[layer_idx].append(CompressOutput(
                    layer_idx=layer_idx,
                    num_blocks_k=task["k_end"] // self.block_size,
                    crow_start=crow_start,
                    crow_end=crow_end,
                    col_start=col_start,
                    col_end=col_end,
                    block_start=block_start,
                    block_end=block_end,
                    hash_strs=task["hash_strs"],
                ))
                
        return ret


class ClientSender:
    def __init__(
        self,
        num_layers: int,
        num_ignored_layers: int,
        num_total_heads: int,  # before tp partition
        tp_rank: int,
        tp_size: int,
        head_dim: int,
        block_size: int,
        top_p: float,
        server_main_port: int,
        device: str,
        max_num_reqs: int = 128,
        # ipc_buffer_size: int = 2048000,
        ipc_buffer_size: int = 3096000,
        # ipc_buffer_size: int = 3548000,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_ignored_layers = num_ignored_layers
        self.num_heads = num_total_heads // tp_size
        self.tp_rank = tp_rank
        self.head_dim = head_dim
        self.top_p = top_p
        self.block_size = block_size
        self.device = device
        self.torch_dtype = torch_dtype

        self.send_layer_list = list(range(num_ignored_layers, num_layers - 1))
        self.comm = ZMQCommunicator()
        self.session_port: int | None = None

        self.batch_state = BatchState(max_num_reqs=max_num_reqs, block_size=block_size, device="cpu")

        self.crow_ipc_buffer = DoubleIPCBuffer(shape=(ipc_buffer_size,), dtype=torch.int32, device=device)
        self.col_ipc_buffer = DoubleIPCBuffer(shape=(ipc_buffer_size,), dtype=torch.int32, device=device)
        self.block_ipc_buffer = DoubleIPCBuffer(shape=(ipc_buffer_size, block_size, block_size), dtype=torch_dtype, device=device)

        self.next_round_id = 0
        self.buffer_busy_round: list[int | None] = [None, None]

        self.compress_manager = CompressManager(
            layer_list=self.send_layer_list,
            max_num_reqs=max_num_reqs,
            block_size=block_size,
            num_heads=self.num_heads,
            head_dim=head_dim,
            top_p=top_p,
            crow_ipc_buffer=self.crow_ipc_buffer,
            col_ipc_buffer=self.col_ipc_buffer,
            block_ipc_buffer=self.block_ipc_buffer,
            rank=tp_rank,
            device=device,
        )

        # Note here setting maxsize will block vllm engine and reduce throughput
        self.task_queue = queue.Queue()

        self.bg_stream = torch.cuda.Stream(device=device)

        self._step_querys = {}
        self._step_key_caches = {}
        self._step_block_table = None

        self.worker_thread = threading.Thread(target=self._compression_worker_loop, daemon=True)
        self.worker_thread.start()

        self.ipc_event: torch.cuda.Event | None = None
        self._init_server(server_main_port)

        print(
            f"Initialize BSR Sender Client. "
            f"num_layers={num_layers}, num_ignored_layers={num_ignored_layers},"
            f"num_total_heads={num_total_heads}, tp_rank={tp_rank}, tp_size={tp_size}, head_dim={head_dim}, "
            f"block_size={block_size}, top_p={top_p}"
            , flush=True
        )

    def _init_server(self, server_main_port: int):
        """
        register to server by sending the IPC wrapper of the shared buffers
        """
        content = {
            "cmd": "REGISTER_SENDER",
            "rank": self.tp_rank,
            "crow_ipc": self.crow_ipc_buffer.get_ipc_wrappers(),
            "col_ipc": self.col_ipc_buffer.get_ipc_wrappers(),
            "block_ipc": self.block_ipc_buffer.get_ipc_wrappers(),
        }
        self.comm.send(content, dst_port=server_main_port)
        resp = self.comm.recv()
        self.session_port = resp["session_port"]

    def update_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
    ):
        """called inside gpu model runner (within each TP worker)
        track the hashes of cols/blocks that stored in current iteration
        """
        self.batch_state.update_from_scheduler_output(scheduler_output)

    def _should_ignore(self, layer_idx: int) -> bool:
        return layer_idx < self.num_ignored_layers or layer_idx == self.num_layers - 1

    def maybe_store(
        self,
        layer_idx: int,
        q: torch.Tensor, # [B, m//tp, D]
        key_cache: torch.Tensor, # [num_pages, 16, m_kv//tp, D] if tp <= m_kv else 1 
        block_table: torch.Tensor, # [num_reqs, max_pages_per_req]
    ):
        """
        for each prefill request,
        1. get col indices
        2. get and store blocks in block_ipc_buffer
        3. if last layer, insert event to make sure all operations are done before server accessing the buffer
        """

        if self._should_ignore(layer_idx) or self.batch_state.decode_only:
            return

        # Collect current layer's query and key_cache for later compression
        self._step_querys[layer_idx] = q
        self._step_key_caches[layer_idx] = key_cache
        if self._step_block_table is None:
            # note here we should clone another copy 
            # because vllm will reuse same buffer for block_table across iteration
            self._step_block_table = block_table.clone().contiguous().to(torch.int32)

        # Trigger compression and sending at the last layer
        if layer_idx == self.send_layer_list[-1]:
            main_stream = torch.cuda.current_stream(q.device)
            main_stream_event = torch.cuda.Event()
            main_stream_event.record(main_stream)

            # Extract light weight copy of batch state
            batch_snapshot = []
            for req_id in self.batch_state.req_ids:
                slot = self.batch_state.get_req_slot(req_id)
                req_state = self.batch_state.get_req_state(req_id)
                batch_snapshot.append(ReqCompressSnapshot(
                    slot=slot,
                    num_new_tokens=req_state.num_new_tokens,
                    num_tokens_need_to_drop=req_state.num_tokens_need_to_drop,
                    num_last_pending_tokens=req_state.num_last_pending_tokens,
                    num_pending_tokens=len(req_state.pending_tokens),
                    num_total_tokens=len(req_state.tokens),
                    block_hashes=req_state.block_hashes[:]
                ))

            # Submit to background thread
            round_id = self.next_round_id
            self.next_round_id += 1

            self.task_queue.put({
                "round_id": round_id,
                "batch_snapshot": batch_snapshot,
                "querys": self._step_querys,
                "key_caches": self._step_key_caches,
                "block_table": self._step_block_table,
                "main_stream_event": main_stream_event,
            })

            self._step_querys = {}
            self._step_key_caches = {}
            self._step_block_table = None

    def _wait_until_buffer_free(self, buf_idx: int):
        while self.buffer_busy_round[buf_idx] is not None:
            msg = self.comm.recv()
            assert msg["cmd"] == "RELEASE_BUFFER", f"unexpected msg: {msg}"
            rel_buf_idx = msg["buf_idx"]
            rel_round_id = msg["round_id"]
            assert self.buffer_busy_round[rel_buf_idx] == rel_round_id, (
                f"release mismatch: local={self.buffer_busy_round}, msg={msg}"
            )
            self.buffer_busy_round[rel_buf_idx] = None

    @torch.inference_mode()
    def _compression_worker_loop(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break

            round_id = task["round_id"]
            buf_idx = round_id % 2

            self._wait_until_buffer_free(buf_idx)
            # print(f"buf_idx: {buf_idx}, round_id: {round_id}, buffer_busy_round: {self.buffer_busy_round}", flush=True)
            self.buffer_busy_round[buf_idx] = round_id

            self.crow_ipc_buffer.reset_round(buf_idx)
            self.col_ipc_buffer.reset_round(buf_idx)
            self.block_ipc_buffer.reset_round(buf_idx)

            with torch.cuda.stream(self.bg_stream):
                self.bg_stream.wait_event(task["main_stream_event"])

                # Core Modify!!
                # if we don't record at bg_stream, the memory of block_table will be recycled after main stream has finished
                task["block_table"].record_stream(self.bg_stream)
                # for layer_idx in task["querys"]:
                    # task["querys"][layer_idx].record_stream(self.bg_stream)
                    # task["key_caches"][layer_idx].record_stream(self.bg_stream)

                compress_results = self.compress_manager.compress(
                    batch_snapshot=task["batch_snapshot"],
                    querys=task["querys"],
                    key_caches=task["key_caches"],
                    block_table=task["block_table"],
                    buf_idx=buf_idx,
                    stream=self.bg_stream,
                )

                if len(compress_results[self.send_layer_list[-1]]) == 0:
                    self.buffer_busy_round[buf_idx] = None
                    continue

                self.ipc_event = torch.cuda.Event(interprocess=True)
                self.ipc_event.record(self.bg_stream)
                ipc_event_wrapper = CudaIpcEventWrapper(self.ipc_event)

                self._send_zmq_results(
                    round_id=round_id,
                    buf_idx=buf_idx,
                    compress_results=compress_results,
                    ipc_event_wrapper=ipc_event_wrapper,
                )

    def _send_zmq_results(
        self,
        round_id: int,
        buf_idx: int,
        compress_results: dict[int, list[CompressOutput]],
        ipc_event_wrapper: CudaIpcEventWrapper,
    ):
        num_total_items = 0
        res = compress_results[self.send_layer_list[0]]
        for r in res:
            num_total_items += len(r.hash_strs)
        # print(f"Sending results for round {round_id}, buf_idx {buf_idx}, items: {num_total_items}", flush=True)
        for l in self.send_layer_list:
            resp = {
                "cmd": "STORE",
                "rank": self.tp_rank,
                "round_id": round_id,
                "buf_idx": buf_idx,
                "layer_idx": l,
                "hashes_list": [res.hash_strs for res in compress_results[l]],
                "data": [{
                    "num_blocks_k": res.num_blocks_k,
                    "crow": (res.crow_start, res.crow_end),
                    "col": (res.col_start, res.col_end),
                    "block": (res.block_start, res.block_end),
                } for res in compress_results[l]],
            }
            if l == self.send_layer_list[-1]:
                resp["event"] = ipc_event_wrapper

            self.comm.send(resp, dst_port=self.session_port)


    @property
    def is_sender(self):
        return True