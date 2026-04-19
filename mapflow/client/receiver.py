import traceback
import os
import torch
import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from mapflow.core.ipc_utils import CudaIPCWrapper, CpuIPCWrapper
from mapflow.kernel.bsr_varlen_page_kernel_search_k import bsr_varlen_page_triton
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.core.sched.output import SchedulerOutput
from mapflow.core.prof_marker import prof_marker
from mapflow.core.zmq import ZMQCommunicator

mp_ctx = mp.get_context('spawn')

class BSRBuffer:
    def __init__(
        self,
        layer_list: list[int],
        max_num_blocks_per_layer: int,
        max_num_seq_blocks_per_layer: int,
        block_size: int,
        max_num_reqs: int,
        max_seq_len: int,
        device: str,
        # dtype: torch.dtype,
    ):
        self.miss_indices = torch.empty((max_seq_len,), dtype=torch.int32, device=device)
        self.hit_indices = torch.empty((max_seq_len,), dtype=torch.int32, device=device)
        self.flash_attn_cu_seqlens_q = torch.empty((max_num_reqs + 1,), dtype=torch.int32, device=device)
        self.bsr_batch_offsets = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.packed_row_block_pid_to_seq_id = torch.empty((max_seq_len // block_size,), dtype=torch.int32, device=device)
        self.packed_row_block_pid_to_row_block_pid = torch.empty((max_seq_len // block_size,), dtype=torch.int32, device=device)
        self.flash_attn_req_indices = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.hit_req_indices = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.flash_attn_seqused_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.layers = {}
        for i in layer_list:
            self.layers[i] = {
                "block_indices": torch.empty((max_num_blocks_per_layer,), dtype=torch.int32, device=device),
                "crow": torch.empty((max_num_seq_blocks_per_layer + 1,), dtype=torch.int32, device=device),
                "col": torch.empty((max_num_blocks_per_layer,), dtype=torch.int32, device=device),
                "cu_crow_indices": torch.empty((max_num_reqs + 1,), dtype=torch.int32, device=device),
                "cu_col_indices": torch.empty((max_num_reqs + 1,), dtype=torch.int32, device=device),
                "cu_col_indices_cpu": torch.empty((max_num_reqs+1,), dtype=torch.int32, device="cpu", pin_memory=True),
            }

    def get_ipc_wrapper(self):
        ipc_wrapper = {}
        ipc_wrapper["miss_indices"] = CudaIPCWrapper(self.miss_indices)
        ipc_wrapper["hit_indices"] = CudaIPCWrapper(self.hit_indices)
        ipc_wrapper["flash_attn_cu_seqlens_q"] = CudaIPCWrapper(self.flash_attn_cu_seqlens_q)
        ipc_wrapper["bsr_batch_offsets"] = CudaIPCWrapper(self.bsr_batch_offsets)
        ipc_wrapper["packed_row_block_pid_to_seq_id"] = CudaIPCWrapper(self.packed_row_block_pid_to_seq_id)
        ipc_wrapper["packed_row_block_pid_to_row_block_pid"] = CudaIPCWrapper(self.packed_row_block_pid_to_row_block_pid)
        ipc_wrapper["flash_attn_req_indices"] = CudaIPCWrapper(self.flash_attn_req_indices)
        ipc_wrapper["hit_req_indices"] = CudaIPCWrapper(self.hit_req_indices)
        ipc_wrapper["flash_attn_seqused_k"] = CudaIPCWrapper(self.flash_attn_seqused_k)
        ipc_wrapper["layers"] = {}
        for layer_idx in self.layers:
            ipc_wrapper["layers"][layer_idx] = {}
            for k, v in self.layers[layer_idx].items():
                if v.is_cuda:
                    ipc_wrapper["layers"][layer_idx][k] = CudaIPCWrapper(v)
                else:
                    ipc_wrapper["layers"][layer_idx][k] = CpuIPCWrapper(v)
        return ipc_wrapper

@dataclass
class ClientBatchState:
    req_ids: list[str] = field(default_factory=list)
    current_batch_layout: list[tuple[str, int]] = field(default_factory=list)  # list of (req_id, num_tokens)
    slot_mapping: dict[str, int] = field(default_factory=dict)
    req_states: dict[str, tuple[int, int]] = field(default_factory=dict)  # req_id -> (num_computed_tokens, num_new_tokens)

    num_miss_tokens: int = 0
    num_hit_tokens: int = 0
    flash_attn_max_seqlen_q: int = 0
    num_packed_block_rows: int = 0
    num_hit_reqs: int = 0
    num_flash_attn_reqs: int = 0
    hit_trapezoid_sizes: int = 0

    def update(
        self,
        data: dict
    ):
        self.req_ids = data["req_ids"]
        self.current_batch_layout = data["current_batch_layout"]
        self.slot_mapping = data["slot_mapping"]
        self.req_states = data["req_states"]
        self.num_miss_tokens = data["num_miss_tokens"]
        self.num_hit_tokens = data["num_hit_tokens"]
        self.flash_attn_max_seqlen_q = data["flash_attn_max_seqlen_q"]
        self.num_packed_block_rows = data["num_packed_block_rows"]
        self.num_hit_reqs = data["num_hit_reqs"]
        self.num_flash_attn_reqs = data["num_flash_attn_reqs"]
        self.hit_trapezoid_sizes = data["hit_trapezoid_sizes"]

    def get_req_slot(self, req_id: str) -> int:
        assert req_id in self.slot_mapping, f"req_id {req_id} not in current requests: {list(self.slot_mapping.keys())}"
        return self.slot_mapping[req_id]

    def get_req_num_computed_tokens(self, req_id: str) -> int:
        assert req_id in self.req_states, f"req_id {req_id} not in current requests: {list(self.req_states.keys())}"
        return self.req_states[req_id][0]
    
    def get_req_num_new_tokens(self, req_id: str) -> int:
        assert req_id in self.req_states, f"req_id {req_id} not in current requests: {list(self.req_states.keys())}"
        return self.req_states[req_id][1]

    @property
    def num_scheduled_tokens(self) -> int:
        return sum([num_tokens for req_id, num_tokens in self.current_batch_layout])

class ClientReceiverWorker(mp_ctx.Process):
    def __init__(
        self,
        retrieve_queue: mp.Queue,
        state_pipe_send: Connection,
        ack_flags: torch.Tensor,
        server_thread_port: int,
        send_layer_list: list[int],
    ):
        super().__init__(daemon=True)
        self.retrieve_queue = retrieve_queue
        self.state_pipe_send = state_pipe_send
        self.ack_flags = ack_flags
        self.server_thread_port = server_thread_port
        self.send_layer_list = send_layer_list

    def run(self):
        self.comm = ZMQCommunicator()
        print(f"[ReceiverProcess] Started. PID: {self.pid}")

        while True:
            try:
                # It is fine to block here because this waits for the main thread 
                # to kick off the next iteration. It doesn't block the GPU path.
                req = self.retrieve_queue.get()
                cmd = req["cmd"]
                if cmd == "EXIT":
                    break
                assert cmd == "RETRIEVE_HASH_BLOCKS"
                scheduler_output = req["scheduler_output"]
                iteration = req["iteration"]
                self._handle_retrieve_blocks(
                    scheduler_output=scheduler_output,
                    iteration=iteration
                )
            except Exception as e:
                print(f"[ReceiverProcess] Exception: {e}")
                traceback.print_exc()

    def _handle_retrieve_blocks(
        self,
        scheduler_output: SchedulerOutput,
        iteration: int
    ):
        req = {
            "scheduler_output": scheduler_output,
        }
        self.comm.send(req, self.server_thread_port)

        num_scheduled_tokens: dict[str, int] = scheduler_output.num_scheduled_tokens
        # No need to propagate due to no scheduled tokens (empty batch)
        if len(num_scheduled_tokens) == 0:
            return
        
        # Skip transferring query/key if all requests are in decoding phase
        if max(num_scheduled_tokens.values()) == 1:
            return

        data = self.comm.recv()
        self.state_pipe_send.send((iteration, data))
        # Elif there are no hit tokens, skip recv from server and directly ack
        if data["num_hit_tokens"] == 0:
            for layer_idx in self.send_layer_list:
                # Lock-free write to shared memory
                self.ack_flags[layer_idx] = iteration
            return

        # Else need sync
        for layer_idx in self.send_layer_list:
            ack_data = self.comm.recv()
            assert ack_data["layer_idx"] == layer_idx, f"Expected layer_idx {layer_idx}, got {ack_data['layer_idx']}"
            
            # Lock-free write to shared memory
            self.ack_flags[layer_idx] = iteration

class ClientProcessSession:
    def __init__(
        self,
        client_batch_state: ClientBatchState,
        send_layer_list: list[int],
        server_thread_port: int,
        state_pipe_recv: Connection,
        state_pipe_send: Connection,
        ack_flags: torch.Tensor,
        force_wait_sync: bool,
    ):
        self.server_thread_port = server_thread_port
        self.client_batch_state = client_batch_state
        self.send_layer_list = send_layer_list
        self.last_updated_iteration = -1
        self.force_wait_sync = force_wait_sync

        # Direct communication channels
        self.retrieve_queue = mp_ctx.Queue()  # main -> process (blocking is fine here)
        self.state_pipe_recv = state_pipe_recv
        self.ack_flags = ack_flags

        # Initialize Process Worker
        self.worker_process = ClientReceiverWorker(
            retrieve_queue=self.retrieve_queue,
            state_pipe_send=state_pipe_send,
            ack_flags=self.ack_flags,
            server_thread_port=server_thread_port,
            send_layer_list=self.send_layer_list
        )
        self.worker_process.start()

    def put(self, req: dict):
        cmd = req["cmd"]
        assert cmd == "RETRIEVE_HASH_BLOCKS"
        self.retrieve_queue.put(req)
        return

    def _sync_state(
        self,
        expected_iteration: int,
        prev_layer_event: torch.cuda.Event | None,
    ) -> bool:
        if self.last_updated_iteration == expected_iteration:
            return True

        while True:
            if self.state_pipe_recv.poll():
                iteration, data = self.state_pipe_recv.recv()
                if iteration == expected_iteration:
                    self.client_batch_state.update(data)
                    self.last_updated_iteration = iteration
                    return True
                # 丢弃旧 iteration，继续等
            else:
                if not self.force_wait_sync:
                    if prev_layer_event.query():
                        return False

    def get_or_fallback(
        self,
        expected_iteration: int,
        expected_layer_idx: int,
        prev_layer_event: torch.cuda.Event | None,
    ) -> bool:
        state_ready = self._sync_state(expected_iteration, prev_layer_event)
        if not state_ready:
            return False

        while True:
            if self.ack_flags[expected_layer_idx] == expected_iteration:
                return True

            if not self.force_wait_sync:
                if prev_layer_event.query():
                    return False

    def __del__(self):
        if hasattr(self, 'worker_process'):
            self.retrieve_queue.put({"cmd": "EXIT"})
            self.worker_process.join()

class ClientReceiver:
    def __init__(
        self,
        num_layers: int,
        weights: Dict[int, torch.Tensor],
        num_heads: int,
        head_dim: int,
        block_size: int,
        num_send_layers: int,
        server_main_port: int,
        max_num_blocks_per_layer: int = 204000,
        max_num_seq_blocks_per_layer: int = 640,
        max_num_reqs: int = 128,
        max_seq_len: int = 40960,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_ignored_layers: int=3,
        force_wait_sync: bool = False,
    ):
        self.num_layers = num_layers
        self.weights = weights
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.num_ignored_layers = num_ignored_layers
        self.num_send_layers = num_send_layers
        self.server_main_port = server_main_port
        for i in self.weights:
            self.weights[i] = self.weights[i].to(device).to(dtype)

        # Initialize send&recv layer mapping
        self.reuse_layer_idx = list(range(self.num_ignored_layers, self.num_layers - 1))
        self.recv_to_send_layer_idx = {}
        for i in self.reuse_layer_idx:
            self.recv_to_send_layer_idx[i] = num_send_layers - num_layers + i  # reverse mapping

        # Initialize bsr_buffer
        send_layer_list = [self.recv_to_send_layer_idx[i] for i in self.reuse_layer_idx]
        self.bsr_buffer = BSRBuffer(
            layer_list=send_layer_list,
            max_num_blocks_per_layer=max_num_blocks_per_layer,
            max_num_seq_blocks_per_layer=max_num_seq_blocks_per_layer,
            block_size=block_size,
            max_num_reqs=max_num_reqs,
            max_seq_len=max_seq_len,
            device=device,
            # dtype=dtype,
        )

        # Initialize output buffer (for flash_attn and bsr_attn)
        self.output_buffer = torch.empty((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)
        
        # Initialize ZMQ socket
        self.comm = ZMQCommunicator()

        # Register bsr_buffer ipc wrappers at server side
        self.data_blocks: torch.Tensor | None = None
        server_thread_port = self._init_server()

        # Initialize small shared client batch_states between main and process wrapper
        self.client_batch_state = ClientBatchState()

        # 1. Lock-free ACK flags in shared CPU memory (initialized to -1)
        self.ack_flags = torch.full((max(send_layer_list)+1,), -1, dtype=torch.int32, device="cpu").share_memory_()
        
        # 2. Direct multiprocessing Pipe for state transfer (bypasses background Queue thread)
        self.state_pipe_recv, self.state_pipe_send = mp_ctx.Pipe(duplex=False)

        # Fork another process to overlap with main process (vllm engine)
        self.process_session = ClientProcessSession(
            client_batch_state=self.client_batch_state,
            send_layer_list=send_layer_list,
            server_thread_port=server_thread_port,
            state_pipe_recv=self.state_pipe_recv,
            state_pipe_send=self.state_pipe_send,
            ack_flags=self.ack_flags,
            force_wait_sync=force_wait_sync,
        )

        self.current_stream = torch.cuda.current_stream()

        self.iteration = 0
        self.max_iterations = 1000000
        self.is_first_hit = True  # for debugging (only print during first hit layer)
        self.layer_gpu_events = [
            torch.cuda.Event(enable_timing=False)
            for _ in range(self.num_layers)
        ]

        self.decode_only = False

    def _init_server(self):
        """called inside __init__ after creating self.bsr_buffer
        used to register ipc wrappers at server side

        req = {
            "cmd": "REGISTER_RECEIVER",
            "ipc_wrapper": self._get_bsr_ipc_wrapper() 
        }
        resp = {
            "status": "success",
            "server_thread_port": int,
            "data_blocks_ipc": CudaIPCWrapper
        }
        """
        req = {
            "cmd": "REGISTER_RECEIVER",
            "ipc_wrapper": self.bsr_buffer.get_ipc_wrapper()
        }
        self.comm.send(req, self.server_main_port)
        resp = self.comm.recv()
        server_thread_port = resp["server_thread_port"]
        self.data_blocks = resp["data_blocks_ipc"].to_tensor(0)
        assert self.data_blocks.device == torch.device(self.device), f"Expected data_blocks on device {self.device}, but got {self.data_blocks.device}"        
        return server_thread_port

    @property
    def is_sender(self):
        return False

    def maybe_record_event(self, layer_idx):
        if not self.decode_only:
            self.layer_gpu_events[layer_idx].record(self.current_stream)

    def retrieve_blocks(
        self,
        scheduler_output: SchedulerOutput
    ):
        self.iteration = (self.iteration + 1) % self.max_iterations
        self.is_first_hit = True
        req = {
            "cmd": "RETRIEVE_HASH_BLOCKS",
            "iteration": self.iteration,
            "scheduler_output": scheduler_output,
        }
        self.process_session.put(req)
        self.flash_attn_block_table = None
        self.bsr_attn_block_table = None
        num_scheduled_tokens: dict[str, int] = scheduler_output.num_scheduled_tokens
        self.decode_only = len(num_scheduled_tokens) > 0 and max(num_scheduled_tokens.values()) == 1

    def _should_ignore_layer(self, layer_idx: int) -> bool:
        return layer_idx < self.num_ignored_layers or layer_idx == self.num_layers - 1

    def print_hit_miss_status(self) -> None:
        """
        Print the hit and miss intervals for each request in the current batch.
        Example output: req_1, miss: [0, 1], [7, 9], hit: [2, 6]
        
        Note: This function pulls GPU tensors to CPU for debugging, 
        which will cause a device synchronization. Use only for debugging.
        """
        num_hit_tokens = self.client_batch_state.num_hit_tokens
        num_miss_tokens = self.client_batch_state.num_miss_tokens
        total_tokens = num_hit_tokens + num_miss_tokens

        if total_tokens == 0:
            print("No tokens in current batch to print.")
            return

        # 1. 从 GPU 拉取 hit_indices 到 CPU 并转为 list
        if num_hit_tokens > 0:
            # 取出有效部分的 indices
            hit_indices = self.bsr_buffer.hit_indices[:num_hit_tokens].cpu().tolist()
        else:
            hit_indices = []

        # 2. 还原出 current batch 全局的 hit_mask
        hit_mask = [False] * total_tokens
        for idx in hit_indices:
            hit_mask[idx] = True

        # 3. 按 request 遍历并划分区间
        offset = 0
        for req_id, num_tokens in self.client_batch_state.current_batch_layout:
            cur_hit_mask = hit_mask[offset : offset + num_tokens]
            
            miss_intervals = []
            hit_intervals = []
            
            start_miss = None
            start_hit = None
            
            for i, is_hit in enumerate(cur_hit_mask):
                if is_hit:
                    if start_miss is not None:
                        miss_intervals.append(f"[{start_miss}, {i - 1}]")
                        start_miss = None
                    if start_hit is None:
                        start_hit = i
                else:
                    if start_hit is not None:
                        hit_intervals.append(f"[{start_hit}, {i - 1}]")
                        start_hit = None
                    if start_miss is None:
                        start_miss = i
                        
            # 处理收尾闭合
            if start_miss is not None:
                miss_intervals.append(f"[{start_miss}, {num_tokens - 1}]")
            if start_hit is not None:
                hit_intervals.append(f"[{start_hit}, {num_tokens - 1}]")
                
            miss_str = ", ".join(miss_intervals) if miss_intervals else "None"
            hit_str = ", ".join(hit_intervals) if hit_intervals else "None"
            
            print(f"{req_id}, miss: {miss_str}, hit: {hit_str}", flush=True)
            
            offset += num_tokens

    def flash_attn_or_bsr(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        alibi_slopes,
        window_size,
        block_table,
        softcap,
        scheduler_metadata,
        fa_version,
        q_descale,
        k_descale,
        v_descale,
        num_splits,
        s_aux,
    ):
        # print(f"layer_idx: {layer_idx}, q: {q.shape}")

        """Called inside vllm bsr_attn"""
        if self._should_ignore_layer(layer_idx) or self.decode_only:
            # directly use flash attention
            flash_attn_varlen_func(
                q=q,
                k=key_cache,
                v=value_cache,
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

        with prof_marker(f"retrieve_sync_layer_{layer_idx}"):
            hit_cache = self.process_session.get_or_fallback(
                self.iteration,
                self.recv_to_send_layer_idx[layer_idx],
                self.layer_gpu_events[layer_idx - 1],
            )

        nnz = 0
        density = 0
        if hit_cache and self.client_batch_state.num_hit_tokens > 0:
            send_layer_idx = self.recv_to_send_layer_idx[layer_idx]
            num_reqs = self.client_batch_state.num_hit_reqs
            hit_trapezoid_sizes = self.client_batch_state.hit_trapezoid_sizes
            
            nnz = self.bsr_buffer.layers[send_layer_idx]["cu_col_indices_cpu"][num_reqs].item()
            density = int(nnz / (hit_trapezoid_sizes * self.weights[layer_idx].shape[0]) * 100)

            if os.environ.get("SKIP_DENSITY_CHECK", "0") == "0":
                # will always try to reuse by setting SKIP_DENSITY_CHECK=1(especially for accuracy checks) 
                # density > 10 means it's not efficient enough to run bsr attention
                if density > 10:
                    hit_cache = False
            else:
                print(
                    f"layer {layer_idx}, density = {nnz} / ({hit_trapezoid_sizes} * {self.weights[layer_idx].shape[0]}) = {density}%"
                    , flush=True
                )

        if not hit_cache or self.client_batch_state.num_hit_tokens == 0:
            flash_attn_varlen_func(
                q=q,
                k=key_cache,
                v=value_cache,
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

        if self.client_batch_state.num_miss_tokens > 0:
            num_reqs = self.client_batch_state.num_flash_attn_reqs
            num_tokens = self.client_batch_state.num_miss_tokens
            token_indices = self.bsr_buffer.miss_indices[:num_tokens]
            flash_attn_req_indices = self.bsr_buffer.flash_attn_req_indices[:num_reqs]
            if self.flash_attn_block_table is None:
                self.flash_attn_block_table = block_table[flash_attn_req_indices]
            
            # print(f"token_indices: {token_indices.max().item()}, q.shape: {q.shape}", flush=True)

            flash_attn_varlen_func(
                q=q[token_indices],
                k=key_cache,
                v=value_cache,
                out=self.output_buffer[:num_tokens], # Write directly into the masked output slices
                cu_seqlens_q=self.bsr_buffer.flash_attn_cu_seqlens_q[:num_reqs + 1],
                max_seqlen_q=self.client_batch_state.flash_attn_max_seqlen_q,
                seqused_k=self.bsr_buffer.flash_attn_seqused_k[:num_reqs],
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                alibi_slopes=alibi_slopes,
                window_size=window_size,
                block_table=self.flash_attn_block_table,
                softcap=softcap,
                scheduler_metadata=scheduler_metadata,
                fa_version=fa_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=num_splits,
                s_aux=s_aux,
            )
            out[token_indices] = self.output_buffer[:num_tokens]

        # perform bsr attention on remaining tokens
        if self.client_batch_state.num_hit_tokens > 0:
            num_reqs = self.client_batch_state.num_hit_reqs
            num_tokens = self.client_batch_state.num_hit_tokens
            token_indices = self.bsr_buffer.hit_indices[:num_tokens]
            num_packed_block_rows = self.client_batch_state.num_packed_block_rows
            hit_req_indices = self.bsr_buffer.hit_req_indices[:num_reqs]
            hit_trapezoid_sizes = self.client_batch_state.hit_trapezoid_sizes

            bsr_batch_offsets = self.bsr_buffer.bsr_batch_offsets[:num_reqs]
            packed_row_block_pid_to_seq_id = self.bsr_buffer.packed_row_block_pid_to_seq_id[:num_packed_block_rows]
            packed_row_block_pid_to_row_block_pid = self.bsr_buffer.packed_row_block_pid_to_row_block_pid[:num_packed_block_rows]
            send_layer_idx = self.recv_to_send_layer_idx[layer_idx]
            
            # after retrieve_sync, the bsr_buffer is ready            
            block_indices = self.bsr_buffer.layers[send_layer_idx]["block_indices"]
            crow = self.bsr_buffer.layers[send_layer_idx]["crow"]
            col = self.bsr_buffer.layers[send_layer_idx]["col"]
            cu_crow_indices = self.bsr_buffer.layers[send_layer_idx]["cu_crow_indices"][:num_reqs + 1]
            cu_col_indices = self.bsr_buffer.layers[send_layer_idx]["cu_col_indices"][:num_reqs + 1]

            if self.bsr_attn_block_table is None:
                self.bsr_attn_block_table = block_table[hit_req_indices]
            
            # avg_row_nnz = nnz // max(1, num_packed_block_rows)
            # avg_row_nnz_bucket = 1 << max(0, math.ceil(math.log2(max(1, avg_row_nnz))))
            # print(
            #     f"layer {layer_idx}, density = {nnz} / ({hit_trapezoid_sizes} * {self.weights[layer_idx].shape[0]}) = {density}%"
            #     f", avg_row_nnz = {avg_row_nnz}, avg_row_nnz_bucket = {avg_row_nnz_bucket}"
            #     , flush=True
            # )

            with prof_marker(f"bsr_varlen_page_kernel_layer_{layer_idx}"):
                bsr_varlen_page_triton(
                    data_blocks=self.data_blocks,
                    block_indices=block_indices,
                    crow=crow,
                    col=col[:nnz],
                    cu_crow_indices=cu_crow_indices,
                    cu_col_indices=cu_col_indices,
                    bsr_batch_offsets=bsr_batch_offsets,
                    num_packed_block_rows=num_packed_block_rows,
                    packed_row_block_pid_to_seq_id=packed_row_block_pid_to_seq_id,
                    packed_row_block_pid_to_row_block_pid=packed_row_block_pid_to_row_block_pid,
                    weights=self.weights[layer_idx],
                    block_table=self.bsr_attn_block_table,
                    value_cache=value_cache,
                    output=out,
                    density=density,
                )