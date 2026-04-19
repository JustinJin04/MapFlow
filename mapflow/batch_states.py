import torch
import queue
from typing import List, Dict, Tuple, Optional
from vllm.v1.core.sched.output import SchedulerOutput
from mapflow.core.prefix_hash import compute_block_hash
from mapflow.core import prof_marker

class RequestState:
    def __init__(
        self, 
        req_id: str, 
        prompt_token_ids: list[int],
        block_size: int = 64, 
    ):
        self.req_id = req_id
        self.prompt_token_ids = prompt_token_ids
        self.block_size = block_size

        self.tokens: list[int] = [] 
        self.block_hashes: list[str] = [] 
        self.last_hash_str: str = "0" 
        self.pending_tokens: list[int] = []        

        self.prefix_cached_len: int = 0
        self.num_last_pending_tokens: int = 0  
        self.num_new_tokens: int = 0
        self.num_tokens_need_to_drop = 0
        
        # 删掉 self.thread_pool 的初始化

    def set_prefix_cached_tokens(self, prefix_cached_tokens: list[int]) -> None:
        self.prefix_cached_len = len(prefix_cached_tokens)
        self.tokens.extend(prefix_cached_tokens)
        
        all_pending = self.pending_tokens + prefix_cached_tokens
        num_tokens = len(all_pending)
        num_full_blocks = num_tokens // self.block_size
        if num_full_blocks > 0:
            last_hash = self.last_hash_str
            new_hashes = []
            for i in range(num_full_blocks):
                start_idx = i * self.block_size
                block_tokens = all_pending[start_idx : start_idx + self.block_size]                
                last_hash = compute_block_hash(last_hash, tuple(block_tokens))
                new_hashes.append(last_hash)
            
            self.block_hashes.extend(new_hashes)
            self.last_hash_str = last_hash            
            remainder = num_tokens % self.block_size
            self.pending_tokens = all_pending[-remainder:] if remainder > 0 else []
        else:
            self.pending_tokens = all_pending

    def update_tokens(self, new_tokens: list[int]) -> None:
        """
        Called during STORE_SCHEDULER_OUTPUT.
        Updates token history and pre-calculates hashes for any newly formed blocks.
        """
        self.num_new_tokens = len(new_tokens)
        self.num_last_pending_tokens = len(self.pending_tokens)
        self.tokens.extend(new_tokens)
        
        all_pending = self.pending_tokens + new_tokens
        num_tokens = len(all_pending)
        num_full_blocks = num_tokens // self.block_size
        if num_full_blocks > 0:
            last_hash = self.last_hash_str
            new_hashes = []
            for i in range(num_full_blocks):
                start_idx = i * self.block_size
                block_tokens = all_pending[start_idx : start_idx + self.block_size]                
                last_hash = compute_block_hash(last_hash, tuple(block_tokens))
                new_hashes.append(last_hash)
            
            self.block_hashes.extend(new_hashes)
            self.last_hash_str = last_hash            
            remainder = num_tokens % self.block_size
            self.pending_tokens = all_pending[-remainder:] if remainder > 0 else []
        else:
            self.pending_tokens = all_pending

        # update num_tokens_need_to_drop
        # to decide how many tokens need to be dropped for the incomming query
        # prefix_cached: 48, block_size: 64
        # - query has 16+ tokens, then drop first 16 tokens
        # - query has <=16 tokens, drop all tokens and move prefix_cached towards
        if self.prefix_cached_len % self.block_size == 0:
            self.num_tokens_need_to_drop = 0
        else:
            right_aligned_prefix_cached_len = (self.prefix_cached_len + self.block_size - 1) // self.block_size * self.block_size
            if self.prefix_cached_len + self.num_new_tokens <= right_aligned_prefix_cached_len:
                self.prefix_cached_len += self.num_new_tokens
                self.num_tokens_need_to_drop = self.num_new_tokens
            else:
                self.num_tokens_need_to_drop = right_aligned_prefix_cached_len - self.prefix_cached_len
                self.prefix_cached_len += self.num_tokens_need_to_drop


    def preempted(self):
        self.prompt_token_ids = self.tokens.copy()
        self.tokens = []
        self.block_hashes = []
        self.last_hash_str = "0"
        self.pending_tokens = []
        self.prefix_cached_len = 0
        self.num_last_pending_tokens = 0
        self.num_new_tokens = 0
        self.num_tokens_need_to_drop = 0
        # self.tokens = []
        # self.block_hashes = []
        # self.last_hash_str = "0"
        # self.num_last_pending_tokens = 0
        # self.num_new_tokens = 0
        # self.pending_tokens = []

    @property
    def num_computed_tokens(self) -> int:
        return len(self.tokens) - self.num_new_tokens

# def count_continuous_ones(mask: torch.Tensor, zero_tensor: torch.Tensor):
#     padded_mask = torch.cat([zero_tensor, mask.int(), zero_tensor])  # Pad with False on both sides
#     diff = padded_mask[1:] - padded_mask[:-1]
#     starts = (diff == 1).nonzero(as_tuple=True)[0]
#     ends = (diff == -1).nonzero(as_tuple=True)[0]
#     return ends - starts

def get_continuous_ones_info(mask: torch.Tensor, n: int, zero_tensor: torch.Tensor):
    # 1. 复用你的 diff 逻辑找到边界
    # 注意：padded_mask 比原 mask 长了 2 (左右各补一个 0)
    padded_mask = torch.cat([zero_tensor, mask.int(), zero_tensor])
    diff = padded_mask[1:] - padded_mask[:-1]
    
    # starts: 1 开始的位置；ends: 1 结束后的那个 0 的位置
    starts = (diff == 1).nonzero(as_tuple=True)[0]
    ends = (diff == -1).nonzero(as_tuple=True)[0]
    
    # 2. 关键点：在原 mask 坐标系下，连续 1 的最后一个元素的索引
    # 因为 ends 指向的是补零后的“1 之后的 0”，
    # 所以在原 mask 坐标系中，结束索引 = ends - 1
    last_one_indices = ends - 1
    
    # 3. 映射到 position 坐标系
    # 映射公式：pos = n - (L - 1 - index)
    L = mask.shape[0]
    positions = n - (L - 1 - last_one_indices)
    
    # 4. 计算长度（你原有的逻辑）
    lengths = ends - starts
    
    return lengths, positions

def find_consecutive_ones(mask: torch.Tensor, zero_tensor: torch.Tensor):
    shifted_mask = torch.cat([zero_tensor, mask.int()])
    diff = shifted_mask[1:] - shifted_mask[:-1]
    return (diff == 1).nonzero(as_tuple=True)[0]
    

class BatchState:
    def __init__(
        self,
        max_num_reqs: int,
        block_size: int = 64,
        device: str = "cuda",
    ):
        self.max_num_reqs = max_num_reqs
        self.block_size = block_size
        self.device = device

        self.slot_mapping: Dict[str, int] = {}  # req_id -> slot idx
        self.available_slots = queue.Queue()
        for i in range(max_num_reqs):
            self.available_slots.put(i)
        self.req_states: Dict[str, RequestState] = {}  # req_id -> RequestState

        # following are all updated in self.update:
        self.num_scheduled_tokens: List[int] = []  # [2, 5, 3]
        self.current_batch_layout: List[Tuple[str, int]] = []  # [("req_0", 2), ("req_1", 5), ("req_2", 3)]

        # hash related (receiver side)
        self.aligned_hashes: List[List[str]] = []  # aligned list of hashes for current batch
        self.flattened_hashes: List[str] = []  # flattened list of hashes for current batch
        self.req_hash_to_token_indices: Dict[Tuple[str, str], torch.Tensor] = {} # (req_id, hash_str) -> token indices tensor, notice that same hash str may appear in different reqs
        self.hit_mask: Optional[torch.Tensor] = None  # [num_batch_tokens], bool tensor
        self.miss_mask: Optional[torch.Tensor] = None  # [num_batch_tokens], bool tensor
        # self.hash_mask: Optional[torch.Tensor] = None  # [num_batch_tokens], bool tensor
        self.num_hit_tokens: int = 0
        self.hash_req_ids: List[str] = []  # req_ids that have at least one hash block in current batch
        self.hit_req_ids: List[str] = []  # req_ids that have at least one hit block in current batch
        self.flash_attn_req_ids: List[str] = []  # req_ids that have at least one miss token in current batch
        self.cu_block_rows: Optional[torch.Tensor] = None  # [num_hit_reqs+1], int tensor
        self.hit_slots: Optional[torch.Tensor] = None  # list of slots that have hits
        self.hit_indices: Optional[torch.Tensor] = None  # list of indices that have hits
        self.miss_indices: Optional[torch.Tensor] = None  # list of indices that have misses
        self.flash_attn_req_indices: Optional[torch.Tensor] = None  # list of req indices that have at least one miss token
        self.hit_req_indices: Optional[torch.Tensor] = None  # list of req indices that have at least one hit token
        self.flash_attn_seqused_k: Optional[torch.Tensor] = None

        # flash attention related
        self.flash_attn_cu_seqlens_q: Optional[torch.Tensor] = None
        self.flash_attn_max_seqlen_q: Optional[int] = None
        # self.block_table_indices: Optional[torch.Tensor] = None

        # bsr attention related
        # self.num_hit_reqs = 0
        # Example: for [2*64+2, 5*64+5, 3*64+3]
        # Case 1: hit_list: [[1, 1], [0, 0, 0, 0, 0], [1, 1, 0]]
        # bsr_batch_offsets = [0, 7*64+7]  # [num_hit_reqs], int tensor
        # Case 2: hit_list: [[0, 0], [1, 0, 0, 0, 0], [1, 1, 0]]
        # bsr_batch_offsets = [2*64+2, 7*64+7]  # [num_hit_reqs], int tensor
        self.bsr_batch_offsets: Optional[torch.Tensor] = None  # [num_hit_reqs], int tensor
        self.num_packed_block_rows = 0  # 10
        self.packed_row_block_pid_to_seq_id: Optional[torch.Tensor] = None  # [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        self.packed_row_block_pid_to_row_block_pid: Optional[torch.Tensor] = None # [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]

        # pre-allocated tensors
        self.arange_tensor = torch.arange(0, 40960, device=self.device)  # for indexing, assuming max_batch_tokens won't exceed 40960
        self.zero_tensor = torch.zeros((1,), dtype=torch.int32, device=self.device)  # for counting continuous ones

        # density related states
        # density = nnz / (hit_trapezoid_sizes * m)
        self.hit_trapezoid_sizes: int = 0

    def update(self, scheduled_reqs: list[dict[str, list[int]]], finished_reqs: list[str], hanged_reqs: list[str], preempted_reqs: list[str]):
        """
        Update batch state with newly scheduled requests.
        Allocate slots for new requests and update existing ones.
        """
        # Handle finished requests (remove from input_batch and requests)
        for req_id in finished_reqs:
            assert req_id in self.slot_mapping, f"Finished req_id {req_id} not in current requests: {list(self.slot_mapping.keys())}"
            slot = self.slot_mapping.pop(req_id)
            self.available_slots.put(slot)
            self.req_states.pop(req_id, None)

        # Handle preempted requests (remove from input_batch and slots but keep in requests)
        for req_id in preempted_reqs:
            assert req_id in self.req_states, f"Preempted req_id {req_id} not in current requests: {list(self.req_states.keys())}"
            self.req_states[req_id].preempted()
            if req_id in finished_reqs:
                continue
            assert req_id in self.slot_mapping, f"Unscheduled req_id {req_id} not in current requests: {list(self.slot_mapping.keys())}"
            slot = self.slot_mapping.pop(req_id)
            self.available_slots.put(slot)
            # Note: we do NOT remove from self.req_states, as we need to keep the token history

        # Handle current scheduled batch of requests
        self.num_scheduled_tokens = []
        self.current_batch_layout = []
        for req in scheduled_reqs:
            req_id = req["req_id"]
            new_tokens = req["tokens"]
            
            if req_id not in self.slot_mapping:
                if self.available_slots.empty():
                    raise RuntimeError("Max requests limit reached!")
                slot = self.available_slots.get()
                self.slot_mapping[req_id] = slot
            if req_id not in self.req_states:
                self.req_states[req_id] = RequestState(req_id, self.block_size)

            # Update Logic State (Tokens & Hashes)
            self.req_states[req_id].update_tokens(new_tokens)
            self.current_batch_layout.append((req_id, len(new_tokens)))
            self.num_scheduled_tokens.append(len(new_tokens))

        # Update aligned hashes(list[list[str]]) for current batch
        # And initialize hit mask(later will be further updated by self.update_hits)
        self.aligned_hashes = []
        self.flattened_hashes = []
        self.req_hash_to_token_indices = {}
        # self.hash_mask = torch.zeros((self.num_batch_tokens,), dtype=torch.bool, device=self.device)
        self.hash_req_ids = []
        current_batch_offset = 0
        for req_id in self.req_ids:
            req_state = self.req_states[req_id]
            if req_state.num_new_tokens == 1:
                # skip decode tokens
                current_batch_offset += 1
                continue
            req_state = self.req_states[req_id]
            global_start_idx = req_state.num_computed_tokens
            global_end_idx = global_start_idx + req_state.num_new_tokens
            first_full_block_idx = (global_start_idx + req_state.block_size - 1) // req_state.block_size
            last_full_block_idx = global_end_idx // req_state.block_size
            if first_full_block_idx < last_full_block_idx:
                valid_hashes = req_state.block_hashes[first_full_block_idx:last_full_block_idx]
                self.aligned_hashes.append(valid_hashes)
                self.flattened_hashes.extend(valid_hashes)
                chunk_offset_start = first_full_block_idx * req_state.block_size - global_start_idx
                chunk_offset_end = last_full_block_idx * req_state.block_size - global_start_idx
                for i, ha in enumerate(valid_hashes):
                    token_indices_start = current_batch_offset + chunk_offset_start + i * req_state.block_size
                    token_indices_end = token_indices_start + req_state.block_size
                    self.req_hash_to_token_indices[(req_id, ha)] = self.arange_tensor[token_indices_start:token_indices_end]
                # self.hash_mask[current_batch_offset + chunk_offset_start:current_batch_offset + chunk_offset_end] = True
                self.hash_req_ids.append(req_id)

            current_batch_offset += req_state.num_new_tokens

    def update_from_scheduler_output(self, scheduler_output: SchedulerOutput):
        """
        Update batch state with newly scheduled requests.
        Allocate slots for new requests and update existing ones.
        """

        # Handle finished requests
        for req_id in scheduler_output.finished_req_ids:
            slot = self.slot_mapping.pop(req_id)
            self.available_slots.put(slot)
            self.req_states.pop(req_id, None)

        # Handle preempted requests
        for req_id in scheduler_output.preempted_req_ids:
            self.req_states[req_id].preempted()
            if req_id in scheduler_output.finished_req_ids:
                continue
            slot = self.slot_mapping.pop(req_id)
            self.available_slots.put(slot)

        self.num_scheduled_tokens = []
        self.current_batch_layout = []
        

        scheduled_new_reqs = {req.req_id: req for req in scheduler_output.scheduled_new_reqs}
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        cached_req_id_to_idx = {
            rid: i for i, rid in enumerate(scheduled_cached_reqs.req_ids)
        } if scheduled_cached_reqs and scheduled_cached_reqs.req_ids else {}

        for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            if req_id not in self.slot_mapping:
                if self.available_slots.empty():
                    raise RuntimeError("Max requests limit reached!")
                slot = self.available_slots.get()
                self.slot_mapping[req_id] = slot

            self.current_batch_layout.append((req_id, num_scheduled_tokens))
            self.num_scheduled_tokens.append(num_scheduled_tokens)

            if req_id in scheduled_new_reqs:
                new_req = scheduled_new_reqs[req_id]
                prompt_token_ids = new_req.prompt_token_ids
                self.req_states[req_id] = RequestState(req_id, prompt_token_ids, self.block_size)
                
                num_computed_tokens = new_req.num_computed_tokens
                if num_computed_tokens > 0:
                    prefix_cached_tokens = prompt_token_ids[:num_computed_tokens]
                    self.req_states[req_id].set_prefix_cached_tokens(prefix_cached_tokens)
                    
                scheduled_tokens = prompt_token_ids[num_computed_tokens:num_computed_tokens+num_scheduled_tokens]
                self.req_states[req_id].update_tokens(scheduled_tokens)
            else:
                index = cached_req_id_to_idx.get(req_id)
                if req_id in scheduled_cached_reqs.resumed_req_ids:
                    # assert 0
                    prompt_token_ids = self.req_states[req_id].prompt_token_ids
                    num_computed_tokens = scheduled_cached_reqs.num_computed_tokens[index]
                    if num_computed_tokens > 0:
                        prefix_cached_tokens = prompt_token_ids[:num_computed_tokens]
                        self.req_states[req_id].set_prefix_cached_tokens(prefix_cached_tokens)
                    scheduled_tokens = prompt_token_ids[num_computed_tokens:num_computed_tokens+num_scheduled_tokens]
                    self.req_states[req_id].update_tokens(scheduled_tokens)
                else:
                    if num_scheduled_tokens == 1:
                        self.req_states[req_id].update_tokens([0])
                    else:
                        prompt_token_ids = self.req_states[req_id].prompt_token_ids
                        num_computed_tokens = scheduled_cached_reqs.num_computed_tokens[index]
                        scheduled_tokens = prompt_token_ids[num_computed_tokens:num_computed_tokens+num_scheduled_tokens]
                        self.req_states[req_id].update_tokens(scheduled_tokens)

        self.aligned_hashes = []
        self.flattened_hashes = []
        self.req_hash_to_token_indices = {}
        self.hash_req_ids = []
        current_batch_offset = 0

        for req_id, _ in self.current_batch_layout:
            req_state = self.req_states[req_id]
            
            if req_state.num_new_tokens == 1:
                current_batch_offset += 1
                continue
                
            global_start_idx = req_state.num_computed_tokens
            global_end_idx = global_start_idx + req_state.num_new_tokens
            first_full_block_idx = (global_start_idx + req_state.block_size - 1) // req_state.block_size
            last_full_block_idx = global_end_idx // req_state.block_size
            
            if first_full_block_idx < last_full_block_idx:
                valid_hashes = req_state.block_hashes[first_full_block_idx:last_full_block_idx]
                self.aligned_hashes.append(valid_hashes)
                self.flattened_hashes.extend(valid_hashes)
                chunk_offset_start = first_full_block_idx * req_state.block_size - global_start_idx
                
                for i, ha in enumerate(valid_hashes):
                    token_indices_start = current_batch_offset + chunk_offset_start + i * req_state.block_size
                    token_indices_end = token_indices_start + req_state.block_size
                    self.req_hash_to_token_indices[(req_id, ha)] = self.arange_tensor[token_indices_start:token_indices_end]
                
                self.hash_req_ids.append(req_id)

            current_batch_offset += req_state.num_new_tokens

    def get_req_slot(self, req_id: str) -> int:
        assert req_id in self.slot_mapping, f"req_id {req_id} not in current requests: {list(self.slot_mapping.keys())}"
        return self.slot_mapping[req_id]
    
    def get_req_state(self, req_id: str) -> RequestState:
        assert req_id in self.req_states, f"req_id {req_id} not in current requests: {list(self.req_states.keys())}"
        return self.req_states[req_id]

    def update_hits(self, hits: List[List[bool]]) -> None:
        """len(hits) == len(aligned_hashes)
        refer to _refine_mask_based_on_hits
        update self.hit_req_ids,self.hit_slots, self.hit_mask, self.num_hit_tokens, self.cu_block_rows
        """
        self.hit_req_ids = []
        self.hit_slots_list = []
        self.hit_mask = torch.zeros((self.num_batch_tokens,), dtype=torch.bool, device=self.device)
        self.num_hit_tokens = 0
        self.cu_block_rows_list = [0]
        block_true_tensor = torch.ones((self.block_size,), dtype=torch.bool, device=self.device)
        block_false_tensor = ~block_true_tensor
        assert len(hits) == len(self.aligned_hashes), f"len(hits) {len(hits)} != len(aligned_hashes) {len(self.aligned_hashes)}"
        self.hit_trapezoid_sizes = 0
        with prof_marker("hit_mask"):
            for req_id, hash_list, hit_list in zip(self.hash_req_ids, self.aligned_hashes, hits):
                assert len(hash_list) == len(hit_list), f"len(hash_list) {len(hash_list)} != len(hit_list) {len(hit_list)}"
                q_blocks = 0
                for h_str, is_hit in zip(hash_list, hit_list):
                    if is_hit:
                        token_indices = self.req_hash_to_token_indices[(req_id, h_str)]
                        self.num_hit_tokens += len(token_indices)
                        self.hit_mask[token_indices] = block_true_tensor
                        q_blocks += 1
                    else:
                        token_indices = self.req_hash_to_token_indices[(req_id, h_str)]
                        self.hit_mask[token_indices] = block_false_tensor
                if q_blocks > 0:
                    req_state = self.req_states[req_id]
                    ceil_blocks = (req_state.num_computed_tokens + self.block_size - 1) // self.block_size
                    self.hit_trapezoid_sizes += (2 * ceil_blocks + q_blocks + 1) * q_blocks // 2
                if any(hit_list):
                    self.hit_req_ids.append(req_id)
                    slot = self.get_req_slot(req_id)
                    self.hit_slots_list.append(slot)
                    self.cu_block_rows_list.append(self.num_hit_tokens // self.block_size)

        self.cu_block_rows = torch.tensor(self.cu_block_rows_list, dtype=torch.int32, device=self.device)
        self.hit_slots = torch.tensor(self.hit_slots_list, dtype=torch.int32, device=self.device)
        self.hit_indices = torch.nonzero(self.hit_mask, as_tuple=False).squeeze(1)
        self.miss_mask = ~self.hit_mask
        self.miss_indices = torch.nonzero(self.miss_mask, as_tuple=False).squeeze(1)

        # flash attn related states
        with prof_marker("flash_attn_states"):
            seq_off = 0
            flash_attn_cu_seqlens_q_list = [0]
            flash_attn_max_seqlen = 0
            self.flash_attn_req_ids = []
            flash_attn_req_indices_list = []
            flash_attn_seqused_k_list = []
            for i, (req_id, num_tokens) in enumerate(self.current_batch_layout):
                cur_miss_mask = self.miss_mask[seq_off:seq_off + num_tokens]
                seq_off += num_tokens
                # assert torch.all(torch.diff(cur_miss_mask.int()) >= 0), f"req_id: {req_id}, cur_miss_mask: {cur_miss_mask.tolist()}"
                cur_miss_indices = torch.nonzero(cur_miss_mask, as_tuple=False).squeeze(1)
                # if all the tokens are hits (all cur_miss_mask are false), skip
                if cur_miss_indices.numel() == 0:
                    continue
                # if [1, 1, 0, 0, 1, 1], we get [i, i] for miss_indices_list and [2, 4] for cu_seqlens_list
                seqlen = len(self.req_states[req_id].tokens)
                lengths, positions = get_continuous_ones_info(cur_miss_mask, seqlen, self.zero_tensor)
                self.flash_attn_req_ids.extend([req_id]*len(lengths))
                flash_attn_req_indices_list.extend([i]*len(lengths))
                flash_attn_cu_seqlens_q_list.extend((flash_attn_cu_seqlens_q_list[-1] + torch.cumsum(lengths, dim=0)).tolist())
                flash_attn_max_seqlen = max(flash_attn_max_seqlen, lengths.max().item())
                flash_attn_seqused_k_list.extend(positions.tolist())
            self.flash_attn_cu_seqlens_q = torch.tensor(flash_attn_cu_seqlens_q_list, dtype=torch.int32, device=self.device)
            self.flash_attn_max_seqlen_q = flash_attn_max_seqlen
            self.flash_attn_req_indices = torch.tensor(flash_attn_req_indices_list, dtype=torch.int32, device=self.device)
            self.flash_attn_seqused_k = torch.tensor(flash_attn_seqused_k_list, dtype=torch.int32, device=self.device)

        # bsr attn related states
        with prof_marker("bsr_attn_states"):
            bsr_batch_offsets_list = []
            acc_num_tokens = 0
            hit_req_indices_list = []
            for i, (req_id, num_tokens) in enumerate(self.current_batch_layout):
                if req_id in self.hit_req_ids:
                    hit_req_indices_list.append(i)
                    req_state = self.req_states[req_id]
                    pre_unaligned_tokens = (req_state.num_computed_tokens + self.block_size - 1) // self.block_size * self.block_size - req_state.num_computed_tokens
                    bsr_batch_offsets_list.append(acc_num_tokens + pre_unaligned_tokens)
                acc_num_tokens += num_tokens
            self.hit_req_indices = torch.tensor(hit_req_indices_list, dtype=torch.int32, device=self.device)
            # self.bsr_batch_offsets = find_consecutive_ones(self.hit_mask, self.zero_tensor)
            self.bsr_batch_offsets = torch.tensor(bsr_batch_offsets_list, dtype=torch.int32, device=self.device)

            self.num_packed_block_rows = self.cu_block_rows[-1].item()
            if self.num_packed_block_rows > 0:
                block_lens = self.cu_block_rows[1:] - self.cu_block_rows[:-1]
                self.packed_row_block_pid_to_seq_id = torch.repeat_interleave(
                    torch.arange(len(block_lens), device=self.device, dtype=torch.int32),
                    block_lens
                )
                self.packed_row_block_pid_to_row_block_pid = torch.cat([
                    torch.arange(bl, device=self.device, dtype=torch.int32)
                    for bl in block_lens
                ])

    def print_hit_miss_status(self) -> None:
        """
        Print the hit and miss intervals for each request in the current batch.
        Example output: req_1, miss: [0, 1], [7, 9], hit: [2, 6]
        """
        if self.hit_mask is None:
            print("Warning: hit_mask is not initialized yet.")
            return

        offset = 0
        for req_id, num_tokens in self.current_batch_layout:
            # 截取当前 request 的 hit_mask 并转为普通 list 方便处理
            cur_hit_mask = self.hit_mask[offset : offset + num_tokens].tolist()
            
            miss_intervals = []
            hit_intervals = []
            
            start_miss = None
            start_hit = None
            
            # 遍历当前 req 的 mask 提取连续区间
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
                        
            # 处理收尾逻辑（如果数组末尾的区间尚未闭合）
            if start_miss is not None:
                miss_intervals.append(f"[{start_miss}, {num_tokens - 1}]")
            if start_hit is not None:
                hit_intervals.append(f"[{start_hit}, {num_tokens - 1}]")
                
            # 格式化输出字符串
            miss_str = ", ".join(miss_intervals) if miss_intervals else "None"
            hit_str = ", ".join(hit_intervals) if hit_intervals else "None"
            
            print(f"{req_id}, miss: {miss_str}, hit: {hit_str}", flush=True)
            
            offset += num_tokens

    @property
    def num_miss_tokens(self) -> int:
        return self.num_batch_tokens - self.num_hit_tokens
    
    @property
    def num_hit_reqs(self) -> int:
        return len(self.hit_req_ids)
    
    @property
    def num_flash_attn_reqs(self) -> int:
        return len(self.flash_attn_req_ids)

    @property
    def req_ids(self) -> List[str]:
        """list of req_ids that has same order as current_batch_layout"""
        return [req_id for req_id, _ in self.current_batch_layout]

    @property
    def num_batch_tokens(self) -> int:
        return sum(num_tokens for _, num_tokens in self.current_batch_layout)

    @property
    def decode_only(self) -> bool:
        return max(num_tokens for _, num_tokens in self.current_batch_layout) == 1