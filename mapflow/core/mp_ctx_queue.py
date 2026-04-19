import torch
from dataclasses import dataclass
# import torch.multiprocessing as mp


class GpuIndexBuffer:
    def __init__(self, size_numel: int = 100 * 1024 * 1024, device: str = "cuda"):
        self.buffer = torch.zeros(size_numel, dtype=torch.int32, device=device)
        self.capacity = size_numel
        self.write_ptr = 0
        self.device = device

    def write(self, tensor: torch.Tensor) -> tuple[int, int]:
        if tensor.dtype != torch.int32:
            raise ValueError(f"GpuIndexBuffer only supports int32 currently, got {tensor.dtype}")

        flat = tensor.reshape(-1).contiguous()
        numel = flat.numel()
        assert numel <= self.capacity, f"numel: {numel}, capacity: {self.capacity}"

        start = self.write_ptr
        end = start + numel

        if end > self.capacity:
            # ring buffer wrap-around
            start = 0
            end = numel
            self.write_ptr = end
        else:
            self.write_ptr = end

        self.buffer[start:end].copy_(flat, non_blocking=True)
        return start, numel

    def read(self, start: int, numel: int, shape: torch.Size) -> torch.Tensor:
        return self.buffer[start:start + numel].view(shape)

    def get_handle(self):
        return self.buffer


@dataclass
class TensorBufferMeta:
    start: int
    numel: int
    shape: torch.Size


class BiDirQueue:

    def __init__(
        self,
        mp_ctx,
        device: str = "cuda",
        req_buffer_size_numel: int = 100 * 1024 * 1024,
        resp_buffer_size_numel: int = 100 * 1024 * 1024,
    ):
        self.req_recv, self.req_send = mp_ctx.Pipe(duplex=False)
        self.resp_recv, self.resp_send = mp_ctx.Pipe(duplex=False)
        self.resp_1_recv, self.resp_1_send = mp_ctx.Pipe(duplex=False)

        self.req_buffer = GpuIndexBuffer(
            size_numel=req_buffer_size_numel,
            device=device,
        )
        self.resp_buffer = GpuIndexBuffer(
            size_numel=resp_buffer_size_numel,
            device=device,
        )
        self.device = device

    # -------------------------
    # request direction helpers
    # -------------------------
    def wrap_req_tensor(self, tensor: torch.Tensor) -> TensorBufferMeta:
        shape = tensor.shape
        start, numel = self.req_buffer.write(tensor)
        return TensorBufferMeta(start=start, numel=numel, shape=shape)

    def read_req_tensor(self, meta: TensorBufferMeta) -> torch.Tensor:
        return self.req_buffer.read(meta.start, meta.numel, meta.shape)

    # --------------------------
    # response direction helpers
    # --------------------------
    def wrap_resp_tensor(self, tensor: torch.Tensor) -> TensorBufferMeta:
        shape = tensor.shape
        start, numel = self.resp_buffer.write(tensor)
        return TensorBufferMeta(start=start, numel=numel, shape=shape)

    def read_resp_tensor(self, meta: TensorBufferMeta) -> torch.Tensor:
        return self.resp_buffer.read(meta.start, meta.numel, meta.shape)


    def put_req(self, req):
        self.req_send.send(req)

    def get_req(self):
        return self.req_recv.recv()

    def put_resp(self, resp):
        self.resp_send.send(resp)

    def get_resp(self):
        return self.resp_recv.recv()

    def put_resp_1(self, resp):
        self.resp_1_send.send(resp)

    def get_resp_1(self):
        return self.resp_1_recv.recv()

    def req_empty(self):
        return not self.req_recv.poll()

    # -------------------------
    # optional: forbid old APIs
    # -------------------------
    def tensor_wrapper(self, tensor: torch.Tensor):
        raise RuntimeError(
            "Deprecated API: use wrap_req_tensor() or wrap_resp_tensor() explicitly."
        )

    def read_tensor(self, meta: TensorBufferMeta):
        raise RuntimeError(
            "Deprecated API: use read_req_tensor() or read_resp_tensor() explicitly."
        )