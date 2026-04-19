import torch
import pickle

class CudaIPCWrapper:
    def __init__(self, tensor: torch.Tensor):
        assert tensor.is_contiguous(), "Tensor must be contiguous"
        
        storage = tensor.untyped_storage()
        self.handle = storage._share_cuda_()
        
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device_idx = tensor.device.index
        
        self.offset = tensor.storage_offset()
        self.numel = tensor.numel() 

    def to_tensor(self, device_idx=None):
        if device_idx is None:
            device_idx = self.device_idx
        storage = torch.UntypedStorage._new_shared_cuda(device_idx, *self.handle[1:])
        t = torch.empty((0,), device=f"cuda:{device_idx}", dtype=self.dtype)
        t.set_(storage, storage_offset=self.offset, size=self.shape)
        return t

    def clone(self, device, non_blocking=True):
        new_tensor = torch.empty(self.shape, dtype=self.dtype, device=device)
        new_tensor.copy_(self.to_tensor(), non_blocking=non_blocking)
        return new_tensor

    @staticmethod
    def serialize(obj):
        return pickle.dumps(obj)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)

class CudaIpcEventWrapper:
    def __init__(self, event: torch.cuda.Event):
        self.ipc_handle = event.ipc_handle()
        self.device_idx = torch.cuda.current_device() 

    def reconstruct_event(self):
        return torch.cuda.Event.from_ipc_handle(self.device_idx, self.ipc_handle)


class CpuIPCWrapper:
    def __init__(self, tensor: torch.Tensor):
        assert tensor.is_contiguous(), "Tensor must be contiguous"
        assert not tensor.is_cuda, "Tensor must be on CPU"

        if not tensor.is_shared():
            tensor = tensor.share_memory_()

        storage = tensor.untyped_storage()
        
        self.handle = storage._share_filename_cpu_()
        
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.offset = tensor.storage_offset()
        self.numel = tensor.numel()

    def to_tensor(self):
        storage = torch.UntypedStorage._new_shared_filename_cpu(*self.handle)
        
        t = torch.tensor([], dtype=self.dtype)
        t.set_(storage, storage_offset=self.offset, size=self.shape)
        return t

    def clone(self, device, non_blocking=True):
        t = self.to_tensor()
        return t.to(device, non_blocking=non_blocking)

    @staticmethod
    def serialize(obj):
        return pickle.dumps(obj)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)