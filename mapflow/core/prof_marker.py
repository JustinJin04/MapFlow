import torch
import contextlib
from typing import Optional
import nvtx

@contextlib.contextmanager
def prof_marker(name: str, color: Optional[str] = None):
    with nvtx.annotate(name, color=color):
        yield