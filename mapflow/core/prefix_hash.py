import hashlib
import struct
from functools import lru_cache

@lru_cache(maxsize=1024)
def compute_block_hash(prev_hash: str, token_ids: tuple[int]) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(b'_')

    if token_ids:
        h.update(struct.pack(f'{len(token_ids)}I', *token_ids))

    return h.hexdigest()
