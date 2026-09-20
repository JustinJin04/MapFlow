# SPDX-License-Identifier: Apache-2.0
"""vLLM backend for MInference-style long-context sparse prefill."""

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.minference_kernel import minference_varlen_func
from vllm.v1.attention.backends.utils import KVCacheLayoutType

logger = init_logger(__name__)


def _read_non_negative_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


class MInferenceAttentionBackend(FlashAttentionBackend):
    """FlashAttention-compatible backend with sparse long-prefill dispatch."""

    supported_dtypes = [torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        return "MINFERENCE"

    @staticmethod
    def get_impl_cls() -> type["MInferenceAttentionImpl"]:
        return MInferenceAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return FlashAttentionMetadataBuilder

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType:
        return "NHD"


class MInferenceAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.minference_block_size = _read_non_negative_int(
            "VLLM_MINFERENCE_BLOCK_SIZE", 64
        )
        if self.minference_block_size not in (32, 64, 128):
            raise ValueError("VLLM_MINFERENCE_BLOCK_SIZE must be 32, 64, or 128")
        self.minference_top_k = _read_non_negative_int("VLLM_MINFERENCE_TOP_K", 4)
        self.minference_local_blocks = _read_non_negative_int(
            "VLLM_MINFERENCE_LOCAL_BLOCKS", 4
        )
        self.minference_sink_blocks = _read_non_negative_int(
            "VLLM_MINFERENCE_SINK_BLOCKS", 1
        )
        self.minference_min_seq_len = _read_non_negative_int(
            "VLLM_MINFERENCE_MIN_SEQ_LEN", 4096
        )
        if not (
            self.minference_top_k
            or self.minference_local_blocks
            or self.minference_sink_blocks
        ):
            raise ValueError(
                "at least one MInference block budget must be greater than zero"
            )
        logger.info_once(
            "Using MInference backend (block=%d, top_k=%d, local=%d, "
            "sink=%d, min_seq_len=%d)",
            self.minference_block_size,
            self.minference_top_k,
            self.minference_local_blocks,
            self.minference_sink_blocks,
            self.minference_min_seq_len,
            scope="local",
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        # The sparse baseline targets decoder prefill only. FlashAttention owns
        # profiling, encoder/cross attention, cascade, DCP and quantized paths.
        if (
            attn_metadata is None
            or self.attn_type != AttentionType.DECODER
            or attn_metadata.use_cascade
            or self.dcp_world_size > 1
            or self.kv_cache_dtype.startswith("fp8")
            or attn_metadata.max_query_len < self.minference_min_seq_len
            or attn_metadata.query_start_loc.shape[0] != 2
            or output_scale is not None
            or output_block_scale is not None
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # print(f"forward with minference.")
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        cu_seqlens_q = attn_metadata.query_start_loc
        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
        sliding_window = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        minference_varlen_func(
            q=query[:num_actual_tokens],
            k_cache=key_cache,
            v_cache=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
            s_aux=self.sinks,
            block_size=self.minference_block_size,
            top_k=self.minference_top_k,
            local_blocks=self.minference_local_blocks,
            sink_blocks=self.minference_sink_blocks,
            min_seq_len=self.minference_min_seq_len,
        )
        return output
