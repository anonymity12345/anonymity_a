"""Inference attention with FA3 -> FA2 -> PyTorch SDPA dispatch.

The FlashAttention kernels bundled with vLLM are optional.  Keeping the
selection here also lets the native HF path use the same kernels as vLLM on
Hopper without requiring a separate flash-attn installation.
"""

from functools import lru_cache
import warnings

import torch
import torch.nn.functional as F


ATTENTION_NAME = "ex_omni_auto"
_failed_versions = {}


@lru_cache(maxsize=16)
def _flash_kernel(device_index):
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
            is_fa_version_supported,
        )
        for version in (3, 2):
            if (version not in _failed_versions.get(device_index, ())
                    and is_fa_version_supported(version, device_index)):
                return flash_attn_varlen_func, version
    except (ImportError, OSError, RuntimeError):
        pass
    # Native HF installations may have flash-attn without vLLM.  Its varlen
    # API uses a different positional order, so adapt it once at selection.
    capability = torch.cuda.get_device_capability(device_index)[0]
    if capability == 9 and 3 not in _failed_versions.get(device_index, ()):
        try:
            from flash_attn_interface import flash_attn_varlen_func as external_fa3
            return _external_varlen(external_fa3), 3
        except (ImportError, OSError, RuntimeError):
            pass
    if capability >= 8 and 2 not in _failed_versions.get(device_index, ()):
        try:
            from flash_attn import flash_attn_varlen_func as external_fa2
            return _external_varlen(external_fa2), 2
        except (ImportError, OSError, RuntimeError):
            pass
    return None, None


def _external_varlen(function):
    def call(q, k, v, max_q, cu_q, max_k, cu_k, *, softmax_scale,
             causal, window_size, fa_version):
        return function(q, k, v, cu_q, cu_k, max_q, max_k,
                        softmax_scale=softmax_scale, causal=causal,
                        window_size=window_size)
    return call


def selected_backend(device=None):
    """Report the kernel selected for an inference CUDA device."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        return "sdpa"
    index = device.index if device.index is not None else torch.cuda.current_device()
    _, version = _flash_kernel(index)
    return f"fa{version}" if version else "sdpa"


def attention(query, key, value, mask=None, *, causal=False, scale=None,
              dropout=0.0, sliding_window=None, prefer_flash=True):
    """Accept B,H,Q,D tensors and return B,H,Q,D, preserving GQA heads."""
    batch, heads, q_len, dim = query.shape
    k_len = key.shape[-2]
    use_flash = (prefer_flash and query.is_cuda and query.dtype in (torch.float16, torch.bfloat16)
                 and not torch.is_grad_enabled() and dropout == 0 and mask is None
                 and dim <= 256)
    if use_flash:
        index = query.device.index if query.device.index is not None else torch.cuda.current_device()
        q = query.transpose(1, 2).contiguous().view(batch * q_len, heads, dim)
        k = key.transpose(1, 2).contiguous().view(batch * k_len, key.shape[1], key.shape[-1])
        v = value.transpose(1, 2).contiguous().view(batch * k_len, value.shape[1], value.shape[-1])
        cu_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * q_len
        cu_k = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * k_len
        window = (sliding_window - 1, 0) if sliding_window else (-1, -1)
        attempted = set()
        for _ in range(2):
            kernel, version = _flash_kernel(index)
            if kernel is None or version in attempted:
                break
            attempted.add(version)
            try:
                result = kernel(q, k, v, q_len, cu_q, k_len, cu_k,
                                softmax_scale=scale, causal=causal,
                                window_size=window, fa_version=version)
                return result.view(batch, q_len, heads, dim).transpose(1, 2)
            except RuntimeError as exc:
                # A compiled kernel can pass the architecture check yet fail
                # at launch (for example, an unsupported PTX toolchain).
                _failed_versions.setdefault(index, set()).add(version)
                if hasattr(_flash_kernel, "cache_clear"):
                    _flash_kernel.cache_clear()
                warnings.warn(f"FA{version} kernel failed on cuda:{index}; trying next backend: {exc}",
                              RuntimeWarning, stacklevel=2)

    if mask is not None and mask.ndim == 2:
        # HF's flash-style mask is B,K with True at usable keys.  Build the
        # bottom-right causal mask explicitly for cached multi-token decode.
        mask = mask[:, None, None, :k_len].bool()
        if causal:
            q_pos = torch.arange(q_len, device=query.device) + k_len - q_len
            k_pos = torch.arange(k_len, device=query.device)
            mask = mask & (k_pos[None, None, None, :] <= q_pos[None, None, :, None])
            if sliding_window:
                mask = mask & (k_pos[None, None, None, :] > q_pos[None, None, :, None] - sliding_window)
        causal = False
    elif mask is not None and mask.ndim == 4:
        mask = mask[..., :k_len]
        causal = False
    elif causal and q_len != k_len:
        if q_len == 1:
            causal = False
        else:
            q_pos = torch.arange(q_len, device=query.device) + k_len - q_len
            k_pos = torch.arange(k_len, device=query.device)
            mask = k_pos[None, None, None, :] <= q_pos[None, None, :, None]
            if sliding_window:
                mask = mask & (k_pos[None, None, None, :] > q_pos[None, None, :, None] - sliding_window)
            causal = False
    if sliding_window and mask is None:
        q_pos = torch.arange(q_len, device=query.device) + k_len - q_len
        k_pos = torch.arange(k_len, device=query.device)
        mask = (k_pos[None, None, None, :] <= q_pos[None, None, :, None]) & (
            k_pos[None, None, None, :] > q_pos[None, None, :, None] - sliding_window)
        causal = False
    use_gqa = query.shape[1] != key.shape[1]
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=dropout,
        is_causal=causal, scale=scale, enable_gqa=use_gqa,
    )


def transformers_attention(module, query, key, value, attention_mask,
                           dropout=0.0, scaling=None, sliding_window=None,
                           **kwargs):
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError("Attention weights are unavailable with ex_omni_auto")
    output = attention(
        query, key, value, attention_mask,
        causal=kwargs.get("is_causal", getattr(module, "is_causal", False)),
        scale=scaling, dropout=dropout, sliding_window=sliding_window,
    )
    return output.transpose(1, 2).contiguous(), None


def register_transformers_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, transformers_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, flash_attention_mask)
