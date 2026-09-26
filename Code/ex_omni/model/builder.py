#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import transformers
from transformers import AutoTokenizer
import torch
from ex_omni.model import *
from ex_omni.attention import ATTENTION_NAME, register_transformers_attention


def load_tokenizer_auto(model_name_or_path):
    """Load an Auto tokenizer, preferring the slow implementation.

    When the checkpoint directory only provides ``vocab.json`` / ``added_tokens.json``
    without ``merges.txt`` (as on some model snapshots),
    the slow ``Qwen2Tokenizer`` fails to instantiate. In that case we fall
    back to the fast tokenizer, which can be rebuilt from ``tokenizer.json``.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as slow_exc:
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except Exception:
            raise slow_exc


def normalize_attn_implementation(attn_implementation):
    if not attn_implementation:
        return None

    normalized = attn_implementation.lower()
    if normalized == "flash_attn":
        return "flash_attention_2"
    if normalized in {"auto", "fa3->fa2->sdpa"}:
        return ATTENTION_NAME
    return normalized


def resolve_attn_implementation(config, attn_implementation=None):
    resolved = normalize_attn_implementation(attn_implementation)
    if resolved is not None:
        return resolved

    config_attn = getattr(config, "attn_implementation", None)
    if config_attn is None:
        config_attn = getattr(config, "_attn_implementation", None)

    resolved = normalize_attn_implementation(config_attn)
    return resolved or "sdpa"


def _prepare_attention(config, requested, is_inference):
    resolved = resolve_attn_implementation(config, requested)
    if requested is None and is_inference and resolved == "sdpa":
        resolved = ATTENTION_NAME
    if resolved == ATTENTION_NAME:
        register_transformers_attention()
    config.attn_implementation = resolved
    config._attn_implementation = resolved
    return resolved


def resolve_torch_dtype(config, load_bf16=None):
    if load_bf16 is True:
        return torch.bfloat16
    if load_bf16 is False:
        return torch.float32

    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = getattr(config, "model_torch_dtype", None)

    if isinstance(config_dtype, str):
        normalized = config_dtype.lower()
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"float32", "fp32"}:
            return torch.float32
    elif isinstance(config_dtype, torch.dtype):
        return config_dtype

    return torch.float32

def load_pretrained_qwen_model(model_path, load_bf16=None, device_map="auto", device="cuda", attn_implementation=None, is_inference=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    _prepare_attention(config, attn_implementation, is_inference)
    resolved_dtype = resolve_torch_dtype(config, load_bf16)
    config.inference = is_inference
    config.speech_encoder_type = "whisper"
    kwargs['torch_dtype'] = resolved_dtype
    tokenizer = load_tokenizer_auto(model_path)
    model = LlavaHerQwen3ForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    context_len = getattr(model.config, "max_sequence_length", 2048)

    return tokenizer, model, context_len


def load_model_for_inference(
    model_name_or_path,
    load_bf16=None,
    device_map="auto",
    device="cuda",
    attn_implementation=None,
    is_inference=True,
    **kwargs,
):

    config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    _prepare_attention(config, attn_implementation, is_inference)
    resolved_dtype = resolve_torch_dtype(config, load_bf16)
    config.inference = is_inference
    config.speech_encoder_type = "whisper"

    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    kwargs['torch_dtype'] = resolved_dtype

    tokenizer = load_tokenizer_auto(model_name_or_path)
    model = LlavaHerQwen3ForCausalLM.from_pretrained(model_name_or_path, config=config, **kwargs)

    return tokenizer, model
