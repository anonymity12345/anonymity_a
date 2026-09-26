"""Native Qwen3 attention/KV-cache stages for vLLM-Omni v0.14.0.

Version one supports one active sequence per stage, full prefill and TP=PP=1.
Per-request preprocessing uses Omni's state store.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .components import source_module, speech_components
from .contracts import speech_budget, stage_weight_name
from .sampling import SpeechSampler


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def validate_config(config):
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Ex-Omni v1 requires tensor_parallel_size=pipeline_parallel_size=1")
    if config.scheduler_config.max_num_seqs != 1:
        raise ValueError("Ex-Omni v1 requires max_num_seqs=1")
    if config.scheduler_config.enable_chunked_prefill:
        raise ValueError("Full-sequence TQGF requires enable_chunked_prefill=false")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Latent alignment requires enable_prefix_caching=false")
    if config.model_config.async_chunk:
        raise ValueError("Ex-Omni v1 requires async_chunk=false")
    if not config.model_config.enable_prompt_embeds:
        raise ValueError("Ex-Omni requires enable_prompt_embeds=true for stage preprocessing")
    if config.speculative_config is not None:
        raise ValueError("Ex-Omni v1 does not support speculative decoding")


def load_stage_weights(module, weights, stage):
    def mapped():
        for name, tensor in weights:
            target = stage_weight_name(name, stage)
            if target is not None:
                yield target, tensor

    loaded = AutoWeightsLoader(module).load_weights(mapped())
    missing = set(dict(module.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"{stage}: checkpoint did not initialize parameters: {sorted(missing)}")
    return loaded


class ExOmniThinker(Qwen3ForCausalLM):
    has_preprocess = True
    has_postprocess = False
    have_multimodal_outputs = True

    def __init__(self, *, vllm_config, prefix=""):
        validate_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        root = Path(self.config.ex_omni_ckpt_root)
        whisper = WhisperConfig.from_pretrained(root / "whisper-large-v3", local_files_only=True)
        # Use SDPA for deterministic speech-unit behavior.
        whisper._attn_implementation = "sdpa"
        self.speech_encoder = WhisperEncoder(whisper)
        projector = source_module("model/speech_projector/speech_projector.py", "_projector")
        self.speech_projector = projector.EncoderProjectorConcat(self.config)

    def preprocess(self, input_ids, input_embeds, **info):
        if "speech_features" not in info or scalar(info.get("speech_consumed", [False])):
            return input_ids, input_embeds, {}
        features = info["speech_features"].to(
            device=input_embeds.device, dtype=input_embeds.dtype
        ).reshape(1, 128, 3000)
        encoded = self.speech_encoder(features).last_hidden_state
        projected = self.speech_projector(encoded).squeeze(0)
        offset = int(scalar(info["speech_offset"]))
        count = int(scalar(info["speech_count"]))
        if count != len(projected) or offset + count > len(input_embeds):
            raise ValueError("Speech placeholder length does not match projected Whisper features")
        input_embeds = input_embeds.clone()
        input_embeds[offset:offset + count] = projected
        return input_ids, input_embeds, {"speech_consumed": [True]}

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        return OmniOutput(text_hidden_states=hidden)

    def load_weights(self, weights):
        return load_stage_weights(self, weights, "thinker")


class ExOmniTalker(nn.Module):
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping
    has_preprocess = True
    has_postprocess = False
    have_multimodal_outputs = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        validate_config(vllm_config)
        self.config = vllm_config.model_config.hf_config
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=f"{prefix}.model".lstrip("."))
        dim = self.config.hidden_size
        self.unit_vocab_size = self.config.ex_omni_unit_vocab_size
        self.text_embedding = nn.Embedding(self.config.ex_omni_text_vocab_size, dim)
        self.llm_embedding = nn.Embedding(2, dim)
        self.llm_decoder = nn.Linear(dim, self.unit_vocab_size + 3)
        self.input_proj = nn.Sequential(
            nn.Linear(self.config.ex_omni_thinker_dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim * 4)
        )
        self.tqgf = speech_components().TQGF(dim)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._text_count = None
        self._seed = self.config.ex_omni_seed
        self._speech_sampler = None

    def embed_input_ids(self, input_ids, **kwargs):
        return self.model.embed_input_ids(input_ids)

    def preprocess(self, input_ids, input_embeds, **info):
        if scalar(info.get("talker_initialized", [False])):
            if len(input_ids) != 1:
                raise RuntimeError("Talker preemption/recomputed prefill is unsupported; increase KV memory")
            return input_ids, input_embeds, {}
        text = torch.as_tensor(info["text_ids"], device=input_embeds.device, dtype=torch.long).reshape(1, -1)
        self._text_count = text.numel()
        speech_budget(self._text_count)
        hidden = info["thinker_hidden"].to(input_embeds).reshape(1, -1, self.config.ex_omni_thinker_dim)
        hidden = self.input_proj(hidden)
        # Same rearrange as baseline: 'b n (d1 d2) -> b (n d2) d1', d2=4.
        hidden = hidden.reshape(1, hidden.shape[1], -1, 4).permute(0, 1, 3, 2).flatten(1, 2)
        text_embeds = self.tqgf(hidden, self.text_embedding(text), None).squeeze(0)
        embeds = torch.cat((self.llm_embedding.weight[0:1], text_embeds, self.llm_embedding.weight[1:2]))
        if embeds.shape != input_embeds.shape:
            raise ValueError(f"Talker prompt shape mismatch: {embeds.shape} != {input_embeds.shape}")
        torch.manual_seed(self._seed)
        self._speech_sampler = SpeechSampler(
            self.unit_vocab_size, self._text_count, self.config.ex_omni_sampling
        )
        return input_ids, embeds, {"talker_initialized": [True]}

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        hidden = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return OmniOutput(text_hidden_states=hidden)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        logits = self.llm_decoder(hidden_states)
        # Profiling runs may not have a request/preprocess context.
        if self._text_count is None or sampling_metadata is None:
            return logits
        if logits.shape[0] != 1:
            raise ValueError("Ex-Omni Talker supports one active request")
        if not sampling_metadata.all_greedy:
            raise ValueError("Omni sampler must use temperature=0; RAS is applied inside the model")
        # vLLM omits output_token_ids when no native penalties are requested.
        # Keep RAS history locally under the enforced one-request contract.
        chosen = self._speech_sampler.next(logits[0])
        # Use the original RAS decision, then make vLLM's greedy sampler select
        # exactly that ID. vLLM continues to own scheduling and paged KV caches.
        forced = torch.full_like(logits, -torch.inf)
        forced[0, chosen] = 0
        return forced

    def load_weights(self, weights):
        return load_stage_weights(self, weights, "talker")
