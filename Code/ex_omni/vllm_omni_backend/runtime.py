"""Synchronous full-output pipeline for RTF comparisons."""

import json
from pathlib import Path
import time

import torch

from . import check_runtime
from .components import blendshape_class
from .contracts import generation_hidden, speech_budget


def load_blendshape(root, device):
    from safetensors import safe_open
    from transformers import PretrainedConfig

    config = PretrainedConfig.from_dict(json.loads((Path(root) / "Ex-Omni/config.json").read_text()))
    model = blendshape_class()(config)
    state = {}
    prefix = "model.blendshape_generator."
    for shard in sorted((Path(root) / "Ex-Omni").glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith(prefix):
                    state[name[len(prefix):]] = reader.get_tensor(name)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device=device, dtype=torch.bfloat16)


class OmniBackend:
    def __init__(self, root, prepared, device="cuda:0", max_new_tokens=128):
        check_runtime()
        from vllm import SamplingParams
        from vllm_omni.entrypoints.omni import Omni

        self.device = device
        self.config = json.loads((Path(root) / "Ex-Omni/config.json").read_text())
        self.omni = Omni(
            model=str(Path(root).resolve() / "Ex-Omni"),
            stage_configs_path=str(Path(prepared).resolve() / "stages.yaml"),
        )
        self.sampling_params = list(self.omni.default_sampling_params_list)
        self.sampling_params[0] = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, repetition_penalty=1.0, seed=42,
        )
        try:
            self.blendshape = load_blendshape(root, device)
        except Exception:
            self.omni.close()
            raise
        self.last_stage_observations = {}

    @torch.inference_mode()
    def generate(self, ids, features=None, speech_offset=None, timing_origin=None):
        from vllm_omni.inputs.data import OmniTokensPrompt

        information = {"ex_omni_request": [1]}
        if features is not None:
            information = dict(
                speech_features=features.cpu(), speech_offset=[speech_offset],
                speech_count=[features.shape[-1] // 2 // self.config["speech_encoder_ds_rate"]],
            )
        prompts = [OmniTokensPrompt(prompt_token_ids=ids, additional_information=information)]
        if timing_origin is None:
            outputs = self.omni.generate(prompts, self.sampling_params, use_tqdm=False)
        else:
            # Consume stage completions through the persistent iterator.
            outputs = self.omni._run_generation(prompts, self.sampling_params, use_tqdm=False)
        self.last_stage_observations = {}
        stages = {}
        for output in outputs:
            request = output.request_output
            if isinstance(request, list):
                if len(request) != 1:
                    raise RuntimeError("Expected one request per stage")
                request = request[0]
            stages[output.stage_id] = request
            if timing_origin is not None:
                metrics = getattr(request, "metrics", None)
                stats = None
                if metrics is not None:
                    values = metrics if isinstance(metrics, dict) else vars(metrics)
                    stats = {name: float(values[name]) for name in (
                        "arrival_time", "first_token_ts", "last_token_ts",
                        "first_token_latency", "num_generation_tokens",
                    ) if name in values}
                self.last_stage_observations[output.stage_id] = dict(
                    completed_s=time.perf_counter() - timing_origin, stats=stats,
                    metrics_type=type(metrics).__name__,
                    metrics_keys=list(values) if metrics is not None else None,
                )
        if set(stages) != {0, 1}:
            raise RuntimeError(f"Expected completed Thinker and Talker stages, received {list(stages)}")
        text_ids = list(stages[0].outputs[0].token_ids)
        talker = stages[1]
        unit_ids = list(talker.outputs[0].token_ids)
        hidden = generation_hidden(
            talker.outputs[0].multimodal_output["latent"], len(talker.prompt_token_ids), len(unit_ids)
        )
        eos = self.config["unit_vocab_size"]
        _, maximum = speech_budget(len(text_ids) - 1)
        if len(unit_ids) == maximum + 1 and unit_ids[-1] == eos:
            hidden = hidden[:, :-1]
            unit_ids = unit_ids[:-1]
        units = [u for u in unit_ids if u < eos]
        if not units or any(u > eos for u in unit_ids):
            raise RuntimeError("Talker returned empty audio or unsupported reserved units")
        tensor = torch.tensor([units], device=self.device, dtype=torch.long)
        bs = self.blendshape.predict(hidden.to(self.device, torch.bfloat16), tensor)
        return text_ids, units, bs

    def close(self):
        self.omni.close()
