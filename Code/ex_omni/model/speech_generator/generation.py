"""Native text generation with the AR speech decoder's alignment constraints."""

import torch
from transformers.generation.utils import GenerationMixin


@torch.no_grad()
def generate_text(model, inputs=None, generation_config=None, **kwargs):
    config = generation_config if generation_config is not None else model.generation_config
    if kwargs.get("num_beams", config.num_beams) != 1:
        raise NotImplementedError("Speech hidden-state alignment requires num_beams=1")
    if kwargs.get("num_return_sequences", config.num_return_sequences) != 1:
        raise NotImplementedError("Speech generation requires num_return_sequences=1")
    if kwargs.get("assistant_model") is not None or kwargs.get("custom_generate") is not None:
        raise NotImplementedError("Speech hidden-state alignment requires standard autoregressive generation")
    return GenerationMixin.generate(
        model, inputs=inputs, generation_config=generation_config, **kwargs
    )
