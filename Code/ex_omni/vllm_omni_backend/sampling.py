"""Reference-compatible RAS decisions, independent of the vLLM sampler."""

import torch

from .components import speech_components
from .contracts import speech_budget


class SpeechSampler:
    def __init__(self, unit_vocab_size, text_count, mode="ras"):
        if mode not in ("ras", "greedy"):
            raise ValueError(mode)
        self.eos = unit_vocab_size
        self.minimum, self.maximum = speech_budget(text_count)
        self.mode = mode
        self.history = []

    def next(self, logits):
        if logits.ndim != 1 or logits.shape[0] != self.eos + 3:
            raise ValueError("Invalid speech logits shape")
        if len(self.history) >= self.maximum:
            return self.eos
        if self.mode == "greedy":
            scores = logits.clone()
            if len(self.history) < self.minimum:
                scores[self.eos] = -torch.inf
            chosen = scores.argmax().item()
        else:
            scores = logits.log_softmax(-1)
            while True:
                chosen = speech_components().ras_sampling(scores, self.history, 25).item()
                if chosen != self.eos or len(self.history) >= self.minimum:
                    break
        if chosen > self.eos:
            raise RuntimeError(
                "Talker sampled a reserved skip token. Baseline replays its previous input in this case; "
                "v1 cannot reproduce that cache transition. Sample is invalid; do not include in timing."
            )
        if chosen != self.eos:
            self.history.append(chosen)
        return chosen
