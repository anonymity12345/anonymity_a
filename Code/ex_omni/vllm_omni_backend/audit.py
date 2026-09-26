"""CPU-only audit of actual checkpoint headers and adapter dependencies."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import struct

from . import OMNI_COMMIT, VLLM_VERSION
from .contracts import stage_weight_name


def checkpoint_headers(directory):
    tensors = {}
    shards = sorted(Path(directory).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards in {directory}")
    for shard in shards:
        with shard.open("rb") as reader:
            size = struct.unpack("<Q", reader.read(8))[0]
            if size > 100_000_000:
                raise ValueError(f"Invalid safetensors header size in {shard}")
            header = json.loads(reader.read(size))
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            tensors[name] = value
    return tensors


def audit(root):
    root = Path(root)
    header = checkpoint_headers(root / "Ex-Omni")
    config = json.loads((root / "Ex-Omni/config.json").read_text())
    backbone = json.loads((root / "Qwen3/Qwen3-0.6B/config.json").read_text())
    required = {
        "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
        "model.speech_generator.llm.model.model.embed_tokens.weight": [backbone["vocab_size"], backbone["hidden_size"]],
        "model.speech_generator.speech_embedding.weight": [config["unit_vocab_size"] + 3, backbone["hidden_size"]],
        "model.speech_generator.llm_decoder.weight": [config["unit_vocab_size"] + 3, backbone["hidden_size"]],
        "model.speech_generator.llm_decoder.bias": [config["unit_vocab_size"] + 3],
    }
    for name, shape in required.items():
        if name not in header or header[name]["shape"] != shape:
            raise ValueError(f"Checkpoint/config mismatch: {name}; expected {shape}")
    maps = {}
    for stage in ("thinker", "talker"):
        mapped = {name: stage_weight_name(name, stage) for name in header}
        mapped = {k: v for k, v in mapped.items() if v is not None}
        if len(set(mapped.values())) != len(mapped):
            raise ValueError(f"Weight mapping collision in {stage}")
        maps[stage] = dict(tensor_count=len(mapped), groups=dict(Counter(v.split(".")[0] for v in mapped.values())))
    versions = {}
    for name in ("torch", "transformers", "vllm", "vllm-omni"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        checkpoint_header_validation="passed", checkpoint_tensors=len(header), stages=maps,
        installed_versions=versions, target_vllm=VLLM_VERSION, target_omni_commit=OMNI_COMMIT,
        gpu_inference_validated=False, performance_measured=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.ckpt_root), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)
