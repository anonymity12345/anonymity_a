"""Prepare configuration overrides without modifying checkpoint files."""

import argparse
import json
from pathlib import Path

from . import OMNI_COMMIT


def prepare(root, output, sampling="ras", seed=42, devices="0,0", memory="0.55,0.18", enforce_eager=True, request_stats=False, kv_cache_gib=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if sampling not in ("ras", "greedy"):
        raise ValueError("sampling must be ras or greedy")
    gpu_ids = devices.split(",")
    budgets = [float(v) for v in memory.split(",")]
    cache_gib = [float(v) for v in kv_cache_gib.split(",")] if kv_cache_gib is not None else None
    if len(gpu_ids) != 2 or len(budgets) != 2 or any(not 0 < v < 1 for v in budgets):
        raise ValueError("devices and memory must each have two entries")
    if cache_gib is not None and (len(cache_gib) != 2 or any(v <= 0 for v in cache_gib)):
        raise ValueError("kv_cache_gib must contain two positive GiB values")
    if gpu_ids[0] == gpu_ids[1] and sum(budgets) >= 0.9:
        raise ValueError("Leave at least 10% GPU memory for codec, blendshape and rendering")
    paths = {
        "pretrain_speech_encoder_weights": root / "whisper-large-v3",
        "pretrain_speech_generator_weights": root / "Qwen3/Qwen3-0.6B",
        "pretrain_speech_embedding_path": root / "glm-4-voice-decoder/flow.pt",
    }
    for path in [root / "Ex-Omni/config.json", *(p / "config.json" if p.suffix != ".pt" else p for p in paths.values())]:
        if not path.is_file():
            raise FileNotFoundError(path)
    base = json.loads((root / "Ex-Omni/config.json").read_text())
    base.update({k: str(v) for k, v in paths.items()})
    backbone = json.loads((paths["pretrain_speech_generator_weights"] / "config.json").read_text())
    talker = dict(backbone)
    talker.update(
        architectures=["ExOmniTalker"], vocab_size=base["unit_vocab_size"] + 3,
        tie_word_embeddings=False, eos_token_id=base["unit_vocab_size"], bos_token_id=0,
        ex_omni_unit_vocab_size=base["unit_vocab_size"],
        ex_omni_text_vocab_size=backbone["vocab_size"], ex_omni_thinker_dim=base["hidden_size"],
        ex_omni_sampling=sampling, ex_omni_seed=seed,
    )
    thinker = dict(base, architectures=["ExOmniThinker"], ex_omni_ckpt_root=str(root))
    for name, config in (("hf", base), ("thinker", thinker), ("talker", talker)):
        directory = output / name
        if directory.resolve() == (root / "Ex-Omni").resolve():
            raise ValueError("Prepared config directory must differ from the checkpoint")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if name == "hf":
            for source in (root / "Ex-Omni").iterdir():
                if source.name != "config.json" and source.is_file():
                    link = directory / source.name
                    if not link.exists():
                        link.symlink_to(source.resolve())

    stages = []
    for stage_id, name in enumerate(("thinker", "talker")):
        engine = dict(
            model_stage=name, model_arch=f"ExOmni{name.title()}",
            hf_config_path=str(output / name), tokenizer=str(root / "Ex-Omni"),
            worker_type="ar", engine_output_type="latent", dtype="bfloat16",
            scheduler_cls="vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
            gpu_memory_utilization=budgets[stage_id], enforce_eager=enforce_eager,
            disable_log_stats=not request_stats,
            enable_prompt_embeds=True, enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, async_chunk=False, max_num_seqs=1,
            max_model_len=4096, max_num_batched_tokens=4096,
            tensor_parallel_size=1, pipeline_parallel_size=1,
            generation_config="vllm", seed=seed,
        )
        if stage_id == 1:
            engine["skip_tokenizer_init"] = True
        if cache_gib is not None:
            engine["kv_cache_memory_bytes"] = int(cache_gib[stage_id] * (1024 ** 3))
        stage = dict(
            stage_id=stage_id, stage_type="llm",
            runtime=dict(process=True, devices=gpu_ids[stage_id], max_batch_size=1),
            engine_args=engine, final_output=True, final_output_type="text" if stage_id == 0 else "latent",
            default_sampling_params=dict(
                temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                max_tokens=128 if stage_id == 0 else 751, seed=seed,
                detokenize=stage_id == 0,
            ),
        )
        if stage_id == 0:
            stage["is_comprehension"] = True
        else:
            stage["engine_input_source"] = [0]
            stage["custom_process_input_func"] = "ex_omni.vllm_omni_backend.contracts.thinker_to_talker"
            stage["default_sampling_params"]["stop_token_ids"] = [base["unit_vocab_size"]]
        stages.append(stage)
    config = dict(stage_args=stages, runtime=dict(
        enabled=True, defaults=dict(window_size=-1, max_inflight=1),
        edges=[{"from": 0, "to": 1, "window_size": -1}],
    ))
    # JSON is a YAML subset accepted by OmegaConf, with no optional YAML dependency here.
    (output / "stages.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(dict(
        checkpoint_root=str(root), omni_commit=OMNI_COMMIT, sampling=sampling, seed=seed,
        devices=gpu_ids, gpu_memory_utilization=budgets, enforce_eager=enforce_eager,
        kv_cache_gib=cache_gib,
        request_stats=request_stats,
    ), indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampling", choices=("ras", "greedy"), default="ras")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="0,0")
    parser.add_argument("--memory", default="0.55,0.18")
    parser.add_argument("--cuda-graphs", action="store_true", help="Allow vLLM CUDA graph capture")
    args = parser.parse_args()
    print(prepare(args.ckpt_root, args.output, args.sampling, args.seed, args.devices, args.memory, not args.cuda_graphs))
