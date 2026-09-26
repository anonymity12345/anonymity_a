"""Matched HF/vLLM-Omni full-output benchmark. Run each engine in its own env."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


PROMPTS = [
    "Tell me a fun fact about the ocean.",
    "Introduce yourself in one sentence.",
    "What's a good way to stay motivated while learning?",
    "Explain the difference between a meteor and a comet.",
    "Give me three tips for writing cleaner code.",
    "Describe what happens inside a black hole.",
    "Summarize the plot of your favorite childhood story.",
    "Why is the sky blue during the day but red at sunset?",
    "List five benefits of regular exercise.",
    "Tell me a short joke about computers.",
]
SYSTEM = "You are a multimodal assistant that understands both speech and text, and can respond using natural language or synthesized speech."
CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def render_with_prefix(weights, render_batches, prefix, alignment):
    """Reuse complete render cycles without changing subsequent batch shapes/devices."""
    if alignment < 1:
        raise ValueError("Render alignment must be positive")
    available = sum(len(batch) for batch in prefix)
    if available > len(weights):
        raise ValueError("Rendered prefix exceeds response length")
    keep = available if available == len(weights) else available // alignment * alignment
    remaining = keep
    for batch in prefix:
        if remaining <= 0:
            break
        count = min(len(batch), remaining)
        yield batch[:count]
        remaining -= count
    if keep < len(weights):
        yield from render_batches(weights[keep:])


def encode_first_video(frames, path, audio_path):
    import numpy as np
    from ex_omni import render_utils as render

    render.images_to_video(np.concatenate(frames), str(path), fps=30, audio_input=str(audio_path))
    return time.perf_counter()


def summarize(records):
    import numpy as np

    if not records:
        raise ValueError("No timed samples")
    walls = [r["wall"] for r in records]
    durations = [r["output_audio_s"] for r in records]
    if any(v <= 0 for v in walls + durations):
        raise ValueError("Invalid timing/audio duration")
    result = dict(
        n_valid_samples=len(records), avg_rtf=float(np.mean([w / d for w, d in zip(walls, durations)])),
        p50_latency_s=float(np.percentile(walls, 50)), p95_latency_s=float(np.percentile(walls, 95)),
        system_realtime_x=sum(durations) / sum(walls), samples_per_s=len(records) / sum(walls),
    )
    for name in ("model_s", "codec_and_wav_s", "blendshape_transfer_s", "render_s", "video_encode_mux_s",
                 "complete_audio_s", "complete_video_s"):
        if all(name in record for record in records):
            result[f"mean_{name}"] = float(np.mean([record[name] for record in records]))
    first_response = (
        "thinker_ttft_s", "thinker_tpop_ms", "talker_first_unit_s",
        "talker_tpop_ms", "thinker_complete_s", "talker_complete_s",
        "codec_first_chunk_s", "overall_audio_latency_s",
        "render_first_chunk_s", "overall_video_latency_s",
    )
    if all(all(name in record for name in first_response) for record in records):
        result["first_response"] = {
            name: {
                "mean": float(np.mean([record[name] for record in records])),
                "p50": float(np.percentile([record[name] for record in records], 50)),
                "p95": float(np.percentile([record[name] for record in records], 95)),
            }
            for name in first_response
        }
    return result


class HFBackend:
    def __init__(self, prepared, sampling, device, max_new_tokens):
        from ex_omni.model.builder import load_model_for_inference

        self.tokenizer, self.model = load_model_for_inference(
            str(prepared / "hf"), load_bf16=True, device_map=None, attn_implementation="auto"
        )
        self.model.eval().to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        if sampling == "greedy":
            def greedy(scores, history, sampling, ignore_eos):
                scores = scores.clone()
                if ignore_eos:
                    scores[self.model.get_model().speech_generator.unit_vocab_size] = -float("inf")
                return scores.argmax().reshape(1)
            self.model.get_model().speech_generator.sampling_ids = greedy

    def generate(self, ids, features=None, speech_offset=None):
        import torch

        if features is None:
            features = torch.zeros(1, 128, 3000)
        if speech_offset is not None:
            # Collapse expanded placeholders back to the configured sentinel.
            count = features.shape[-1] // 2 // self.model.config.speech_encoder_ds_rate
            ids = ids[:speech_offset] + [-300] + ids[speech_offset + count:]
        with torch.inference_mode():
            text, units, bs = self.model.generate(
                torch.tensor([ids], device=self.device),
                speech=features.to(self.device, torch.bfloat16),
                speech_lengths=torch.tensor([features.shape[-1]]),
                do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                faster_infer=False,
            )
        return text[0].tolist(), [int(u) for u in units.split()], bs

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("hf", "vllm_omni"), required=True)
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("text", "speech", "both"), default="text")
    parser.add_argument("--text-prompts-file", type=Path, help="JSON list of strings or {prompt: ...} rows; index 0 is warm-up")
    parser.add_argument("--speech-files", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=("ras", "greedy"), default="ras")
    parser.add_argument("--devices", default="0,0", help="Thinker,Talker GPU IDs; also used for generated stage config")
    parser.add_argument("--memory", default="0.55,0.18")
    parser.add_argument("--kv-cache-gib", help="Explicit Thinker,Talker KV cache GiB; avoids same-GPU graph startup over-allocation")
    parser.add_argument("--post-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--render-batch-size", type=int, default=16)
    parser.add_argument("--render-devices", help="Comma-separated GPU IDs for frame-parallel rendering of one response")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--stream-video", action="store_true", help="Encode each rendered batch concurrently")
    parser.add_argument("--codec-batch-cfg", action="store_true", help="Evaluate conditional and unconditional Flow branches as one batch")
    parser.add_argument("--codec-compile-estimator", choices=("none", "default"), default="none", help="Compile the Flow estimator after codec loading")
    parser.add_argument("--codec-cuda-graph-estimator", action="store_true", help="Capture the fixed first audio chunk Flow estimator")
    parser.add_argument("--no-codec-cuda-graph-estimator", action="store_true", help="A/B reference: disable only the fixed first-chunk CUDA graph")
    parser.add_argument("--optimized", action="store_true", help="Enable CUDA Graph, compiled first-chunk codec, batched CFG, streaming video, and bounded KV caches")
    parser.add_argument("--cuda-graphs", action="store_true", help="Allow vLLM CUDA graph capture")
    parser.add_argument("--latency-metrics", action="store_true", help="Measure engine first-token times and first playable audio/video segments")
    parser.add_argument("--overlap-first-video", action="store_true", help="Encode the first MP4 concurrently with full audio decoding")
    parser.add_argument("--reuse-first-frames", action="store_true", help="Reuse aligned first-segment RGB batches in the complete video")
    parser.add_argument("--ready-file", type=Path, help="Signal that warm-up completed")
    parser.add_argument("--start-file", type=Path, help="Wait for this file before timed requests")
    args = parser.parse_args()
    if args.optimized:
        if args.engine != "vllm_omni" or args.no_render:
            parser.error("--optimized requires vllm_omni and video rendering")
        args.cuda_graphs = True
        args.codec_batch_cfg = True
        args.codec_compile_estimator = "default"
        args.codec_cuda_graph_estimator = not args.no_codec_cuda_graph_estimator
        args.stream_video = True
        if args.kv_cache_gib is None:
            args.kv_cache_gib = "4,2"
    if min(args.samples, args.max_new_tokens, args.render_batch_size) < 1:
        parser.error("samples, max-new-tokens and render-batch-size must be positive")
    if args.text_prompts_file:
        source = json.loads(args.text_prompts_file.read_text())
        prompts = [row["prompt"] if isinstance(row, dict) else row for row in source]
        if len(prompts) < 2 or any(not isinstance(row, str) or not row for row in prompts):
            parser.error("--text-prompts-file must contain at least two non-empty prompts")
        if args.samples > len(prompts) - 1:
            parser.error("--samples exceeds prompts after warm-up")
    else:
        prompts = PROMPTS
    if args.input_mode != "text" and (not args.speech_files or len(args.speech_files) < min(args.samples + 1, len(prompts))):
        parser.error("Provide a warm-up audio file followed by one file per timed prompt")
    if args.text_prompts_file and args.input_mode != "text" and len(args.speech_files) != len(prompts):
        parser.error("Custom prompts require one paired speech file per prompt")
    if args.speech_files and any(not p.is_file() for p in args.speech_files):
        parser.error("Speech input file missing")
    if bool(args.ready_file) != bool(args.start_file):
        parser.error("--ready-file and --start-file must be supplied together")
    if args.latency_metrics and (args.engine != "vllm_omni" or args.no_render):
        parser.error("--latency-metrics requires vllm_omni and full video rendering")
    if (args.overlap_first_video or args.reuse_first_frames) and not args.latency_metrics:
        parser.error("First-segment optimizations require --latency-metrics")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        parser.error("Output directory already contains a run; choose a fresh directory")
    root = args.ckpt_root.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import numpy as np
    import torch
    import soundfile as sf
    from ex_omni.attention import selected_backend
    from transformers import AutoTokenizer, AutoFeatureExtractor
    from .prepare import prepare

    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; use CPU unit tests in the debug environment")
    prepared = prepare(root, output / "config", args.sampling, args.seed, args.devices, args.memory,
                       not args.cuda_graphs, request_stats=args.latency_metrics,
                       kv_cache_gib=args.kv_cache_gib)
    tokenizer = AutoTokenizer.from_pretrained(root / "Ex-Omni", use_fast=False, local_files_only=True)
    tokenizer.add_tokens(["<speech>"], special_tokens=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    speech_id = tokenizer.convert_tokens_to_ids("<speech>")
    extractor = AutoFeatureExtractor.from_pretrained(root / "whisper-large-v3", local_files_only=True)
    model_config = json.loads((root / "Ex-Omni/config.json").read_text())
    modes = ["text", "speech"] if args.input_mode == "both" else [args.input_mode]
    metadata = dict(
        engine=args.engine, sampling=args.sampling, seed=args.seed,
        hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        devices=args.devices, post_device=args.post_device, torch_version=torch.__version__,
        kv_cache_gib=args.kv_cache_gib,
        checkpoint_config_sha256=hashlib.sha256((root / "Ex-Omni/config.json").read_bytes()).hexdigest(),
        text_prompts_source=str(args.text_prompts_file.resolve()) if args.text_prompts_file else "builtin",
        text_prompts_sha256=hashlib.sha256(args.text_prompts_file.read_bytes()).hexdigest() if args.text_prompts_file else None,
        n_prompts=len(prompts),
        max_new_tokens=args.max_new_tokens, render_batch_size=args.render_batch_size,
        include_render=not args.no_render, samples_per_mode=args.samples, warmup_per_mode=1,
        cuda_graphs=args.cuda_graphs,
        optimized_preset=args.optimized,
        latency_metrics=args.latency_metrics,
        stream_video=args.stream_video,
        render_devices=args.render_devices,
        overlap_first_video=args.overlap_first_video,
        reuse_first_frames=args.reuse_first_frames,
        codec_batch_cfg=args.codec_batch_cfg,
        codec_compile_estimator=args.codec_compile_estimator,
        codec_cuda_graph_estimator=args.codec_cuda_graph_estimator,
        auxiliary_attention_backend=selected_backend(args.post_device),
        sample_rate=22050, video_fps=30, image_size=[512, 512],
        timing="complete input available through full audio, blendshape, optional full video and mux",
        stage_timing="CUDA synchronized after model generation; render and ffmpeg measured separately",
        first_chunk_metrics=False, speech_files=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.speech_files or []],
    )
    if args.latency_metrics:
        metadata.update(
            first_chunk_metrics=True,
            first_audio_units=10,
            first_video_frames=30,
            first_token_definition="vLLM engine first generated token, including special tokens",
            first_audio_definition="complete input to first 10-unit WAV written after complete Talker stage",
            first_video_definition="complete input to playable first 30-frame MP4 with first audio chunk",
            full_output_includes_first_segment_work=True,
            stage_timing="codec_and_wav_s spans model completion through full WAV, including first WAV/render preparation. First MP4 encoding may overlap; complete_audio_s and complete_video_s are input-relative completion times. wall waits for both MP4s. Stage spans are not isolated kernel costs.",
        )
    from importlib.metadata import version
    metadata["transformers_version"] = version("transformers")
    if args.engine == "vllm_omni":
        metadata.update(vllm_version=version("vllm"), vllm_omni_version=version("vllm-omni"))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    backend = None
    parallel_renderer = None
    from concurrent.futures import ThreadPoolExecutor

    encoder_pool = ThreadPoolExecutor(max_workers=1) if args.overlap_first_video else None
    try:
        if args.engine == "hf":
            backend = HFBackend(prepared, args.sampling, args.post_device, args.max_new_tokens)
        else:
            from .runtime import OmniBackend
            backend = OmniBackend(root, prepared, args.post_device, args.max_new_tokens)
        from ex_omni.flow_inference import AudioDecoder
        decoder = AudioDecoder(
            str(Path(__file__).resolve().parents[2] / "cosyvoice/vocab_16K.yaml"),
            str(root / "glm-4-voice-decoder/flow.pt"), str(root / "glm-4-voice-decoder/hift.pt"),
            device=args.post_device,
            batch_cfg=args.codec_batch_cfg,
            compile_estimator=args.codec_compile_estimator,
            cuda_graph_estimator=args.codec_cuda_graph_estimator,
        )
        if not args.no_render:
            from ex_omni import render_utils as render
            mean, faces, shapes, names = render.load_mesh_data(str(Path(__file__).resolve().parents[2] / "asset/EmoTalk.npz"))
            mean, faces, shapes = [v.to(args.post_device) for v in (mean, faces, shapes)]
            renderer = render.setup_renderer(mean, image_size=(512, 512))
            if args.render_devices:
                parallel_renderer = render.ParallelFrameRenderer(
                    mean, faces, shapes,
                    [f"cuda:{int(value)}" for value in args.render_devices.split(',')])

        def render_batches(weights):
            if parallel_renderer is not None:
                yield from parallel_renderer.render(weights, args.render_batch_size)
            else:
                for i in range(0, len(weights), args.render_batch_size):
                    vertices = render.apply_blendshapes_batch(mean, shapes, weights[i:i + args.render_batch_size])
                    yield render.render_frames(renderer, vertices, faces).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()

        summaries = {}
        for mode in modes:
            records = []
            for run in range(args.samples + 1):
                index = 0 if run == 0 else 1 + (run - 1) % (len(prompts) - 1)
                tag = f"{mode}_{run:03d}"
                torch.manual_seed(args.seed)
                torch.cuda.synchronize(args.post_device)
                start = time.perf_counter()
                features, offset, input_duration = None, None, None
                content = prompts[index] if mode == "text" else "<speech>"
                ids = tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM}, {"role": "user", "content": content}
                ], add_generation_prompt=True)
                if mode == "speech":
                    import librosa
                    wave, sr = sf.read(args.speech_files[index], dtype="float32", always_2d=True)
                    wave = wave.mean(axis=1)
                    input_duration = len(wave) / sr
                    if not np.isfinite(wave).all() or not len(wave):
                        raise ValueError("Empty/non-finite speech input")
                    if sr != 16000:
                        wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
                    features = extractor(wave, sampling_rate=16000, return_tensors="pt")["input_features"]
                    count = features.shape[-1] // 2 // model_config["speech_encoder_ds_rate"]
                    offset = ids.index(speech_id)
                    ids = ids[:offset] + [0] * count + ids[offset + 1:]
                if args.latency_metrics:
                    text_ids, units, bs = backend.generate(ids, features, offset, timing_origin=start)
                else:
                    text_ids, units, bs = backend.generate(ids, features, offset)
                torch.cuda.synchronize(args.post_device)
                after_generate = time.perf_counter()
                if not units:
                    raise RuntimeError("No speech units generated")
                first_metrics = {}
                first_frames = []
                first_video_future = None
                if args.latency_metrics:
                    observations = backend.last_stage_observations
                    if set(observations) != {0, 1} or any(not observations[s]["stats"] for s in (0, 1)):
                        raise RuntimeError(f"Missing vLLM per-stage request statistics: {observations}")
                    for stage_id, label in ((0, "thinker"), (1, "talker")):
                        stats = observations[stage_id]["stats"]
                        token_count = int(stats["num_generation_tokens"])
                        first_s = stats["first_token_ts"] - start
                        last_s = stats["last_token_ts"] - start
                        if token_count < 2 or not 0 < first_s <= last_s <= observations[stage_id]["completed_s"]:
                            raise RuntimeError(f"Invalid {label} token timing: {observations[stage_id]}")
                        first_metrics.update({
                            f"{label}_first_token_s": first_s,
                            f"{label}_tpop_ms": 1000 * (last_s - first_s) / (token_count - 1),
                            f"{label}_tokens": token_count,
                            f"{label}_complete_s": observations[stage_id]["completed_s"],
                        })
                    first_metrics["thinker_ttft_s"] = first_metrics["thinker_first_token_s"]
                    first_metrics["talker_first_unit_s"] = first_metrics["talker_first_token_s"]
                    chunk_start = time.perf_counter()
                    with torch.random.fork_rng(devices=[torch.device(args.post_device).index or 0]):
                        chunk_audio = decoder.offline_inference(torch.tensor([units[:10]], device=args.post_device))
                    chunk_wav = chunk_audio.float().cpu().numpy().reshape(-1)
                    first_metrics["codec_first_chunk_s"] = time.perf_counter() - chunk_start
                    first_metrics["first_audio_ready_s"] = time.perf_counter() - start
                    chunk_path = output / f"{tag}_first_audio.wav"
                    sf.write(chunk_path, chunk_wav, 22050, subtype="PCM_16")
                    first_metrics["overall_audio_latency_s"] = time.perf_counter() - start
                    bs_array = bs.squeeze(0).detach().float().cpu().numpy()
                    if bs_array.ndim != 2 or bs_array.shape[-1] not in (51, 52) or not np.isfinite(bs_array).all():
                        raise RuntimeError("Invalid blendshape output")
                    weights = np.clip(bs_array, 0, 1)
                    if weights.shape[-1] == 51:
                        weights = np.pad(weights, ((0, 0), (0, 1)))
                    weights = torch.tensor(weights, device=args.post_device)
                    weights[:, 26] = 0
                    first_video_start = time.perf_counter()
                    first_frame_count = min(30, len(weights))
                    first_frames = list(render_batches(weights[:first_frame_count]))
                    encode_args = (first_frames, output / f"{tag}_first_video.mp4", chunk_path)
                    if encoder_pool is not None:
                        first_video_future = encoder_pool.submit(encode_first_video, *encode_args)
                    else:
                        first_video_end = encode_first_video(*encode_args)
                        first_metrics["render_first_chunk_s"] = first_video_end - first_video_start
                        first_metrics["overall_video_latency_s"] = first_video_end - start
                with torch.inference_mode():
                    # Full responses have different lengths. Keep the fixed 10-unit
                    # first chunk compiled, but avoid recompiling Flow per request.
                    audio = decoder.offline_inference(
                        torch.tensor([units], device=args.post_device), use_compiled=False
                    )
                wav = audio.float().cpu().numpy().reshape(-1)
                if not len(wav) or not np.isfinite(wav).all():
                    raise RuntimeError("Empty/non-finite generated audio")
                audio_path = output / f"{tag}.wav"
                sf.write(audio_path, wav, 22050, subtype="PCM_16")
                after_codec = time.perf_counter()
                if not args.latency_metrics:
                    bs_array = bs.squeeze(0).detach().float().cpu().numpy()
                after_blendshape_transfer = time.perf_counter()
                if bs_array.ndim != 2 or bs_array.shape[-1] not in (51, 52) or not np.isfinite(bs_array).all():
                    raise RuntimeError("Invalid blendshape output")
                if not args.no_render:
                    if not args.latency_metrics:
                        weights = np.clip(bs_array, 0, 1)
                        if weights.shape[-1] == 51:
                            weights = np.pad(weights, ((0, 0), (0, 1)))
                        weights = torch.tensor(weights, device=args.post_device)
                        weights[:, 26] = 0
                    video_path = str(output / f"{tag}.mp4")
                    alignment = args.render_batch_size * (len(parallel_renderer.workers) if parallel_renderer else 1)
                    full_batches = render_with_prefix(
                        weights, render_batches, first_frames if args.reuse_first_frames else [], alignment)
                    if args.stream_video:
                        with render.VideoStreamWriter(video_path, image_size=(512, 512), fps=30, audio_input=str(audio_path)) as writer:
                            for frame_batch in full_batches:
                                writer.write(frame_batch)
                            after_render = time.perf_counter()
                        after_mux = time.perf_counter()
                    else:
                        frames = list(full_batches)
                        after_render = time.perf_counter()
                        render.images_to_video(np.concatenate(frames), video_path, fps=30, audio_input=str(audio_path))
                        after_mux = time.perf_counter()
                else:
                    after_render = after_mux = after_blendshape_transfer
                if first_video_future is not None:
                    first_video_end = first_video_future.result()
                    first_metrics["render_first_chunk_s"] = first_video_end - first_video_start
                    first_metrics["overall_video_latency_s"] = first_video_end - start
                first_metrics["complete_audio_s"] = after_codec - start
                first_metrics["complete_video_s"] = after_mux - start
                torch.cuda.synchronize(args.post_device)
                wall = time.perf_counter() - start
                # Diagnostic serialization is excluded from inference timing.
                np.save(output / f"{tag}_blendshape.npy", bs_array)
                duration = len(wav) / 22050
                record = dict(
                    sample_idx=index, prompt=prompts[index], generated_text=tokenizer.decode(text_ids, skip_special_tokens=True),
                    text_token_ids=text_ids, speech_units=units, blendshape_frames=len(bs_array),
                    wall=wall, output_audio_s=duration, output_rtf=wall / duration,
                    model_s=after_generate - start, codec_and_wav_s=after_codec - after_generate,
                    blendshape_transfer_s=after_blendshape_transfer - after_codec,
                    render_s=after_render - after_blendshape_transfer,
                    video_encode_mux_s=after_mux - after_render,
                    input_audio_s=input_duration, input_rtf=wall / input_duration if input_duration else None,
                    audio_file=audio_path.name, blendshape_file=f"{tag}_blendshape.npy",
                    **first_metrics,
                )
                if run > 0:
                    records.append(record)
                    (output / f"raw_{mode}.json").write_text(json.dumps(records, indent=2) + "\n")
                print(f"[{mode}] {'warmup' if run == 0 else run}: wall={wall:.3f}s output RTF={wall / duration:.3f}", flush=True)
                if run == 0 and args.ready_file:
                    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                    args.ready_file.touch()
                    deadline = time.monotonic() + 1800
                    while not args.start_file.exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"Timed-run start signal missing: {args.start_file}")
                        time.sleep(0.1)
            summaries[mode] = summarize(records)
        graph_estimator = getattr(decoder, "_compiled_estimator", None)
        metadata["codec_cuda_graph_captured"] = bool(getattr(graph_estimator, "_graph", None))
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        rows = ["# Ex-Omni Full-Output Benchmark", "", f"Engine: {args.engine}; Talker sampling: {args.sampling}.", "",
                "Warm-up excluded. Full-output timing; no first-chunk latency metrics.", "",
                "| Input | N | Mean output RTF | P50 seconds | P95 seconds | Output seconds / wall second |", "|:--|--:|--:|--:|--:|--:|"]
        for mode, value in summaries.items():
            rows.append(f"| {mode} | {value['n_valid_samples']} | {value['avg_rtf']:.4f} | {value['p50_latency_s']:.3f} | {value['p95_latency_s']:.3f} | {value['system_realtime_x']:.3f} |")
        if args.latency_metrics:
            rows += [
                "", "## First-response latency", "",
                "Timing starts when the complete input is available. TTFT is the engine's first generated token; verify that this token is visible text before comparing with a visible-token TTFT. Talker first unit is also input-relative. The first playable WAV contains up to 10 units; the first playable MP4 contains up to 30 frames. The current Omni handoff exposes speech units only after Talker completes, so these are actual user-facing first-media times, not an estimated 10th-unit time.",
                "", "| Input | TTFT P50/P95 (s) | Thinker TPOP P50 (ms) | Talker first unit P50 (s) | Talker TPOP P50 (ms) | First WAV P50/P95 (s) | First MP4 P50/P95 (s) |",
                "|:--|:--|--:|--:|--:|:--|:--|",
            ]
            for mode, value in summaries.items():
                first = value["first_response"]
                pair = lambda key: f"{first[key]['p50']:.3f}/{first[key]['p95']:.3f}"
                rows.append(
                    f"| {mode} | {pair('thinker_ttft_s')} | {first['thinker_tpop_ms']['p50']:.2f} | "
                    f"{first['talker_first_unit_s']['p50']:.3f} | {first['talker_tpop_ms']['p50']:.2f} | "
                    f"{pair('overall_audio_latency_s')} | {pair('overall_video_latency_s')} |"
                )
        (output / "summary.md").write_text("\n".join(rows) + "\n")
    finally:
        if encoder_pool is not None:
            encoder_pool.shutdown(wait=True, cancel_futures=True)
        if parallel_renderer is not None:
            parallel_renderer.close()
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
