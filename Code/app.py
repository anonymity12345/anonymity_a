"""Interactive text/speech inference and 3D avatar demo."""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EX_OMNI_TEMP_DIR", "/tmp/ex_omni_outputs")

import time  # noqa: E402
import uuid  # noqa: E402
import types  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402
import gradio as gr  # noqa: E402
import huggingface_hub  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

# diffusers==0.27.2 (required by the included flow-matching components)
# does `from huggingface_hub import cached_download`, which newer hub removed.
# Provide a backward-compatible alias so its import succeeds.
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
CKPT_ROOT = Path(os.environ.get("EX_OMNI_CKPT_ROOT", "/tmp/ex_omni_ckpt"))
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path(os.environ["EX_OMNI_TEMP_DIR"])
TEMP_DIR.mkdir(parents=True, exist_ok=True)

EX_OMNI_REPO = os.environ.get("EX_OMNI_MODEL_REPO")
GLM_DECODER_REPO = "zai-org/glm-4-voice-decoder"
WHISPER_REPO = "openai/whisper-large-v3"
QWEN3_REPO = "Qwen/Qwen3-0.6B"

HF_TOKEN = os.environ.get("HF_TOKEN")

EX_OMNI_DIR = CKPT_ROOT / "Ex-Omni"
GLM_DIR = CKPT_ROOT / "glm-4-voice-decoder"
WHISPER_DIR = CKPT_ROOT / "whisper-large-v3"
QWEN3_DIR = CKPT_ROOT / "Qwen3" / "Qwen3-0.6B"

TEMPLATE_PATH = "asset/EmoTalk.npz"

GPU_DURATION = int(os.environ.get("EX_OMNI_GPU_DURATION", "150"))


# --------------------------------------------------------------------------- #
# Download checkpoints (no CUDA touched here)
# --------------------------------------------------------------------------- #
def _download_all():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        required = (
            EX_OMNI_DIR / "config.json",
            GLM_DIR / "flow.pt",
            GLM_DIR / "hift.pt",
            WHISPER_DIR / "config.json",
            QWEN3_DIR / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing offline checkpoints: " + ", ".join(missing))
        print("[startup] using local checkpoints in offline mode.", flush=True)
        return
    if EX_OMNI_REPO:
        print("[startup] downloading model checkpoint ...", flush=True)
        snapshot_download(
            EX_OMNI_REPO,
            local_dir=str(EX_OMNI_DIR),
            token=HF_TOKEN,
            allow_patterns=["*.json", "*.safetensors", "*.pth", "*.jinja", "*.txt", "vocab.json", "*.model"],
        )
    elif not (EX_OMNI_DIR / "config.json").is_file():
        raise FileNotFoundError(
            "Provide the main checkpoint under EX_OMNI_CKPT_ROOT/Ex-Omni "
            "or set EX_OMNI_MODEL_REPO to a checkpoint repository."
        )
    print("[startup] downloading GLM-4 voice decoder ...", flush=True)
    snapshot_download(
        GLM_DECODER_REPO,
        local_dir=str(GLM_DIR),
        token=HF_TOKEN,
        allow_patterns=["*.pt", "*.yaml"],
    )
    print("[startup] downloading Whisper speech encoder (architecture + feature extractor) ...", flush=True)
    snapshot_download(
        WHISPER_REPO,
        local_dir=str(WHISPER_DIR),
        token=HF_TOKEN,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "vocab.json", "merges.txt"],
        ignore_patterns=["*fp32*", "model.safetensors.index.fp32.json"],
    )
    print("[startup] downloading Qwen3-0.6B (speech-generator backbone config) ...", flush=True)
    snapshot_download(
        QWEN3_REPO,
        local_dir=str(QWEN3_DIR),
        token=HF_TOKEN,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "vocab.json", "merges.txt", "tokenizer.json"],
    )
    print("[startup] all checkpoints ready.", flush=True)


_download_all()


# --------------------------------------------------------------------------- #
# Build runtime argument defaults
# --------------------------------------------------------------------------- #
def _build_args():
    a = types.SimpleNamespace()
    a.model_path = str(EX_OMNI_DIR)
    a.flow_ckpt_path = str(GLM_DIR / "flow.pt")
    a.hift_ckpt_path = str(GLM_DIR / "hift.pt")
    a.temperature = 0.2
    a.top_p = None
    a.num_beams = 1
    a.max_new_tokens = 128
    a.input_type = "mel"
    a.mel_size = 128
    a.s2s = True
    a.speech_generator_type = "ar"
    a.load_bf16 = True
    a.attn_implementation = "auto"
    a.save_blendshape = True
    a.max_history_length = 20
    a.auto_clear = True
    a.template_type = "emotalk"
    a.template_path = TEMPLATE_PATH
    # render params (EmoTalk path uses render_utils.setup_renderer defaults)
    a.render_width = 512
    a.render_height = 512
    a.render_fps = 30
    return a


ARGS = _build_args()


# --------------------------------------------------------------------------- #
# Resolve speech-module checkpoint paths before loading.
# --------------------------------------------------------------------------- #
import json  # noqa: E402

_cfg_path = EX_OMNI_DIR / "config.json"
_cfg = json.loads(_cfg_path.read_text())
_cfg["pretrain_speech_encoder_weights"] = str(WHISPER_DIR)
_cfg["pretrain_speech_generator_weights"] = str(QWEN3_DIR)
_cfg["pretrain_speech_embedding_path"] = str(GLM_DIR / "flow.pt")
_cfg_path.write_text(json.dumps(_cfg, indent=2))
print("[startup] patched config speech-module paths to local snapshots.", flush=True)

# Supply merges.txt from the compatible tokenizer snapshot when absent.
import shutil  # noqa: E402

_ex_merges = EX_OMNI_DIR / "merges.txt"
if not _ex_merges.exists() and (QWEN3_DIR / "merges.txt").exists():
    shutil.copy(QWEN3_DIR / "merges.txt", _ex_merges)
    print("[startup] merges.txt missing in Ex-Omni snapshot; copied from Qwen3-0.6B fallback.", flush=True)


# --------------------------------------------------------------------------- #
# Load the model + audio decoder + render assets at module scope.
# We load on CPU (device_map=None) then move to CUDA via the string "cuda"
# to keep the placement path explicit (avoids accelerate device_map="auto").
# --------------------------------------------------------------------------- #
print("[startup] loading Ex-Omni model (this streams ~22GB of weights) ...", flush=True)

from ex_omni.constants import SPEECH_TOKEN_INDEX  # noqa: E402
from ex_omni.flow_inference import AudioDecoder  # noqa: E402
from ex_omni.model.builder import load_model_for_inference  # noqa: E402
from ex_omni.render_utils import (  # noqa: E402
    apply_blendshapes_batch,
    create_blendshape_mapping,
    VideoStreamWriter,
    load_mesh_data,
    remap_weights,
    render_frames,
    setup_renderer,
)

tokenizer, model = load_model_for_inference(
    ARGS.model_path,
    load_bf16=True,
    device_map=None,          # load on CPU; move to cuda below
    device="cuda",
    attn_implementation="auto",
)
model.eval()
model.to("cuda")
print("[startup] Ex-Omni model on CUDA.", flush=True)

# tokenizer setup (mirrors deploy_base.MultiUserChatbot.__init__)
tokenizer.add_tokens(["<speech>"], special_tokens=True)
tokenizer.chat_template = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
    "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
SPEECH_TOKEN_ID = tokenizer.convert_tokens_to_ids("<speech>")
SYSTEM_MESSAGE = (
    "You are a multimodal assistant that understands both speech and text, "
    "and can respond using natural language or synthesized speech."
)

print("[startup] loading GLM-4 audio decoder ...", flush=True)
audio_decoder = AudioDecoder(
    config_path="./cosyvoice/vocab_16K.yaml",
    flow_ckpt_path=ARGS.flow_ckpt_path,
    hift_ckpt_path=ARGS.hift_ckpt_path,
    device="cuda",
    batch_cfg=True,
)
print("[startup] audio decoder ready.", flush=True)

print("[startup] loading render assets (EmoTalk template) ...", flush=True)
meanshape, faces, blendshape, TEMPLATE_BS_LIST = load_mesh_data(ARGS.template_path)
meanshape = meanshape.to("cuda")
faces = faces.to("cuda")
blendshape = blendshape.to("cuda")
mapping_indices = create_blendshape_mapping(TEMPLATE_BS_LIST, TEMPLATE_BS_LIST)
renderer = setup_renderer(meanshape, image_size=(ARGS.render_height, ARGS.render_width))
RENDER_BATCH_SIZE = max(1, int(os.environ.get("EX_OMNI_RENDER_BATCH_SIZE", "16")))
MAX_CONTEXT_TURNS = 8
MAX_CONTEXT_TOKENS = 2048
MAX_VISIBLE_TURNS = 20
print("[startup] render assets ready. Startup complete.", flush=True)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def _process_speech(speech_file):
    audio_array, sr = sf.read(speech_file, dtype="float32", always_2d=True)
    # audio_array: (frames, channels) -> mono (frames,)
    audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
    if audio_array is None or np.isnan(audio_array).any():
        raise ValueError("Error loading speech input.")
    speech_encoder = model.get_model().speech_encoder
    features = speech_encoder.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    )
    enc_dev = next(speech_encoder.parameters()).device
    enc_dtype = next(speech_encoder.parameters()).dtype
    input_features = features["input_features"].to(dtype=enc_dtype, device=enc_dev, non_blocking=True)
    return input_features, input_features.shape[-1]


def _render_video(bs_pred, audio_path, timestamp):
    bs_weights = remap_weights(bs_pred, mapping_indices)
    bs_weights = torch.from_numpy(bs_weights).float().to(meanshape.device)
    # NOTE: force `mouthClose` (index 26 in EmoTalk's 52-dim ARKit blendshape order)
    # to 0.0 during rendering. On this specific EmoTalk mesh template the
    # `mouthClose` shape looks visually broken (over-clamps the lips) even
    # for small predicted weights, so we zero it out to avoid the artefact.
    # The audio pipeline is unaffected; only the visual blendshape driving
    # the 3D avatar is edited here.
    bs_weights[:, 26] = 0.0
    video_path = str(TEMP_DIR / f"avatar_{timestamp}.mp4")
    with VideoStreamWriter(video_path, image_size=(ARGS.render_height, ARGS.render_width),
                           fps=ARGS.render_fps, audio_input=audio_path) as writer:
        for start in range(0, bs_weights.shape[0], RENDER_BATCH_SIZE):
            weight_batch = bs_weights[start:start + RENDER_BATCH_SIZE]
            deformed = apply_blendshapes_batch(meanshape, blendshape, weight_batch)
            images = render_frames(renderer, deformed, faces)
            writer.write(images.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy())
    return video_path


def _prompt_ids(model_history, current_message):
    # Only the current voice message has encoder features. Earlier voice turns
    # are retained as text markers plus the assistant's response.
    previous = list(model_history or [])[-2 * MAX_CONTEXT_TURNS:]
    while True:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            *previous,
            {"role": "user", "content": current_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(input_ids) <= MAX_CONTEXT_TOKENS or len(previous) < 2:
            return input_ids
        previous = previous[2:]


def _display_user_message(text_input, audio_input):
    if not audio_input:
        return text_input
    audio = gr.Audio(value=audio_input, autoplay=False)
    return [text_input, audio] if text_input else audio


def clear_conversation():
    return [], "", None, None, None, "Conversation cleared.", []


def generate(text_input, audio_input, temperature, max_new_tokens, render_video,
             chat_history, model_history):
    text_input = str(text_input or "").strip()
    if not text_input and not audio_input:
        raise gr.Error("Please enter a text prompt or provide a speech input.")

    displayed = list(chat_history or [])
    displayed.append({"role": "user", "content": _display_user_message(text_input, audio_input)})
    displayed = displayed[-(2 * MAX_VISIBLE_TURNS - 1):]
    previous = list(model_history or [])
    yield displayed, "", None, None, None, "Generating...", previous

    t0 = time.perf_counter()
    timestamp = int(time.time() * 1000)
    session_id = uuid.uuid4().hex[:8]

    # Build chat prompt
    if audio_input:
        gen_msg = f"<speech>\n{text_input}" if text_input else "<speech>"
    else:
        gen_msg = text_input
    input_id = _prompt_ids(model_history, gen_msg)
    if not input_id:
        raise gr.Error("Tokenizer returned empty input ids.")
    for idx, tid in enumerate(input_id):
        if tid == SPEECH_TOKEN_ID:
            input_id[idx] = SPEECH_TOKEN_INDEX

    model_device = next(model.parameters()).device
    input_ids = torch.tensor([input_id], dtype=torch.long, device=model_device)

    # Speech features
    if audio_input:
        speech_tensor, speech_len = _process_speech(audio_input)
        speech_lengths = torch.LongTensor([speech_len])
    else:
        speech_encoder = model.get_model().speech_encoder
        enc_dev = next(speech_encoder.parameters()).device
        enc_dtype = next(speech_encoder.parameters()).dtype
        speech_tensor = torch.zeros(1, ARGS.mel_size, 3000, dtype=enc_dtype, device=enc_dev)
        speech_lengths = torch.LongTensor([3000])

    do_sample = float(temperature) > 0
    gen_kwargs = {
        "speech": speech_tensor,
        "speech_lengths": speech_lengths,
        "do_sample": do_sample,
        "num_beams": 1,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "faster_infer": False,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    with torch.inference_mode():
        output_ids, output_units, bs_pred = model.generate(input_ids, **gen_kwargs)

    text_response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Speech synthesis
    audio_path = None
    if output_units is not None:
        try:
            units = [int(x) for x in str(output_units).split()]
            if units:
                tts_token = torch.tensor(units, device="cuda").unsqueeze(0)
                tts_speech = audio_decoder.offline_inference(tts_token)
                # tts_speech: (1, num_samples) -> soundfile wants (frames, channels)
                wav = tts_speech.detach().to(torch.float32).cpu().numpy().squeeze()
                audio_path = str(TEMP_DIR / f"speech_{session_id}_{timestamp}.wav")
                sf.write(audio_path, wav, samplerate=22050, subtype="PCM_16")
        except Exception:
            print("Speech synthesis failed:\n" + traceback.format_exc(), flush=True)
            audio_path = None

    # Avatar video
    video_path = None
    if render_video and bs_pred is not None:
        try:
            bs = bs_pred.squeeze(0).detach().to(torch.float32).cpu().numpy()
            if bs.shape[-1] == 51:
                bs = np.pad(bs, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
            bs = np.clip(bs, 0, 1)
            video_path = _render_video(bs, audio_path, timestamp)
        except Exception:
            print("Video rendering failed:\n" + traceback.format_exc(), flush=True)
            video_path = None

    elapsed = time.perf_counter() - t0
    status = f"Done in {elapsed:.1f}s  ·  {'speech+text' if audio_input else 'text'} input"
    if not text_response:
        text_response = "(no text response)"
    displayed.append({"role": "assistant", "content": text_response})
    prior_user_text = text_input
    if audio_input:
        prior_user_text = (f"{text_input}\n" if text_input else "") + "[Prior speech audio unavailable]"
    previous.extend([
        {"role": "user", "content": prior_user_text},
        {"role": "assistant", "content": text_response},
    ])
    yield (displayed[-2 * MAX_VISIBLE_TURNS:], "", None, audio_path, video_path,
           status, previous[-2 * MAX_CONTEXT_TURNS:])


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
.dark .gradio-container { color: var(--body-text-color); }
#result-video video { max-height: 300px !important; object-fit: contain; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Citrus(), title="Ex-Omni") as demo:
    gr.Markdown("# Ex-Omni")
    model_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=360,
                group_consecutive_messages=True,
            )
            text_input = gr.Textbox(
                label="Message",
                placeholder="e.g. Tell me a fun fact about the ocean.",
                lines=2,
            )
            audio_input = gr.Audio(
                label="Speech input (optional)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            render_video = gr.Checkbox(
                value=True,
                label="Render video",
            )
            with gr.Accordion("Advanced options", open=False):
                temperature = gr.Slider(
                    0.0, 1.0, value=0.2, step=0.05, label="Temperature (0 = greedy)"
                )
                max_new_tokens = gr.Slider(
                    16, 256, value=128, step=8, label="Max new tokens"
                )
            with gr.Row():
                run_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear conversation")

        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Speech response", type="filepath")
            video_output = gr.Video(label="Video", height=300, elem_id="result-video", autoplay=True)
            status_output = gr.Textbox(label="Status", interactive=False)

    inputs = [text_input, audio_input, temperature, max_new_tokens, render_video,
              chatbot, model_history]
    outputs = [chatbot, text_input, audio_input, audio_output, video_output,
               status_output, model_history]

    run_btn.click(fn=generate, inputs=inputs, outputs=outputs,
                  concurrency_id="ex-omni-model", concurrency_limit=1,
                  show_progress="hidden")
    text_input.submit(fn=generate, inputs=inputs, outputs=outputs,
                      concurrency_id="ex-omni-model", concurrency_limit=1,
                      show_progress="hidden")
    clear_btn.click(fn=clear_conversation, outputs=outputs, queue=False)

    gr.Examples(
        examples=[
            ["Tell me a fun fact about the ocean.", None, 0.2, 128, True],
            ["Introduce yourself in one sentence.", None, 0.2, 96, True],
            ["What's a good way to stay motivated while learning?", None, 0.4, 128, True],
        ],
        inputs=[text_input, audio_input, temperature, max_new_tokens, render_video],
        cache_examples=False,
        run_on_click=False,
    )


if __name__ == "__main__":
    # EX_OMNI_HOST / EX_OMNI_PORT override the default Gradio address.
    _launch_kwargs = {"show_error": True}
    _host = os.environ.get("EX_OMNI_HOST")
    _port = os.environ.get("EX_OMNI_PORT")
    if _host:
        _launch_kwargs["server_name"] = _host
    if _port:
        _launch_kwargs["server_port"] = int(_port)
    demo.queue(max_size=20).launch(**_launch_kwargs)
