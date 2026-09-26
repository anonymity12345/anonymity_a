import os
import torch
import torchaudio
import numpy as np
import re
from hyperpyyaml import load_hyperpyyaml
import uuid
import time
from contextlib import contextmanager, nullcontext
from collections import defaultdict


class StaticCUDAGraphEstimator(torch.nn.Module):
    """Capture the first Flow shape; use eager/compiled execution for other shapes.

    The first audio segment has a fixed ten-unit input, while full responses
    vary in length.  A single graph avoids unbounded graph memory and repeated
    captures for the full response.
    """

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self._signature = None
        self._static_args = None
        self._output = None
        self._graph = None
        self._disabled = False

    @staticmethod
    def _key(args):
        return tuple(None if value is None else
                     (tuple(value.shape), value.dtype, value.device)
                     for value in args)

    def forward(self, *args):
        if not args[0].is_cuda or torch.is_grad_enabled() or self._disabled:
            return self.estimator(*args)
        signature = self._key(args)
        if self._graph is None and self._signature is None:
            stream = None
            try:
                stream = torch.cuda.Stream(device=args[0].device)
                stream.wait_stream(torch.cuda.current_stream(args[0].device))
                with torch.cuda.stream(stream):
                    self._static_args = tuple(None if value is None else value.clone() for value in args)
                    for _ in range(2):
                        self.estimator(*self._static_args)
                    self._graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self._graph, stream=stream):
                        self._output = self.estimator(*self._static_args)
                torch.cuda.current_stream(args[0].device).wait_stream(stream)
                self._signature = signature
                print(f"[codec] captured Flow estimator CUDA graph for {signature[0][0]}", flush=True)
            except RuntimeError as exc:
                if stream is not None:
                    torch.cuda.current_stream(args[0].device).wait_stream(stream)
                self._disabled = True
                self._graph = None
                self._static_args = None
                print(f"[codec] CUDA graph unavailable; using estimator directly: {exc}", flush=True)
                return self.estimator(*args)
        if signature != self._signature:
            return self.estimator(*args)
        for static, value in zip(self._static_args, args):
            if static is not None:
                static.copy_(value)
        self._graph.replay()
        return self._output


@contextmanager
def _codec_stage(name, device, timings):
    if timings is None:
        yield
        return
    if torch.device(device).type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        yield
        end.record(torch.cuda.current_stream(device))
        timings[name] = (start, end)
    else:
        start = time.perf_counter()
        yield
        timings[name] = time.perf_counter() - start


def _resolve_hub_or_local_path(ckpt_ref):
    """Return a local filesystem path for ``ckpt_ref``.

    ``ckpt_ref`` may be:
      * an existing local file path -> returned unchanged.
      * ``"org/name/subpath/inside_repo.pt"`` -> downloaded via
        :func:`huggingface_hub.hf_hub_download` (repo split at the first two
        path segments, remainder becomes the file inside the repo).
      * ``"org/name:filename"`` -> same as above but with an explicit
        ``repo:filename`` delimiter, useful when the filename itself contains
        slashes (rare here but explicit).

    This keeps the CLI knobs (``--flow_ckpt_path`` / ``--hift_ckpt_path``)
    fully backward compatible with existing local-file usage while enabling
    zero-download deployments that point directly at
    ``zai-org/glm-4-voice-decoder/flow.pt``.
    """
    if not isinstance(ckpt_ref, str):
        return ckpt_ref
    if os.path.isfile(ckpt_ref):
        return ckpt_ref

    repo_id = None
    filename = None
    if ":" in ckpt_ref:
        repo_id, filename = ckpt_ref.split(":", 1)
    else:
        parts = ckpt_ref.split("/")
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])

    if not repo_id or not filename:
        # Not a HF-style reference — surface the original path so downstream
        # ``torch.load`` produces a clear "file not found" error.
        return ckpt_ref

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class AudioDecoder:
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path, device="cuda",
                 normalize_inference=True, batch_cfg=False, compile_estimator="none",
                 cuda_graph_estimator=False):
        self.device = device
        self.normalize_inference = normalize_inference
        if not normalize_inference and (batch_cfg or compile_estimator != "none" or cuda_graph_estimator):
            raise ValueError("Codec optimizations require normalized inference state")
        if cuda_graph_estimator and not batch_cfg:
            raise ValueError("Codec CUDA graph requires batched CFG so each estimator output is consumed before replay")

        with open(config_path, 'r') as f:
            self.scratch_configs = load_hyperpyyaml(f)

        # Load models. flow_ckpt_path / hift_ckpt_path may be either a local
        # path or an ``org/name/filename`` reference resolved on the fly via
        # huggingface_hub (see _resolve_hub_or_local_path).
        self.flow = self.scratch_configs['flow']
        self.flow.load_state_dict(
            torch.load(_resolve_hub_or_local_path(flow_ckpt_path), map_location=self.device)
        )
        self.hift = self.scratch_configs['hift']
        self.hift.load_state_dict(
            torch.load(_resolve_hub_or_local_path(hift_ckpt_path), map_location=self.device)
        )

        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        if normalize_inference:
            self.flow.eval()
            self.hift.eval()
        self.flow.decoder.local_noise_rng = normalize_inference
        self.flow.decoder.batch_cfg = batch_cfg
        self._eager_estimator = self.flow.decoder.estimator
        self._compiled_estimator = None
        if compile_estimator != "none":
            self._compiled_estimator = torch.compile(
                self._eager_estimator, mode=compile_estimator,
                fullgraph=True, dynamic=False,
            )
            self.flow.decoder.estimator = self._compiled_estimator
        if cuda_graph_estimator:
            estimator = StaticCUDAGraphEstimator(self.flow.decoder.estimator)
            self.flow.decoder.estimator = estimator
            self._compiled_estimator = estimator
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 1
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

    @torch.inference_mode()
    def token2wav(self, token, *args, **kwargs):
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        context = torch.random.fork_rng(devices=devices) if self.normalize_inference else nullcontext()
        with context:
            if self.normalize_inference:
                # HiFT also samples noise; keep codec RNG separate from Talker.
                torch.random.default_generator.manual_seed(42)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(42)
            return self._token2wav(token, *args, **kwargs)

    def _token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192),
                  finalize=False, timings=None):
        with _codec_stage("flow_s", self.device, timings):
            tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device))

        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            with _codec_stage("hift_s", self.device, timings):
                tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]

        else:
            with _codec_stage("hift_s", self.device, timings):
                tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            del self.hift_cache_dict[uuid]
            del self.mel_overlap_dict[uuid]
        return tts_speech, tts_mel

    def offline_inference(self, token, timing_callback=None, use_compiled=True):
        this_uuid = str(uuid.uuid1())
        timings = {} if timing_callback is not None else None
        previous_estimator = self.flow.decoder.estimator
        if not use_compiled and self._compiled_estimator is not None:
            self.flow.decoder.estimator = self._eager_estimator
        try:
            tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True, timings=timings)
        finally:
            self.flow.decoder.estimator = previous_estimator
        audio = tts_speech.cpu()
        if timing_callback is not None:
            result = {}
            for name, value in timings.items():
                if isinstance(value, tuple):
                    start, end = value
                    end.synchronize()
                    result[name] = start.elapsed_time(end) / 1000.0
                else:
                    result[name] = value
            timing_callback(result)
        return audio

    def stream_inference(self, token):
        token.to(self.device)
        this_uuid = str(uuid.uuid1())

        # Prepare other necessary input tensors
        llm_embedding = torch.zeros(1, 192).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)

        tts_speechs = []
        tts_mels = []

        block_size = self.flow.encoder.block_size
        prev_mel = None

        for idx in range(0, token.size(1), block_size):
            # if idx>block_size: break
            tts_token = token[:, idx:idx + block_size]

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                flow_prompt_speech_token = token[:, :idx]

            if idx + block_size >= token.size(-1):
                is_finalize = True
            else:
                is_finalize = False

            tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                 prompt_token=flow_prompt_speech_token.to(self.device),
                                                 prompt_feat=prompt_speech_feat.to(self.device), finalize=is_finalize)

            prev_mel = tts_mel
            prev_speech = tts_speech

            tts_speechs.append(tts_speech)
            tts_mels.append(tts_mel)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech.cpu()


    def token2mel(
        self,
        token,              
        uuid=None,
        prompt_token=None,  
        prompt_feat=None,    
        embedding=None,     
        prompt_token_len=None, 
        prompt_feat_len=None,  
        finalize=False,
    ):
        """
        返回:
        feat: [B, L_max, output_size]  生成的全段特征(含prompt段)
        feat_len: [B]  每条样本生成长度(含prompt段)
        prompt_feat_len: [B]  每条样本的prompt长度(若无prompt则为0)
        """
        device = self.device

        B = token.size(0)
        if prompt_token is None:
            prompt_token = torch.zeros(B, 0, dtype=token.dtype, device=token.device)
        if prompt_feat is None:
            prompt_feat = torch.zeros(B, 0, 80, dtype=torch.float32, device=token.device)
        if embedding is None:
            embedding = torch.zeros(B, 192, dtype=torch.float32, device=token.device)
        if prompt_token_len is None:
            prompt_token_len = torch.zeros(B, dtype=torch.int32, device=token.device)
        if prompt_feat_len is None:
            prompt_feat_len = torch.zeros(B, dtype=torch.int32, device=token.device)

        # 统一 dtype/device
        token = token.to(device)
        prompt_token = prompt_token.to(device)
        prompt_feat = prompt_feat.to(device)
        embedding = embedding.to(device)
        token_len = torch.tensor([token.shape[1]] * B, dtype=torch.int32, device=device) if not torch.is_tensor(token) \
            else torch.as_tensor(token_len if 'token_len' in locals() else [token.shape[1]] * B, dtype=torch.int32, device=device)

        # 实际推理
        return self.flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
        )


    def inference(
        self,
        token,                # [B, T_tok]
        token_len,            # [B]
        prompt_token,         # [B, T_ptok]
        prompt_token_len,     # [B]
        prompt_feat,          # [B, L_pfeat, 80]
        prompt_feat_len,      # [B]
        embedding,            # [B, 192]
    ):
        device = token.device
        B = token.size(0)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)   
        # concat text and prompt_text (batch-wise)
        token = torch.concat([prompt_token, token], dim=1)   
        token_len = prompt_token_len + token_len           

        # mask: True=valid
        pad_mask = ~make_pad_mask(token_len, T=token.size(1), device=device)  
        mask = pad_mask.float().unsqueeze(-1).to(embedding)                

        # embedding lookup
        token_int = torch.clamp(token, min=0)             
        token_emb = self.input_embedding(token_int)        
        token_emb = token_emb * mask                       
        # text encode
        h, h_lengths = self.encoder(token_emb, token_len)   
        h = self.encoder_proj(h)                             

        feat_len = ((token_len.to(torch.float32) / self.input_frame_rate) * (22050.0 / 256.0)).to(torch.int32)  # [B]

        h, h_lengths = self.length_regulator(h, feat_len)   

        L_max = int(feat_len.max().item())
        conds = torch.zeros([B, L_max, self.output_size], device=device)  
        if prompt_feat.size(1) != 0:
            for i in range(B):
                li = int(prompt_feat_len[i].item())
                if li > 0:
                    conds[i, :li, :] = prompt_feat[i, :li, :]

        conds = conds.transpose(1, 2)                      

        mask = ~make_pad_mask(feat_len, T=L_max, device=device) 
        mask = mask.unsqueeze(1)                                 

        # 解码
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),  
            mask=mask,                         
            spks=embedding,                    
            cond=conds,                      
            n_timesteps=10
        )                                        
        feat = feat.transpose(1, 2).contiguous() 

        return feat, feat_len, prompt_feat_len


def make_pad_mask(lengths: torch.Tensor, T: int = None, device=None):
    """
    lengths: [B] int32/int64
    T: optional, if None will use lengths.max()
    return: [B, T] bool, True = padded 位置
    """
    device = device if device is not None else lengths.device
    B = lengths.size(0)
    if T is None:
        T = int(lengths.max().item())
    arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
    # padded = positions >= length
    pad_mask = arange >= lengths.unsqueeze(1)  # True=pad
    return pad_mask
