"""Stage boundary contracts, also used by CPU correctness tests."""


def generation_hidden(latent, prompt_length, output_length):
    # Each generated ID is predicted by the preceding input's hidden state.
    # The first ID uses the LAST prompt row, not the first decode row.
    expected = prompt_length + output_length - 1
    if prompt_length < 1 or output_length < 1:
        raise ValueError("Empty prompt or generated sequence")
    if latent.ndim != 2 or latent.shape[0] != expected:
        raise ValueError(
            f"Latent alignment mismatch: shape={tuple(latent.shape)}, expected "
            f"{expected} rows (prompt={prompt_length}, output={output_length}). "
            "Check vLLM-Omni version, EOS retention and prefix caching."
        )
    return latent[prompt_length - 1:].unsqueeze(0)


def speech_budget(text_count, cap=750):
    if text_count < 1:
        raise ValueError("Talker requires at least one text token after dropping the last token")
    maximum = min(text_count * 20, cap)
    return min(text_count * 2, maximum), maximum


def stage_weight_name(name, stage):
    """Map checkpoint names to decoder and auxiliary modules; None means unused."""
    if stage == "thinker":
        if name.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.")):
            return name
        if name.startswith("model.speech_encoder.model.encoder."):
            return "speech_encoder." + name.removeprefix("model.speech_encoder.model.encoder.")
        if name.startswith("model.speech_projector."):
            return "speech_projector." + name.removeprefix("model.speech_projector.")
    elif stage == "talker":
        prefix = "model.speech_generator."
        if not name.startswith(prefix):
            return None
        name = name[len(prefix):]
        if name.startswith("llm.model.model.embed_tokens."):
            return "text_embedding." + name.removeprefix("llm.model.model.embed_tokens.")
        if name.startswith("llm.model.model."):
            return "model." + name.removeprefix("llm.model.model.")
        if name.startswith("speech_embedding."):
            return "model.embed_tokens." + name.removeprefix("speech_embedding.")
        if name.startswith(("input_proj.", "tqgf.", "llm_embedding.", "llm_decoder.")):
            return name
    else:
        raise ValueError(stage)
    return None


def thinker_to_talker(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
    from vllm_omni.inputs.data import OmniTokensPrompt

    if len(engine_input_source) != 1:
        raise ValueError("Talker needs exactly one Thinker source")
    outputs = stage_list[engine_input_source[0]].engine_outputs
    if not outputs:
        raise ValueError("Thinker produced no outputs")
    result = []
    for request in outputs:
        output = request.outputs[0]
        ids = list(output.token_ids)
        hidden = generation_hidden(
            output.multimodal_output["latent"], len(request.prompt_token_ids), len(ids)
        )
        speech_budget(len(ids) - 1)
        # The HF implementation drops the final ID even on length termination.
        # Prefix is SOS + text embeddings (N-1 rows) + task embedding.
        result.append(OmniTokensPrompt(
            prompt_token_ids=[0] * (len(ids) + 1),
            additional_information={
                "thinker_hidden": hidden.squeeze(0).float(),
                "text_ids": ids[:-1],
            },
        ))
    return result
